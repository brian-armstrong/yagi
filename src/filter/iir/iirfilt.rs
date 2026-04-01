use crate::error::{Error, Result};
use num_complex::{Complex32, ComplexFloat};
use core::f32;
use std::collections::VecDeque;

use crate::dotprod::DotProd;
use crate::filter::iir::design;
use crate::filter::iir::iirfiltsos::IirFilterSos;

// References:
//  [Pintelon:1990] Rik Pintelon and Johan Schoukens, "Real-Time
//      Integration and Differentiation of Analog Signals by Means of
//      Digital Filtering," IEEE Transactions on Instrumentation and
//      Measurement, vol 39 no. 6, December 1990.

/// IIR filter type (Normal or Second-Order Sections)
#[derive(Debug, Clone, Copy, PartialEq)]
enum IirFilterType {
    Norm,
    Sos,
}

/// Infinite impulse response (IIR) filter
#[derive(Debug, Clone)]
pub struct IirFilter<T, Coeff = T> {
    b: Vec<Coeff>,             // numerator (feed-forward coefficients)
    a: Vec<Coeff>,             // denominator (feed-back coefficients)
    v: VecDeque<T>,        // internal filter state (buffer)
    n: usize,               // filter length (order+1)
    nb: usize,              // numerator length
    na: usize,              // denominator length

    filter_type: IirFilterType,

    qsos: Vec<IirFilterSos<T, Coeff>>, // second-order sections filters

    scale: Coeff,              // output scaling factor
}

impl<T, Coeff> IirFilter<T, Coeff>
where
    T: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T>,
    Coeff: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<T, Output = T> + Into<Complex32>,
    VecDeque<T>: DotProd<Coeff, Output = T>,
    f32: Into<Coeff>,
{
    /// Create a new IIR filter from a numerator and denominator
    /// 
    /// # Notes
    /// 
    /// The number of feed-forward and feed-back coefficients do not need to be equal, but they do need to be non-zero.
    /// Furthermore, the first feed-back coefficient \(a_0\) cannot be equal to zero, otherwise the filter will 
    /// be invalid as this value is factored out from all coefficients.
    /// For stability reasons the number of coefficients should reasonably not exceed about 8 for single-precision 
    /// floating-point.
    /// 
    /// # Arguments
    /// 
    /// * `b` - The numerator coefficients
    /// * `a` - The denominator coefficients
    /// 
    /// # Returns
    /// 
    /// A new IIR filter
    pub fn new(b: &[Coeff], a: &[Coeff]) -> Result<Self> {
        if b.is_empty() {
            return Err(Error::Config("numerator length cannot be zero".into()));
        }
        if a.is_empty() {
            return Err(Error::Config("denominator length cannot be zero".into()));
        }

        let nb = b.len();
        let na = a.len();
        let n = na.max(nb);

        let mut filter = IirFilter {
            b: b.to_vec(),
            a: a.to_vec(),
            v: VecDeque::from(vec![T::default(); n]),
            n,
            nb,
            na,
            filter_type: IirFilterType::Norm,
            qsos: Vec::new(),
            scale: Coeff::one(),
        };

        // normalize coefficients to a[0]
        let a0 = filter.a[0];
        for i in 0..filter.nb {
            filter.b[i] = filter.b[i] / a0;
        }
        for i in 0..filter.na {
            filter.a[i] = filter.a[i] / a0;
        }

        filter.reset();
        Ok(filter)
    }

    // create iirfilt object based on second-order sections form
    // _B      :   numerator, feed-forward coefficients [size: _nsos x 3]
    // _A      :   denominator, feed-back coefficients  [size: _nsos x 3]
    // _nsos   :   number of second-order sections
    // NOTE: The number of second-order sections can be computed from the
    // filter's order, n, as such:
    //   r = n % 2
    //   L = (n-r)/2
    //   nsos = L+r
    pub fn new_sos(b: &[Coeff], a: &[Coeff], nsos: usize) -> Result<Self> {
        if nsos == 0 {
            return Err(Error::Config("filter must have at least one 2nd-order section".into()));
        }

        let mut filter = IirFilter::<T, Coeff> {
            b: b.to_vec(),
            a: a.to_vec(),
            v: VecDeque::new(),
            n: nsos * 2,
            nb: 0,
            na: 0,
            filter_type: IirFilterType::Sos,
            qsos: Vec::with_capacity(nsos),
            scale: Coeff::one(),
        };

        for i in 0..nsos {
            let bt = [b[3*i], b[3*i+1], b[3*i+2]];
            let at = [a[3*i], a[3*i+1], a[3*i+2]];
            filter.qsos.push(IirFilterSos::<T, Coeff>::new(&bt, &at)?);
        }

        filter.set_scale(Coeff::one());
        Ok(filter)
    }

    // create iirfilt (infinite impulse response filter) object based
    // on prototype
    //  _ftype      :   filter type (e.g. LIQUID_IIRDES_BUTTER)
    //  _btype      :   band type (e.g. LIQUID_IIRDES_BANDPASS)
    //  _format     :   coefficients format (e.g. LIQUID_IIRDES_SOS)
    //  _order      :   filter order
    //  _fc         :   low-pass prototype cut-off frequency
    //  _f0         :   center frequency (band-pass, band-stop)
    //  _ap         :   pass-band ripple in dB
    //  _as         :   stop-band ripple in dB
    pub fn new_prototype(
        ftype: design::IirFilterShape,
        btype: design::IirBandType,
        format: design::IirFormat,
        order: usize,
        fc: f32,
        f0: f32,
        ap: f32,
        as_: f32,
    ) -> Result<Self> {
        // filter length
        let mut n = order;

        if btype == design::IirBandType::Bandpass || btype == design::IirBandType::Bandstop {
            n *= 2;
        }

        let r = n % 2; // odd/even order
        let l = (n - r) / 2; // filter semi-length

        let h_len = if format == design::IirFormat::SecondOrderSections { 3*(l+r) } else { n+1 };
        let mut b = vec![0.0; h_len];
        let mut a = vec![0.0; h_len];

        design::iir_design(ftype, btype, format, order, fc, f0, ap, as_, &mut b, &mut a)?;

        let b = b.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
        let a = a.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

        if format == design::IirFormat::SecondOrderSections {
            let filter = IirFilter::<T, Coeff>::new_sos(&b, &a, l+r)?;
            return Ok(filter);
        }

        let filter = IirFilter::<T, Coeff>::new(&b, &a)?;
        Ok(filter)
    }

    // create simplified low-pass Butterworth IIR filter
    //  _n      : filter order
    //  _fc     : low-pass prototype cut-off frequency
    pub fn new_lowpass(order: usize, fc: f32) -> Result<Self> {
        let filter = IirFilter::<T, Coeff>::new_prototype(
            design::IirFilterShape::Butter,
            design::IirBandType::Lowpass,
            design::IirFormat::SecondOrderSections,
            order,
            fc,
            0.0,
            0.1,
            60.0,
        )?;
        Ok(filter)
    }

    // create 8th-order integrating filter
    pub fn new_integrator() -> Result<Self> {
        // integrator digital zeros/poles/gain, [Pintelon:1990] Table II
        //
        // zeros, digital, integrator
        let zdi = [
            Complex32::from(1.175839 * -1.0),
            3.371020 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -125.1125),
            3.371020 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  125.1125),
            4.549710 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -80.96404),
            4.549710 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   80.96404),
            5.223966 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -40.09347),
            5.223966 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   40.09347),
            Complex32::from(5.443743),
        ];
        // poles, digital, integrator
        let pdi = [
            Complex32::from(0.5805235 * -1.0),
            0.2332021 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -114.0968),
            0.2332021 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  114.0968),
            0.1814755 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -66.33969),
            0.1814755 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   66.33969),
            0.1641457 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -21.89539),
            0.1641457 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   21.89539),
            Complex32::from(1.0),
        ];
        // gain, digital, integrator (slight adjustment added for proper gain)
        let kdi = Complex32::from(-1.89213380759321e-05 / 0.9695401191711425781);

        // second-order sections
        let mut b = vec![0.0; 12];
        let mut a = vec![0.0; 12];
        design::iir_design_d2sos(&zdi, &pdi, 8, kdi, &mut b, &mut a)?;

        let b = b.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
        let a = a.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

        let filter = IirFilter::<T, Coeff>::new_sos(&b, &a, 4)?;
        Ok(filter)
    }

    // create 8th-order differentiating filter
    pub fn new_differentiator() -> Result<Self> {
        // differentiator digital zeros/poles/gain, [Pintelon:1990] Table IV
        //
        // zeros, digital, differentiator
        let zdd = [
            Complex32::from(1.702575 * -1.0),
            5.877385 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -221.4063),
            5.877385 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  221.4063),
            4.197421 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -144.5972),
            4.197421 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  144.5972),
            5.350284 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -66.88802),
            5.350284 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   66.88802),
            Complex32::from(1.0),
        ];
        // poles, digital, differentiator
        let pdd = [
            Complex32::from(0.8476936 * -1.0),
            0.2990781 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 * -125.5188),
            0.2990781 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  125.5188),
            0.2232427 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -81.52326),
            0.2232427 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   81.52326),
            0.1958670 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *  -40.51510),
            0.1958670 * Complex32::from_polar(1.0, f32::consts::PI / 180.0 *   40.51510),
            Complex32::from(0.1886088),
        ];
        // gain, digital, differentiator (slight adjustment added for proper gain)
        let kdd = Complex32::from(2.09049284907492e-05 / 1.033477783203125000);

        // second-order sections
        let mut b = vec![0.0; 12];
        let mut a = vec![0.0; 12];
        design::iir_design_d2sos(&zdd, &pdd, 8, kdd, &mut b, &mut a)?;

        let b = b.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
        let a = a.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

        let filter = IirFilter::<T, Coeff>::new_sos(&b, &a, 4)?;
        Ok(filter)
    }   

    // create DC-blocking filter
    //
    //          1 -          z^-1
    //  H(z) = ------------------
    //          1 - (1-alpha)z^-1
    pub fn new_dc_blocker(alpha: f32) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(Error::Config("DC-blocking filter bandwidth must be greater than zero".into()));
        }

        let bf = [1.0, -1.0];
        let af = [1.0, -1.0 + alpha];

        let b = bf.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
        let a = af.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

        let mut filter = IirFilter::<T, Coeff>::new(&b, &a)?;
        filter.set_scale(Coeff::from((1.0 - alpha).sqrt()).unwrap());
        Ok(filter)
    }

    // create phase-locked loop iirfilt object
    //  _w      :   filter bandwidth
    //  _zeta   :   damping factor (1/sqrt(2) suggested)
    //  _K      :   loop gain (1000 suggested)
    pub fn new_pll(w: f32, zeta: f32, k: f32) -> Result<Self> {
        if w <= 0.0 || w >= 1.0 {
            return Err(Error::Config("PLL bandwidth must be in (0,1)".into()));
        }
        if zeta <= 0.0 || zeta >= 1.0 {
            return Err(Error::Config("PLL damping factor must be in (0,1)".into()));
        }
        if k <= 0.0 {
            return Err(Error::Config("PLL loop gain must be greater than zero".into()));
        }

        let mut bf = [0.0; 3];
        let mut af = [0.0; 3];
        design::iir_design_pll_active_lag(w, zeta, k, &mut bf, &mut af)?;

        let b = bf.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
        let a = af.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();

        let filter = IirFilter::<T, Coeff>::new_sos(&b, &a, 1)?;
        Ok(filter)
    }

    // reset internal state of iirfilt object
    pub fn reset(&mut self) {
        if self.filter_type == IirFilterType::Sos {
            for sos in &mut self.qsos {
                sos.reset();
            }
        } else {
            for v in &mut self.v {
                *v = T::default();
            }
        }
    }

    // set output scaling for filter
    pub fn set_scale(&mut self, scale: Coeff) {
        self.scale = scale;
    }

    // get output scaling for filter
    pub fn get_scale(&self) -> Coeff {
        self.scale
    }

    // set coefficients for filter
    pub fn set_coefficients(&mut self, b: &[Coeff], a: &[Coeff]) -> Result<()> {
        if self.filter_type == IirFilterType::Sos && self.qsos.len() == 1 {
            return self.qsos[0].set_coefficients(&[b[0], b[1], b[2]], &[a[0], a[1], a[2]]);
        }

        if b.len() != self.nb || a.len() != self.na {
            return Err(Error::Config("coefficient vector lengths must match filter order".into()));
        }

        self.b = b.to_vec();
        self.a = a.to_vec();

        let a0 = self.a[0];
        for i in 0..self.nb {
            self.b[i] = self.b[i] / a0;
        }
        for i in 0..self.na {
            self.a[i] = self.a[i] / a0;
        }
        Ok(())
    }

    // get coefficients for filter
    pub fn get_coeffs(&self) -> (Vec<Coeff>, Vec<Coeff>) {
        (self.b.clone(), self.a.clone())
    }

    // execute normal iir filter using traditional numerator/denominator
    // form (not second-order sections form)
    //  _x      :   input sample
    //  _y      :   output sample
    fn execute_norm(&mut self, x: T) -> T {
        // advance buffer

        // cheap vecdeque rotate
        self.v.rotate_right(1);
        self.v[0] = T::default();

        // compute new v[0]
        let v0 = self.v.dotprod(&self.a);
        self.v[0] = x - v0;

        let y = self.v.dotprod(&self.b);
        y * self.scale
    }

    // execute filter using second-order sections form
    //  _x      :   input sample
    //  _y      :   output sample
    fn execute_sos(&mut self, x: T) -> T {
        let mut y: T = x;
        for sos in &mut self.qsos {
            y = sos.execute(y);
        }
        y * self.scale
    }

    // execute iir filter, switching to type-specific function
    //  _x      :   input sample
    //  _y      :   output sample
    pub fn execute(&mut self, x: T) -> T {
        match self.filter_type {
            IirFilterType::Norm => self.execute_norm(x),
            IirFilterType::Sos => self.execute_sos(x),
        }
    }

    // execute filter block
    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if x.len() != y.len() {
            return Err(Error::Config("input and output block lengths must be equal".into()));
        }

        match self.filter_type {
            IirFilterType::Norm => {
                for (x_s, y_s) in x.iter().zip(y.iter_mut()) {
                    *y_s = self.execute_norm(*x_s);
                }
            }
            IirFilterType::Sos => {
                for (x_s, y_s) in x.iter().zip(y.iter_mut()) {
                    *y_s = self.execute_sos(*x_s);
                }
            }
        }

        Ok(())
    }

    // get filter length (order + 1)
    pub fn get_length(&self) -> usize {
        self.n
    }

    // compute complex frequency response
    //  _fc     :   frequency
    //  _H      :   output frequency response
    pub fn freqresponse(&self, fc: f32) -> Complex32 {
        let mut h;

        if self.filter_type == IirFilterType::Norm {
            let mut hb = Complex32::default();
            let mut ha = Complex32::default();

            for i in 0..self.nb {
                hb += self.b[i].into() * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * i as f32);
            }

            for i in 0..self.na {
                ha += self.a[i].into() * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * i as f32);
            }

            // TODO : check to see if we need to take conjugate
            h = hb / ha;
        } else {
            h = Complex32::from(1.0);

            for i in 0..self.qsos.len() {
                let hb = self.b[3*i].into() * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * 0.0) +
                         self.b[3*i+1].into() * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * 1.0) +
                         self.b[3*i+2].into() * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * 2.0);

                let ha = self.a[3*i].into() * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * 0.0) +
                         self.a[3*i+1].into() * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * 1.0) +
                         self.a[3*i+2].into() * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * 2.0);

                h *= hb / ha;
            }
        }

        h * self.scale.into()
    }

    // compute power spectral density response of filter object in dB
    pub fn get_psd(&self, fc: f32) -> f32 {
        let h = self.freqresponse(fc);
        10.0 * (h * h.conj()).re.log10()
    }

    // compute group delay in samples
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        let mut groupdelay = 0.0;

        if self.filter_type == IirFilterType::Norm {
            let mut b = vec![0.0; self.nb];
            let mut a = vec![0.0; self.na];
            for i in 0..self.nb {
                b[i] = self.b[i].re();
            }
            for i in 0..self.na {
                a[i] = self.a[i].re();
            }
            groupdelay = design::iir_group_delay(&b, &a, fc)?;
        } else {
            for i in 0..self.qsos.len() {
                groupdelay += self.qsos[i].groupdelay(fc)? - 2.0;
            }
        }
        
        Ok(groupdelay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use design::{IirBandType, IirFilterShape, IirFormat};
    use test_macro::autotest_annotate;
    use crate::fft::spgram::Spgram;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_spectrum};
    use crate::random::randnf;
    use approx::assert_relative_eq;
    
    #[test]
    #[autotest_annotate(autotest_iirfilt_integrator)]
    fn test_iirfilt_integrator() {
        // options
        let num_ones = 10;
        let num_samples = 40;
    
        // allocate memory for data arrays
        let mut buf_0 = vec![0.0f32; num_samples]; // filter input
        let mut buf_1 = vec![0.0f32; num_samples]; // filter output
    
        // generate input signal
        for i in 0..num_samples {
            buf_0[i] = if i < num_ones { 1.0 } else { 0.0 };
        }
    
        // create integrator and run on sample data
        let mut q = IirFilter::<f32, f32>::new_integrator().unwrap();
        q.execute_block(&buf_0, &mut buf_1).unwrap();
    
        // check that last value matches expected
        assert_relative_eq!(buf_1[num_samples - 1], num_ones as f32, epsilon = 0.01);
    }
    
    #[test]
    #[autotest_annotate(autotest_iirfilt_differentiator)]
    fn test_iirfilt_differentiator() {
        // options
        let num_samples = 400;
    
        // allocate memory for data arrays
        let mut buf_0 = vec![0.0f32; num_samples]; // filter input
        let mut buf_1 = vec![0.0f32; num_samples]; // filter output
    
        // generate input signal
        for i in 0..num_samples {
            buf_0[i] = i as f32;
        }
    
        // create differentiator and run on sample data
        let mut q = IirFilter::<f32, f32>::new_differentiator().unwrap();
        q.execute_block(&buf_0, &mut buf_1).unwrap();
    
        // check that derivative is equal to 1
        assert_relative_eq!(buf_1[num_samples - 1], 1.0f32, epsilon = 0.01);
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_dcblock)]
    fn test_iirfilt_dcblock() {
        // options
        let n = 400000; // number of output samples to analyze
        let alpha = 0.2f32; // forgetting factor
        let nfft = 1200; // number of bins in transform
        let tol = 0.7f32; // error tolerance [dB]

        // create base object
        let mut filter = IirFilter::<Complex32, f32>::new_dc_blocker(alpha).unwrap();

        // create and configure objects
        let mut q = Spgram::new(nfft, crate::math::WindowType::Hann, nfft/2, nfft/4).unwrap();

        // start running input through filter
        for _ in 0..n {
            // generate noise input
            let x = Complex32::new(randnf(), randnf()) * f32::consts::FRAC_1_SQRT_2;

            // apply filter
            let y = filter.execute(x);
            // run samples through the spgram object
            q.push(y);
        }

        // verify result
        let psd = q.get_psd();
        let regions = [
            PsdRegion { fmin: -0.500, fmax: -0.200, pmin: -tol, pmax: tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.002, fmax: 0.002, pmin: -tol, pmax: -20.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: 0.200, fmax: 0.500, pmin: -tol, pmax: tol, test_lo: true, test_hi: true },
        ];
        assert!(validate_psd_spectrum(&psd, nfft, &regions).unwrap());
    }

    fn testbench_iirfilt_copy(format: IirFormat) {
        // create base object
        let mut q0 = IirFilter::<Complex32, f32>::new_prototype(
            IirFilterShape::Ellip,
            IirBandType::Lowpass,
            format,
            9,
            0.2,
            0.0,
            0.1,
            60.0
        ).unwrap();

        // start running input through filter
        let num_samples = 80;
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let _y0 = q0.execute(v);
        }

        // copy filter
        let mut q1 = q0.clone();

        // continue running through both filters
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let y0 = q0.execute(v);
            let y1 = q1.execute(v);

            // compare result
            assert_eq!(y0, y1);
        }

        // Rust automatically handles destruction of objects when they go out of scope
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_copy_tf)]
    fn test_iirfilt_copy_tf() {
        testbench_iirfilt_copy(IirFormat::TransferFunction);
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_copy_sos)]
    fn test_iirfilt_copy_sos() {
        testbench_iirfilt_copy(IirFormat::SecondOrderSections);
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_config)]
    fn test_iirfilt_config() {
        // test copying/creating invalid objects
        assert!(IirFilter::<Complex32, f32>::new(&[], &[]).is_err());
        assert!(IirFilter::<Complex32, f32>::new(&[0.0], &[]).is_err());
        assert!(IirFilter::<Complex32, f32>::new(&[], &[1.0]).is_err());
        assert!(IirFilter::<Complex32, f32>::new_sos(&[], &[], 0).is_err());

        // create valid object
        let mut filter = IirFilter::<Complex32, f32>::new_lowpass(7, 0.1).unwrap();

        // check properties
        filter.set_scale(7.22);
        let scale = filter.get_scale();
        assert_eq!(scale, 7.22);
        assert_eq!(filter.get_length(), 8); // 7+1
        // Rust automatically handles destruction of objects when they go out of scope
    }

    #[test]
    #[autotest_annotate(autotest_iir_groupdelay_n3)]
    fn test_iir_groupdelay_n3() {
        // create coefficients array
        let b = [0.20657210, 0.41314420, 0.20657210];
        let a = [1.00000000, -0.36952737, 0.19581573];

        let tol = 1e-3f32;

        // create testing vectors
        let fc = [0.000, 0.125, 0.250, 0.375];
        let g0 = [0.973248939389634, 1.366481121240365, 1.227756735863196, 0.651058521306726];

        // run tests
        for i in 0..4 {
            let g = design::iir_group_delay(&b, &a, fc[i]).unwrap();
            assert_relative_eq!(g, g0[i], epsilon = tol);
        }

        // create filter
        let filter = IirFilter::<f32, f32>::new(&b, &a).unwrap();

        // run tests again
        for i in 0..4 {
            let g = filter.groupdelay(fc[i]).unwrap();
            assert_relative_eq!(g, g0[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iir_groupdelay_n8)]
    fn test_iir_groupdelay_n8() {
        // create coefficients arrays (7th-order Butterworth)
        let b = [
            0.00484212,
            0.03389481,
            0.10168444,
            0.16947407,
            0.16947407,
            0.10168444,
            0.03389481,
            0.00484212,
        ];

        let a = [
             1.00000000,
            -1.38928008,
             1.67502367,
            -1.05389738,
             0.50855154,
            -0.14482945,
             0.02625222,
            -0.00202968,
        ];

        let tol = 1e-3f32;

        // create testing vectors
        let fc = [
            0.00000,
            0.06250,
            0.12500,
            0.18750,
            0.25000,
            0.31250,
            0.37500,
        ];

        let g0 = [
            3.09280801068444,
            3.30599360247944,
            4.18341028373046,
            7.71934054380586,
            4.34330109915390,
            2.60203085226210,
            1.97868452107144,
        ];

        // run tests
        for i in 0..7 {
            let g = design::iir_group_delay(&b, &a, fc[i]).unwrap();
            assert_relative_eq!(g, g0[i], epsilon = tol);
        }

        //
        // test with iir filter (tf)
        //

        // create filter
        let filter = IirFilter::<f32, f32>::new(&b, &a).unwrap();

        // run tests again
        for i in 0..7 {
            let g = filter.groupdelay(fc[i]).unwrap();
            assert_relative_eq!(g, g0[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iir_groupdelay_sos_n8)]
    fn test_iir_groupdelay_sos_n8() {
        // create coefficients arrays (7th-order Butterworth)
        let b = [
            0.00484212, 0.00968423, 0.00484212,
            1.00000000, 2.00000000, 1.00000000,
            1.00000000, 2.00000000, 1.00000000,
            1.00000000, 1.00000000, 0.00000000,
        ];

        let a = [
            1.00000000, -0.33283597, 0.07707999,
            1.00000000, -0.38797498, 0.25551325,
            1.00000000, -0.51008475, 0.65066898,
            1.00000000, -0.15838444, 0.00000000,
        ];

        let tol = 1e-3f32;

        // create testing vectors
        let fc = [
            0.00000,
            0.06250,
            0.12500,
            0.18750,
            0.25000,
            0.31250,
            0.37500,
        ];

        let g0 = [
            3.09280801068444,
            3.30599360247944,
            4.18341028373046,
            7.71934054380586,
            4.34330109915390,
            2.60203085226210,
            1.97868452107144,
        ];

        // create filter
        let filter = IirFilter::<f32, f32>::new_sos(&b, &a, 4).unwrap();

        // run tests
        for i in 0..7 {
            let g = filter.groupdelay(fc[i]).unwrap();
            assert_relative_eq!(g, g0[i], epsilon = tol);
        }
    }

    include!("test_data.rs");

    // autotest helper function
    //  _b      :   filter coefficients (numerator)
    //  _a      :   filter coefficients (denominator)
    //  _h_len  :   filter coefficients length
    //  _x      :   input array
    //  _x_len  :   input array length
    //  _y      :   output array
    //  _y_len  :   output array length
    fn iirfilt_rrrf_test(
        b: &[f32],
        a: &[f32],
        x: &[f32],
        y: &[f32],
    ) -> () {
        let tol = 0.001f32;
    
        // load filter coefficients externally
        let mut q = IirFilter::<f32, f32>::new(b, a).unwrap();
    
        // allocate memory for output
        let mut y_test = vec![0.0; y.len()];
    
        // compute output
        q.execute_block(x, &mut y_test).unwrap();
    
        // Compare results
        for (y_i, y_test_i) in y.iter().zip(y_test.iter()) {
            assert_relative_eq!(*y_i, *y_test_i, epsilon = tol);
        }
    }
    
    #[test]
    #[autotest_annotate(autotest_iirfilt_rrrf_h3x64)]
    fn test_iirfilt_rrrf_h3x64() {
        iirfilt_rrrf_test(
            &IIRFILT_RRRF_DATA_H3X64_B,
            &IIRFILT_RRRF_DATA_H3X64_A,
            &IIRFILT_RRRF_DATA_H3X64_X,
            &IIRFILT_RRRF_DATA_H3X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_rrrf_h5x64)]
    fn test_iirfilt_rrrf_h5x64() {
        iirfilt_rrrf_test(
            &IIRFILT_RRRF_DATA_H5X64_B,
            &IIRFILT_RRRF_DATA_H5X64_A,
            &IIRFILT_RRRF_DATA_H5X64_X,
            &IIRFILT_RRRF_DATA_H5X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_rrrf_h7x64)]
    fn test_iirfilt_rrrf_h7x64() {
        iirfilt_rrrf_test(
            &IIRFILT_RRRF_DATA_H7X64_B,
            &IIRFILT_RRRF_DATA_H7X64_A,
            &IIRFILT_RRRF_DATA_H7X64_X,
            &IIRFILT_RRRF_DATA_H7X64_Y,
        );
    }

    // autotest helper function
    //  _b      :   filter coefficients (numerator)
    //  _a      :   filter coefficients (denominator)
    //  _h_len  :   filter coefficients length
    //  _x      :   input array
    //  _x_len  :   input array length
    //  _y      :   output array
    //  _y_len  :   output array length
    fn iirfilt_crcf_test(
        b: &[f32],
        a: &[f32],
        x: &[Complex32],
        y: &[Complex32],
    ) -> () {
        let tol = 0.001f32;
    
        // load filter coefficients externally
        let mut q = IirFilter::<Complex32, f32>::new(b, a).unwrap();
    
        // allocate memory for output
        let mut y_test = vec![Complex32::new(0.0, 0.0); y.len()];
    
        // compute output
        q.execute_block(x, &mut y_test).unwrap();
    
        // Compare results
        for (y_i, y_test_i) in y.iter().zip(y_test.iter()) {
            assert_relative_eq!(y_i.re, y_test_i.re, epsilon = tol);
            assert_relative_eq!(y_i.im, y_test_i.im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_crcf_h3x64)]
    fn test_iirfilt_crcf_h3x64() {
        iirfilt_crcf_test(
            &IIRFILT_CRCF_DATA_H3X64_B,
            &IIRFILT_CRCF_DATA_H3X64_A,
            &IIRFILT_CRCF_DATA_H3X64_X,
            &IIRFILT_CRCF_DATA_H3X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_crcf_h5x64)]
    fn test_iirfilt_crcf_h5x64() {
        iirfilt_crcf_test(
            &IIRFILT_CRCF_DATA_H5X64_B,
            &IIRFILT_CRCF_DATA_H5X64_A,
            &IIRFILT_CRCF_DATA_H5X64_X,
            &IIRFILT_CRCF_DATA_H5X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_crcf_h7x64)]
    fn test_iirfilt_crcf_h7x64() {
        iirfilt_crcf_test(
            &IIRFILT_CRCF_DATA_H7X64_B,
            &IIRFILT_CRCF_DATA_H7X64_A,
            &IIRFILT_CRCF_DATA_H7X64_X,
            &IIRFILT_CRCF_DATA_H7X64_Y,
        );
    }
    
    // autotest helper function
    //  _b      :   filter coefficients (numerator)
    //  _a      :   filter coefficients (denominator)
    //  _h_len  :   filter coefficients length
    //  _x      :   input array
    //  _x_len  :   input array length
    //  _y      :   output array
    //  _y_len  :   output array length
    fn iirfilt_cccf_test(
        b: &[Complex32],
        a: &[Complex32],
        x: &[Complex32],
        y: &[Complex32],
    ) -> () {
        let tol = 0.001f32;
    
        // load filter coefficients externally
        let mut q = IirFilter::<Complex32, Complex32>::new(b, a).unwrap();
    
        // allocate memory for output
        let mut y_test = vec![Complex32::new(0.0, 0.0); y.len()];
    
        // compute output
        q.execute_block(x, &mut y_test).unwrap();
    
        // Compare results
        for (y_i, y_test_i) in y.iter().zip(y_test.iter()) {
            assert_relative_eq!(y_i.re, y_test_i.re, epsilon = tol);
            assert_relative_eq!(y_i.im, y_test_i.im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_cccf_h3x64)]
    fn test_iirfilt_cccf_h3x64() {
        iirfilt_cccf_test(
            &IIRFILT_CCCF_DATA_H3X64_B,
            &IIRFILT_CCCF_DATA_H3X64_A,
            &IIRFILT_CCCF_DATA_H3X64_X,
            &IIRFILT_CCCF_DATA_H3X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_cccf_h5x64)]
    fn test_iirfilt_cccf_h5x64() {
        iirfilt_cccf_test(
            &IIRFILT_CCCF_DATA_H5X64_B,
            &IIRFILT_CCCF_DATA_H5X64_A,
            &IIRFILT_CCCF_DATA_H5X64_X,
            &IIRFILT_CCCF_DATA_H5X64_Y,
        );
    }

    #[test]
    #[autotest_annotate(autotest_iirfilt_cccf_h7x64)]
    fn test_iirfilt_cccf_h7x64() {
        iirfilt_cccf_test(
            &IIRFILT_CCCF_DATA_H7X64_B,
            &IIRFILT_CCCF_DATA_H7X64_A,
            &IIRFILT_CCCF_DATA_H7X64_X,
            &IIRFILT_CCCF_DATA_H7X64_Y,
        );
    }
}