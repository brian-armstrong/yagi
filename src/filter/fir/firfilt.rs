use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter;
use crate::buffer::Window;
use std::f32::consts::PI;
use num_complex::{ComplexFloat, Complex32};

/// create coefficients for a Kaiser-Bessel windowed sinc filter
///
/// # Arguments
///
/// * `n` - filter length
/// * `fc` - cutoff frequency
/// * `as_` - stop-band attenuation
/// * `mu` - fractional sample offset
///
/// # Returns
///
/// Vec of filter coefficients
pub fn fir_filter_design_kaiser<Coeff>(n: usize, fc: f32, as_: f32, mu: f32) -> Result<Vec<Coeff>>
where Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    let h = filter::fir_design_kaiser(n, fc, as_, mu)?;
    let h_c = h.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    Ok(h_c)
}

/// create coefficients for a square-root Nyquist prototype
///
/// # Arguments
///
/// * `ftype` - filter type
/// * `k` - nominal samples/symbol
/// * `m` - filter delay
/// * `beta` - rolloff factor
/// * `mu` - fractional sample offset
///
/// # Returns
///
/// Vec of filter coefficients
pub fn fir_filter_design_rnyquist<Coeff>(ftype: filter::FirFilterShape, k: usize, m: usize, beta: f32, mu: f32) -> Result<Vec<Coeff>>
where Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    let h = filter::fir_design_prototype(ftype, k, m, beta, mu)?;
    let h_c = h.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    Ok(h_c)
}

/// create coefficients for a Parks-McClellan algorithm
///
/// # Arguments
///
/// * `h_len` - filter length
/// * `fc` - cutoff frequency
/// * `as_` - stop-band attenuation
///
/// # Returns
///
/// Vec of filter coefficients
pub fn fir_filter_design_firdespm<Coeff>(h_len: usize, fc: f32, as_: f32) -> Result<Vec<Coeff>>
where Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    let mut h = filter::fir_design_pm_lowpass(h_len, fc, as_, 0.0)?;
    // scale by filter bandwidth to be consistent with other lowpass prototypes
    for h_i in h.iter_mut() {
        *h_i = *h_i * 0.5 / fc;
    }
    let h_c = h.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    Ok(h_c)
}

/// create coefficients for a rectangular filter prototype
///
/// # Arguments
///
/// * `n` - filter length
///
/// # Returns
///
/// Vec of filter coefficients
pub fn fir_filter_design_rect<Coeff>(n: usize) -> Result<Vec<Coeff>>
where Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    if n == 0 || n > 1024 {
        return Err(Error::Config("filter length must be in [1,1024]".into()));
    }
    let h = vec![Coeff::one(); n];
    Ok(h)
}

/// create coefficients for a DC blocking filter
///
/// # Arguments
///
/// * `m` - filter delay
/// * `as_` - stop-band attenuation
///
/// # Returns
///
/// Vec of filter coefficients
pub fn fir_filter_design_dc_blocker<Coeff>(m: usize, as_: f32) -> Result<Vec<Coeff>>
where Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    f32: Into<Coeff>,
{
    let h = filter::fir_design_notch(m, 0.0, as_)?;
    let h_c = h.iter().map(|&x| x.into()).collect::<Vec<Coeff>>();
    Ok(h_c)
}

/// create coefficients for a notch filter
///
/// # Arguments
///
/// * `m` - filter delay
/// * `as_` - stop-band attenuation
/// * `f0` - center frequency
///
/// # Returns
///
/// Vec of filter coefficients
pub fn fir_filter_design_notch<Coeff>(m: usize, as_: f32, f0: f32) -> Result<Vec<Coeff>>
where Coeff: ComplexFloat<Real = f32> + Into<Complex32> + ComplexNotch,
    f32: Into<Coeff>,
{
    let h = Coeff::notch(m, as_, f0)?;
    Ok(h)
}

/// compute complex frequency response of filter coefficients
///
/// # Arguments
///
/// * `h` - filter coefficients
/// * `scale` - output scaling
/// * `fc` - normalized frequency for evaluation
///
/// # Returns
///
/// The frequency response
pub fn fir_filter_freqresponse<Coeff>(h: &[Coeff], scale: Coeff, fc: f32) -> Complex32
where Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    Complex32: From<Coeff>,
{
    let h_fc = filter::freqresponse(h, fc).unwrap();
    h_fc * Complex32::from(scale)
}


/// compute power spectral density response of filter coefficients in dB
///
/// # Arguments
///
/// * `h` - filter coefficients
/// * `scale` - output scaling
/// * `fc` - normalized frequency for evaluation
///
/// # Returns
///
/// The power spectral density
pub fn fir_filter_get_psd<Coeff>(h: &[Coeff], scale: Coeff, fc: f32) -> f32
where Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
    Complex32: From<Coeff>,
{
    let h = fir_filter_freqresponse(h, scale, fc);
    10.0 * (h * h.conj()).re.log10()
}

/// compute and return group delay of filter coefficients
///
/// # Arguments
///
/// * `h` - filter coefficients
/// * `fc` - normalized frequency for evaluation
///
/// # Returns
///
/// The group delay
pub fn fir_filter_groupdelay<Coeff>(h: &[Coeff], fc: f32) -> Result<f32>
where Coeff: ComplexFloat<Real = f32> + Into<Complex32>,
{
    let h = h.iter().map(|&x| x.re()).collect::<Vec<f32>>();
    filter::fir_group_delay(&h, fc)
}

/// Trait for FirFilter to implement notch filter design
pub trait ComplexNotch
where
    Self: Sized,
{
    fn notch(m: usize, as_: f32, f0: f32) -> Result<Vec<Self>>;
}

impl ComplexNotch for f32 {
    fn notch(m: usize, as_: f32, f0: f32) -> Result<Vec<Self>> {
        let h = filter::fir_design_notch(m, f0, as_)?;
        let h_c = h.iter().map(|&x| x.into()).collect::<Vec<f32>>();
        Ok(h_c)
    }
}

impl ComplexNotch for Complex32 {
    fn notch(m: usize, as_: f32, f0: f32) -> Result<Vec<Self>> {
        // design notch filter as DC blocker, then mix to appropriate frequency 
        let h = filter::fir_design_notch(m, 0.0, as_)?;
        let mut h_c = h.iter().map(|&x| Complex32::new(x, 0.0)).collect::<Vec<Complex32>>();
        for (i, h_i) in h_c.iter_mut().enumerate() {
            let phi = 2.0 * PI * f0 * (i as f32 - m as f32);
            *h_i = *h_i * Complex32::from_polar(1.0, phi);
        }
        Ok(h_c)
    }
}

/// Finite impulse response (FIR) filter
#[derive(Debug, Clone)]
pub struct FirFilter<T, Coeff = T> {
    h: Vec<Coeff>,
    h_rev: Vec<Coeff>,
    h_len: usize,
    w: Window<T>,
    scale: Coeff,
}

impl<T, Coeff> FirFilter<T, Coeff>
where
    T: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T>,
    Coeff: Copy + Default + ComplexFloat<Real = f32> + ComplexNotch,
    [T]: DotProd<Coeff, Output = T>,
    f32: Into<Coeff>,
    Complex32: From<Coeff>,
{
    /// create filter using coefficients directly specified in an array
    /// 
    /// # Arguments
    /// 
    /// * `h` - filter coefficients
    /// 
    /// # Returns
    /// 
    /// A new `Firfilt` object.
    pub fn new(h: &[Coeff]) -> Result<Self> {
        let h_len = h.len();
        if h_len == 0 {
            return Err(Error::Config("filter length must be greater than zero".into()));
        }

        let h_rev = h.iter().rev().cloned().collect::<Vec<Coeff>>();

        let mut q = Self {
            h: h.to_vec(),
            h_rev,
            h_len,
            w: Window::new(h_len)?,
            scale: Coeff::one(),
        };

        q.reset();

        Ok(q)
    }

    /// create filter using Kaiser-Bessel windowed sinc method
    /// 
    /// # Arguments
    /// 
    /// * `n` - filter length
    /// * `fc` - cutoff frequency
    /// * `as_` - stop-band attenuation
    /// * `mu` - fractional sample offset
    /// 
    /// # Returns
    /// 
    /// A new `Firfilt` object.
    pub fn new_kaiser(n: usize, fc: f32, as_: f32, mu: f32) -> Result<Self> {
        let h = fir_filter_design_kaiser(n, fc, as_, mu)?;
        Self::new(&h)
    }

    /// create filter using square-root Nyquist prototype
    /// 
    /// # Arguments
    /// 
    /// * `ftype` - filter type
    /// * `k` - nominal samples/symbol
    /// * `m` - filter delay
    /// * `beta` - rolloff factor
    /// * `mu` - fractional sample offset
    /// 
    /// # Returns
    /// 
    /// A new `Firfilt` object.
    pub fn new_rnyquist(ftype: filter::FirFilterShape, k: usize, m: usize, beta: f32, mu: f32) -> Result<Self> {
        let h = fir_filter_design_rnyquist(ftype, k, m, beta, mu)?;
        Self::new(&h)
    }

    /// create filter using Parks-McClellan algorithm
    /// 
    /// # Arguments
    /// 
    /// * `h_len` - filter length
    /// * `fc` - cutoff frequency
    /// * `as_` - stop-band attenuation
    /// 
    /// # Returns
    /// 
    /// A new `Firfilt` object.
    pub fn new_firdespm(h_len: usize, fc: f32, as_: f32) -> Result<Self> {
        let h = fir_filter_design_firdespm(h_len, fc, as_)?;
        Self::new(&h)
    }

    /// create rectangular filter prototype
    /// 
    /// # Arguments
    /// 
    /// * `n` - filter length
    /// 
    /// # Returns
    /// 
    /// A new `Firfilt` object.
    pub fn new_rect(n: usize) -> Result<Self> {
        let h = fir_filter_design_rect(n)?;
        Self::new(&h)
    }

    /// create DC blocking filter
    /// 
    /// # Arguments
    /// 
    /// * `m` - filter delay
    /// * `as_` - stop-band attenuation
    /// 
    /// # Returns
    /// 
    /// A new `Firfilt` object.
    pub fn new_dc_blocker(m: usize, as_: f32) -> Result<Self> {
        let h = fir_filter_design_dc_blocker(m, as_)?;
        Self::new(&h)
    }

    /// create notch filter
    /// 
    /// # Arguments
    /// 
    /// * `m` - filter delay
    /// * `as_` - stop-band attenuation
    /// * `f0` - center frequency
    /// 
    /// # Returns
    /// 
    /// A new `Firfilt` object.
    pub fn new_notch(m: usize, as_: f32, f0: f32) -> Result<Self> {
        let h = fir_filter_design_notch(m, as_, f0)?;
        Self::new(&h)
    }

    /// set filter coefficients
    /// 
    /// # Arguments
    /// 
    /// * `h` - filter coefficients
    pub fn set_coefficients(&mut self, h: &[Coeff]) -> Result<()> {
        // aka recreate
        let n = h.len();
        if n != self.h_len {
            self.h_len = n;
            self.h.resize(n, Coeff::default());
            self.h_rev.resize(n, Coeff::default());
            self.w.resize(n)?;
        }

        self.h.copy_from_slice(h);
        self.h_rev.copy_from_slice(&h.iter().rev().cloned().collect::<Vec<Coeff>>());
        self.reset();

        Ok(())
    }

    /// reset internal state of filter object
    pub fn reset(&mut self) {
        self.w.reset();
    }

    /// push sample into filter object's internal buffer
    /// 
    /// # Arguments
    /// 
    /// * `x` - single input sample
    pub fn push(&mut self, x: T) {
        self.w.push(x);
    }

    /// write block of samples into filter object's internal buffer
    /// 
    /// # Arguments
    /// 
    /// * `x` - buffer of input samples
    pub fn write(&mut self, x: &[T]) {
        for x_i in x.iter() {
            self.push(*x_i);
        }
    }

    /// execute filter on internal buffer and coefficients
    /// 
    /// # Returns
    /// 
    /// The output sample
    pub fn execute(&self) -> T {
        let x = self.w.read();
        let y = x.dotprod(&self.h_rev);
        y * self.scale
    }

    /// execute filter on one sample, equivalent to push() and execute()
    /// 
    /// # Arguments
    /// 
    /// * `x` - single input sample
    /// 
    /// # Returns
    /// 
    /// The output sample
    pub fn execute_one(&mut self, x: T) -> T {
        self.push(x);
        self.execute()
    }

    /// execute filter on block of samples
    /// 
    /// # Arguments
    /// 
    /// * `x` - buffer of input samples
    /// * `y` - buffer of output samples
    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if x.len() != y.len() {
            return Err(Error::Config("input and output block lengths must be equal".into()));
        }

        for (x_i, y_i) in x.iter().zip(y.iter_mut()) {
            self.push(*x_i);
            *y_i = self.execute();
        }

        Ok(())
    }

    /// set output scaling for filter
    /// 
    /// # Arguments
    /// 
    /// * `scale` - output scaling
    pub fn set_scale(&mut self, scale: Coeff) {
        self.scale = scale;
    }

    /// get output scaling for filter
    /// 
    /// # Returns
    /// 
    /// The output scaling
    pub fn get_scale(&self) -> Coeff {
        self.scale
    }

    /// get length of filter object (number of internal coefficients)
    /// 
    /// # Returns
    /// 
    /// The length of the filter
    pub fn get_length(&self) -> usize {
        self.h_len
    }

    /// get pointer to coefficients array
    /// 
    /// # Returns
    /// 
    /// The coefficients array
    pub fn get_coefficients(&self) -> &[Coeff] {
        &self.h
    }

    /// compute complex frequency response of filter object
    /// 
    /// # Arguments
    /// 
    /// * `fc` - normalized frequency for evaluation
    /// 
    /// # Returns
    /// 
    /// The frequency response
    pub fn freqresponse(&self, fc: f32) -> Complex32 {
        fir_filter_freqresponse(&self.h, self.scale, fc)
    }

    /// compute power spectral density response of filter object in dB
    ///
    /// # Arguments
    ///
    /// * `fc` - normalized frequency for evaluation
    ///
    /// # Returns
    ///
    /// The power spectral density
    pub fn get_psd(&self, fc: f32) -> f32 {
        fir_filter_get_psd(&self.h, self.scale, fc)
    }

    /// compute and return group delay of filter object
    /// 
    /// # Arguments
    /// 
    /// * `fc` - frequency to evaluate
    /// 
    /// # Returns
    /// 
    /// The group delay
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        fir_filter_groupdelay(&self.h, fc)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use test_macro::autotest_annotate;
    use crate::filter::fir::design::FirFilterShape;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_firfilt, validate_psd_firfiltc};

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_kaiser)]
    fn test_firfilt_crcf_kaiser() {
        // design filter
        let mut q = FirFilter::new_kaiser(51, 0.2, 60.0, 0.0).unwrap();
        q.set_scale(0.4);

        // verify resulting spectrum
        let regions = [
            PsdRegion { fmin: -0.5,  fmax: -0.25, pmin: 0.0,  pmax: -60.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.15, fmax: 0.15,  pmin: -0.1, pmax: 0.1,   test_lo: true,  test_hi: true },
            PsdRegion { fmin: 0.25,  fmax: 0.5,   pmin: 0.0,  pmax: -60.0, test_lo: false, test_hi: true },
        ];
        
        assert!(validate_psd_firfilt(&q, 1200, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_firdespm)]
    fn test_firfilt_crcf_firdespm() {
        // design filter
        let mut q = FirFilter::new_firdespm(51, 0.2, 60.0).unwrap();
        q.set_scale(0.4);

        // verify resulting spectrum
        let regions = [
            PsdRegion { fmin: -0.5,  fmax: -0.25, pmin: 0.0,  pmax: -60.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.15, fmax: 0.15,  pmin: -0.1, pmax: 0.1,   test_lo: true,  test_hi: true },
            PsdRegion { fmin: 0.25,  fmax: 0.5,   pmin: 0.0,  pmax: -60.0, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_firfilt(&q, 1200, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_rect)]
    fn test_firfilt_crcf_rect() {
        // design filter
        let mut q = FirFilter::new_rect(4).unwrap();
        q.set_scale(0.25);

        // verify resulting spectrum
        let regions = [
            PsdRegion { fmin: -0.5,  fmax: -0.20, pmin: 0.0, pmax: -10.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.12, fmax: 0.12,  pmin: -5.0, pmax: 1.0,  test_lo: true,  test_hi: true },
            PsdRegion { fmin: 0.20,  fmax: 0.5,   pmin: 0.0, pmax: -10.0, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_firfilt(&q, 301, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_notch)]
    fn test_firfilt_crcf_notch() {
        // design filter and verify resulting spectrum
        let q = FirFilter::<Complex32, f32>::new_notch(20, 60.0, 0.125).unwrap();
        let regions = [
            PsdRegion { fmin: -0.5,   fmax: -0.20,  pmin: -0.1, pmax: 0.1,  test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.126, fmax: -0.124, pmin: 0.0,  pmax: -50.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.06,  fmax: 0.06,   pmin: -0.1, pmax: 0.1,  test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.124,  fmax: 0.126,  pmin: 0.0,  pmax: -50.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: 0.20,   fmax: 0.5,    pmin: -0.1, pmax: 0.1,  test_lo: true, test_hi: true },
        ];

        assert!(validate_psd_firfilt(&q, 1200, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_notch)]
    fn test_firfilt_cccf_notch() {
        // design filter and verify resulting spectrum
        let q = FirFilter::<Complex32, Complex32>::new_notch(20, 60.0, 0.125).unwrap();
        let regions = [
            PsdRegion { fmin: -0.5,   fmax: 0.06,   pmin: -0.1, pmax: 0.1,  test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.124,  fmax: 0.126,  pmin: 0.0,  pmax: -50.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: 0.20,   fmax: 0.5,    pmin: -0.1, pmax: 0.1,  test_lo: true, test_hi: true },
        ];

        assert!(validate_psd_firfiltc(&q, 1200, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_config)]
    fn test_firfilt_config() {
        // no need to check every combination
        assert!(FirFilter::<Complex32, f32>::new(&[]).is_err());
        assert!(FirFilter::<Complex32, f32>::new_kaiser(0, 0.0, 0.0, 0.0).is_err());
        assert!(FirFilter::<Complex32, f32>::new_rnyquist(filter::FirFilterShape::Arkaiser, 0, 0, 0.0, 4.0).is_err());
        assert!(FirFilter::<Complex32, f32>::new_firdespm(0, 0.0, 0.0).is_err());
        assert!(FirFilter::<Complex32, f32>::new_rect(0).is_err());
        assert!(FirFilter::<Complex32, f32>::new_dc_blocker(0, 0.0).is_err());
        assert!(FirFilter::<Complex32, f32>::new_notch(0, 0.0, 0.0).is_err());
        assert!(FirFilter::<Complex32, Complex32>::new_notch(0, 0.0, 0.0).is_err());

        // create proper object and test configurations
        let mut q = FirFilter::<Complex32, f32>::new_kaiser(11, 0.2, 60.0, 0.0).unwrap();

        // assert!(q.print().is_ok());

        q.set_scale(3.0);
        let scale = q.get_scale();
        assert_eq!(scale, 3.0);
        assert_eq!(q.get_length(), 11);
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_recreate)]
    fn test_firfilt_recreate() {
        // create random-ish coefficients
        let n = 21;
        let mut h0 = vec![0.0f32; n];
        let mut h1 = vec![0.0f32; n];
        for i in 0..n {
            h0[i] = (0.3 * i as f32).cos() + (2.0f32.sqrt() * i as f32).sin();
        }

        let mut q = FirFilter::new(&h0).unwrap();

        // assert!(q.print().is_ok());
        q.set_scale(3.0);

        // copy coefficients to separate array
        h1.copy_from_slice(&h0);

        // scale coefficients by a constant
        for h in h1.iter_mut() {
            *h *= 7.1;
        }

        // recreate with new coefficients array
        q.set_coefficients(&h1).unwrap();

        // assert the scale has not changed
        let scale = q.get_scale();
        assert_eq!(scale, 3.0);

        // assert the coefficients are original scaled by 7.1
        let h = q.get_coefficients();
        for i in 0..n {
            assert_relative_eq!(h[i], h0[i] * 7.1, max_relative = 1e-6);
        }

        // re-create with longer coefficients array and test impulse response
        let mut h2 = vec![0.0f32; 2*n+1]; // new random-ish coefficients
        for i in 0..(2*n+1) {
            h2[i] = (0.2 * i as f32 + 1.0).cos() + (2.0f32.ln() * i as f32).sin();
        }
        q.set_coefficients(&h2).unwrap();

        for i in 0..(2*n+1) {
            q.push(Complex32::new(if i == 0 { 1.0 } else { 0.0 }, 0.0));
            let v = q.execute();
            // output is same as input, subject to scaling factor
            assert_relative_eq!(v.re, h2[i] * scale, max_relative = 1e-6);
            assert_relative_eq!(v.im, 0.0, max_relative = 1e-6);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_push_write)]
    fn test_firfilt_push_write() {
        // create two identical objects
        let mut q0 = FirFilter::<f32, f32>::new_kaiser(51, 0.2, 60.0, 0.0).unwrap();
        let mut q1 = FirFilter::<f32, f32>::new_kaiser(51, 0.2, 60.0, 0.0).unwrap();

        // generate pseudo-random inputs, and compare outputs
        let buf: [f32; 8] = [-1.0, 3.0, 5.0, -3.0, 5.0, 1.0, -3.0, -4.0];

        for trial in 0..20 {
            let n = trial % 8;

            // push/write samples
            for i in 0..n {
                q0.push(buf[i]);
            }
            q1.write(&buf[..n]);

            // compute outputs and compare
            let v0 = q0.execute();
            let v1 = q1.execute();
            assert_relative_eq!(v0, v1);
        }

        // No need to explicitly destroy objects in Rust
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_copy)]
    fn test_firfilt_crcf_copy() {
        // design filter from prototype
        let mut filt_orig = FirFilter::<Complex32, f32>::new_kaiser(21, 0.345, 60.0, 0.0).unwrap();
        filt_orig.set_scale(2.0);

        // start running input through filter
        let n = 32;
        for _ in 0..n {
            let x = Complex32::new(crate::random::randnf(), crate::random::randnf());
            filt_orig.push(x);
        }

        // copy filter
        let mut filt_copy = filt_orig.clone();

        // continue running through both filters
        for _ in 0..n {
            // run filters in parallel
            let x = Complex32::new(crate::random::randnf(), crate::random::randnf());
            let y_orig = filt_orig.execute_one(x);
            let y_copy = filt_copy.execute_one(x);

            let h_orig = filt_orig.get_coefficients();
            let h_copy = filt_copy.get_coefficients();

            let scale_orig = filt_orig.get_scale();
            let scale_copy = filt_copy.get_scale();

            let h_len_orig = filt_orig.get_length();
            let h_len_copy = filt_copy.get_length();

            let w_orig = &filt_orig.w;
            let w_copy = &filt_copy.w;

            assert_eq!(h_orig, h_copy);
            assert_eq!(scale_orig, scale_copy);
            assert_eq!(w_orig, w_copy);
            assert_eq!(h_len_orig, h_len_copy);

            // TODO this is a weird one. even though the structs match,
            //   the subsequent dotprod doesn't.
            assert_relative_eq!(y_orig.re, y_copy.re, epsilon = 1e-6);
            assert_relative_eq!(y_orig.im, y_copy.im, epsilon = 1e-6);
        }

        // No need to explicitly destroy filter objects in Rust
    }

    fn firfilt_cccf_notch_test_harness(m: usize, as_: f32, f0: f32) {
        let num_samples = 600;     // number of samples
        let h_len = 2 * m + 1;  // filter length

        // design filter from prototype
        let mut q = FirFilter::<Complex32, Complex32>::new_notch(m, as_, f0).unwrap();

        // generate input signal
        let mut x2 = 0.0f32;
        let mut y2 = 0.0f32;
        for i in 0..(num_samples + h_len) {
            // compute input: tone at f0
            let x = Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * f0 * i as f32);

            // filter input
            q.push(x);
            let y = q.execute();

            // accumulate, compensating for filter delay
            if i >= h_len {
                x2 += x.norm_sqr();
                y2 += y.norm_sqr();
            }
        }

        // compare result
        x2 = (x2 / num_samples as f32).sqrt();
        y2 = (y2 / num_samples as f32).sqrt();
        let tol = 1e-3f32;
        assert_relative_eq!(x2, 1.0f32, epsilon = tol);
        assert_relative_eq!(y2, 0.0f32, epsilon = tol);

        // No need to explicitly destroy filter object in Rust
    }

    // AUTOTESTS: 
    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_notch_0)]
    fn test_firfilt_cccf_notch_0() { firfilt_cccf_notch_test_harness(20, 60.0, 0.000); }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_notch_1)]
    fn test_firfilt_cccf_notch_1() { firfilt_cccf_notch_test_harness(20, 60.0, 0.100); }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_notch_2)]
    fn test_firfilt_cccf_notch_2() { firfilt_cccf_notch_test_harness(20, 60.0, 0.456); }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_notch_3)]
    fn test_firfilt_cccf_notch_3() { firfilt_cccf_notch_test_harness(20, 60.0, 0.500); }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_notch_4)]
    fn test_firfilt_cccf_notch_4() { firfilt_cccf_notch_test_harness(20, 60.0, -0.250); }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_notch_5)]
    fn test_firfilt_cccf_notch_5() { firfilt_cccf_notch_test_harness(20, 60.0, -0.389); }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_coefficients_test)]
    fn test_firfilt_cccf_coefficients() {
        // we are omitting "copy coefficients" because returning a slice is sufficient
        // create filter coefficients (semi-random)
        let h_len = 71;
        let mut h0 = vec![Complex32::new(0.0, 0.0); h_len];
        let mut h1 = vec![Complex32::new(0.0, 0.0); h_len];
        for i in 0..h_len {
            h0[i] = Complex32::from_polar(1.0, 0.2 * i as f32) * crate::math::hamming(i, h_len).unwrap();
        }

        // design filter from external coefficients
        let mut q = FirFilter::<Complex32, Complex32>::new(&h0).unwrap();

        // set scale: note that this is not used when computing coefficients
        q.set_scale(Complex32::new(-0.4, 0.7));

        // copy coefficients from filter object
        // q.copy_coefficients(&mut h1);

        // copy coefficients from filter object
        let h2 = q.get_coefficients();
        h1.copy_from_slice(&h2);

        // ensure values are equal; no need for tolerance as values should be exact
        for i in 0..h_len {
            assert_eq!(h0[i].re, h1[i].re);
            assert_eq!(h0[i].im, h1[i].im);

            assert_eq!(h0[i].re, h2[i].re);
            assert_eq!(h0[i].im, h2[i].im);
        }

        // No need to explicitly destroy filter object or free memory in Rust
    }

    fn testbench_firfilt_rnyquist(ftype: FirFilterShape, // filter type
                                  k: usize,             // samples/symbol
                                  m: usize,             // semi-length
                                  beta: f32,            // excess bandwidth factor
                                  dt: f32)              // fractional delay
    {
        // derived values
        let hc_len = 4 * k * m + 1; // composite filter length

        // arrays
        let mut hc = vec![0.0; hc_len]; // composite filter

        // design the filter(s)
        let ht = filter::fir_design_prototype(ftype, k, m, beta, dt).unwrap();

        // special case for GMSK
        let hr = if ftype == FirFilterShape::Gmsktx {
            filter::fir_design_prototype(filter::FirFilterShape::Gmskrx, k, m, beta, dt).unwrap()
        } else {
            ht.clone()
        };

        // TODO: check group delay

        // compute composite filter response (auto-correlation)
        for i in 0..hc_len {
            let lag = i as i32 - (2 * k * m) as i32;
            hc[i] = filter::filter_crosscorr(&ht, &hr, lag as isize);
        };

        // compute filter inter-symbol interference
        let rxx0 = hc[2 * k * m];
        let mut isi_rms = 0.0;
        for i in 1..(2 * m) {
            let e = hc[i * k] / rxx0; // composite at integer multiple of k except for i==0 ideally 0
            isi_rms += e * e; // squared magnitude, doubled to account for filter symmetry
        }
        isi_rms = 10.0 * (isi_rms / (2 * m - 1) as f32).log10();

        // compute relative stop-band energy
        let nfft = 2048;
        let as_ = 20.0 * filter::filter_energy(&ht, 0.5 * (1.0 + beta) / k as f32, nfft).unwrap().log10();

        assert_relative_eq!(rxx0, k as f32, max_relative = 0.01);
        assert!(isi_rms < -50.0);
        assert!(as_ < -50.0);
    }

    // test different filter designs, nominal parameters
    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_baseline_arkaiser)]
    fn test_firfilt_rnyquist_baseline_arkaiser() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 2, 9, 0.3, 0.0); }

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_baseline_rkaiser)]
    fn test_firfilt_rnyquist_baseline_rkaiser() { testbench_firfilt_rnyquist(FirFilterShape::Rkaiser, 2, 9, 0.3, 0.0); }

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_baseline_rrc)]
    fn test_firfilt_rnyquist_baseline_rrc() { testbench_firfilt_rnyquist(FirFilterShape::Rrcos, 2, 9, 0.3, 0.0); }

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_baseline_hm3)]
    fn test_firfilt_rnyquist_baseline_hm3() { testbench_firfilt_rnyquist(FirFilterShape::Hm3, 2, 9, 0.3, 0.0); }

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_baseline_gmsktxrx)]
    fn test_firfilt_rnyquist_baseline_gmsktxrx() { testbench_firfilt_rnyquist(FirFilterShape::Gmsktx, 2, 9, 0.3, 0.0); }

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_baseline_rfexp)]
    fn test_firfilt_rnyquist_baseline_rfexp() { testbench_firfilt_rnyquist(FirFilterShape::Rfexp, 2, 9, 0.3, 0.0); }

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_baseline_rfsech)]
    fn test_firfilt_rnyquist_baseline_rfsech() { testbench_firfilt_rnyquist(FirFilterShape::Rfsech, 2, 9, 0.3, 0.0); }

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_baseline_rfarcsech)]
    fn test_firfilt_rnyquist_baseline_rfarcsech() { testbench_firfilt_rnyquist(FirFilterShape::Rfarcsech, 2, 9, 0.3, 0.0); }

    // test different parameters
    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_0)]
    fn test_firfilt_rnyquist_0() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 2, 4, 0.33, 0.0); } // short length

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_1)]
    fn test_firfilt_rnyquist_1() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 2, 12, 0.20, 0.0); } // longer length

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_2)]
    fn test_firfilt_rnyquist_2() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 2, 40, 0.20, 0.0); } // very long length

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_3)]
    fn test_firfilt_rnyquist_3() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 3, 12, 0.20, 0.0); } // k=3

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_4)]
    fn test_firfilt_rnyquist_4() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 4, 12, 0.20, 0.0); } // k=4

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_5)]
    fn test_firfilt_rnyquist_5() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 5, 12, 0.20, 0.0); } // k=5

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_6)]
    fn test_firfilt_rnyquist_6() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 20, 12, 0.20, 0.0); } // k=20

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_7)]
    fn test_firfilt_rnyquist_7() { testbench_firfilt_rnyquist(FirFilterShape::Arkaiser, 2, 12, 0.80, 0.0); } // large excess bandwidth

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_8)]
    fn test_firfilt_rnyquist_8() { testbench_firfilt_rnyquist(FirFilterShape::Rkaiser, 2, 12, 0.20, 0.5); } // iterative design, typical

    #[test]
    #[autotest_annotate(autotest_firfilt_rnyquist_9)]
    fn test_firfilt_rnyquist_9() { testbench_firfilt_rnyquist(FirFilterShape::Rkaiser, 20, 40, 0.20, 0.5); } // iterative design, stressed


    #[test]
    #[autotest_annotate(autotest_fir_groupdelay_n3)]
    fn test_fir_groupdelay_n3() {
        // create coefficients array
        let h: [f32; 3] = [0.1, 0.2, 0.4];

        let tol = 1e-3f32;

        // create testing vectors
        let fc: [f32; 4] = [0.000, 0.125, 0.250, 0.375];
        
        let g0: [f32; 4] = [1.42857142857143,
                            1.54756605839643,
                            2.15384615384615,
                            2.56861651421767];

        // run tests
        for i in 0..4 {
            let g = filter::fir_group_delay(&h, fc[i]).unwrap();
            assert_relative_eq!(g, g0[i], max_relative = tol);
        }

        // create filter
        let filter = FirFilter::<f32, f32>::new(&h).unwrap();

        // run tests again
        for i in 0..4 {
            let g = filter.groupdelay(fc[i]).unwrap();
            assert_relative_eq!(g, g0[i], max_relative = tol);
        }
    }

    include!("firfilt_test_data.rs");

    // autotest helper function
    //  h      :   filter coefficients
    //  h_len  :   filter coefficients length
    //  x      :   input array
    //  x_len  :   input array length
    //  y      :   output array
    //  y_len  :   output array length
    fn firfilt_rrrf_test(h: &[f32], x: &[f32], y: &[f32]) {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = FirFilter::<f32, f32>::new(h).unwrap();

        // allocate memory for output
        let mut y_test = vec![0.0; y.len()];

        // compute output
        for (i, &x_i) in x.iter().enumerate() {
            q.push(x_i);
            y_test[i] = q.execute();
            
            assert_relative_eq!(y_test[i], y[i], max_relative = tol);
        }
    }

    // autotest helper function
    //  h      :   filter coefficients
    //  h_len  :   filter coefficients length
    //  x      :   input array
    //  x_len  :   input array length
    //  y      :   output array
    //  y_len  :   output array length
    fn firfilt_crcf_test(h: &[f32], x: &[Complex32], y: &[Complex32]) {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = FirFilter::<Complex32, f32>::new(h).unwrap();

        // allocate memory for output
        let mut y_test = vec![Complex32::new(0.0, 0.0); y.len()];

        // compute output
        for (i, &x_i) in x.iter().enumerate() {
            q.push(x_i);
            y_test[i] = q.execute();
            
            assert_relative_eq!(y_test[i].re, y[i].re, max_relative = tol);
            assert_relative_eq!(y_test[i].im, y[i].im, max_relative = tol);
        }
    }

    // autotest helper function
    //  h      :   filter coefficients
    //  h_len  :   filter coefficients length
    //  x      :   input array
    //  x_len  :   input array length
    //  y      :   output array
    //  y_len  :   output array length
    fn firfilt_cccf_test(h: &[Complex32], x: &[Complex32], y: &[Complex32]) {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = FirFilter::<Complex32, Complex32>::new(h).unwrap();

        // allocate memory for output
        let mut y_test = vec![Complex32::new(0.0, 0.0); y.len()];

        // compute output
        for (i, &x_i) in x.iter().enumerate() {
            q.push(x_i);
            y_test[i] = q.execute();
            
            assert_relative_eq!(y_test[i].re, y[i].re, max_relative = tol);
            assert_relative_eq!(y_test[i].im, y[i].im, max_relative = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_rrrf_data_h4x8)]
    fn test_firfilt_rrrf_data_h4x8() {
        firfilt_rrrf_test(
            &FIRFILT_RRRF_DATA_H4X8_H,
            &FIRFILT_RRRF_DATA_H4X8_X,
            &FIRFILT_RRRF_DATA_H4X8_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_rrrf_data_h7x16)]
    fn test_firfilt_rrrf_data_h7x16() {
        firfilt_rrrf_test(
            &FIRFILT_RRRF_DATA_H7X16_H,
            &FIRFILT_RRRF_DATA_H7X16_X,
            &FIRFILT_RRRF_DATA_H7X16_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_rrrf_data_h13x32)]
    fn test_firfilt_rrrf_data_h13x32() {
        firfilt_rrrf_test(
            &FIRFILT_RRRF_DATA_H13X32_H,
            &FIRFILT_RRRF_DATA_H13X32_X,
            &FIRFILT_RRRF_DATA_H13X32_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_rrrf_data_h23x64)]
    fn test_firfilt_rrrf_data_h23x64() {
        firfilt_rrrf_test(
            &FIRFILT_RRRF_DATA_H23X64_H,
            &FIRFILT_RRRF_DATA_H23X64_X,
            &FIRFILT_RRRF_DATA_H23X64_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_data_h4x8)]
    fn test_firfilt_crcf_data_h4x8() {
        firfilt_crcf_test(
            &FIRFILT_CRCF_DATA_H4X8_H,
            &FIRFILT_CRCF_DATA_H4X8_X,
            &FIRFILT_CRCF_DATA_H4X8_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_data_h7x16)]
    fn test_firfilt_crcf_data_h7x16() {
        firfilt_crcf_test(
            &FIRFILT_CRCF_DATA_H7X16_H,
            &FIRFILT_CRCF_DATA_H7X16_X,
            &FIRFILT_CRCF_DATA_H7X16_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_data_h13x32)]
    fn test_firfilt_crcf_data_h13x32() {
        firfilt_crcf_test(
            &FIRFILT_CRCF_DATA_H13X32_H,
            &FIRFILT_CRCF_DATA_H13X32_X,
            &FIRFILT_CRCF_DATA_H13X32_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_crcf_data_h23x64)]
    fn test_firfilt_crcf_data_h23x64() {
        firfilt_crcf_test(
            &FIRFILT_CRCF_DATA_H23X64_H,
            &FIRFILT_CRCF_DATA_H23X64_X,
            &FIRFILT_CRCF_DATA_H23X64_Y
        );
    }
    
    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_data_h4x8)]
    fn test_firfilt_cccf_data_h4x8() {
        firfilt_cccf_test(
            &FIRFILT_CCCF_DATA_H4X8_H,
            &FIRFILT_CCCF_DATA_H4X8_X,
            &FIRFILT_CCCF_DATA_H4X8_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_data_h7x16)]
    fn test_firfilt_cccf_data_h7x16() {
        firfilt_cccf_test(
            &FIRFILT_CCCF_DATA_H7X16_H,
            &FIRFILT_CCCF_DATA_H7X16_X,
            &FIRFILT_CCCF_DATA_H7X16_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_data_h13x32)]
    fn test_firfilt_cccf_data_h13x32() {
        firfilt_cccf_test(
            &FIRFILT_CCCF_DATA_H13X32_H,
            &FIRFILT_CCCF_DATA_H13X32_X,
            &FIRFILT_CCCF_DATA_H13X32_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_firfilt_cccf_data_h23x64)]
    fn test_firfilt_cccf_data_h23x64() {
        firfilt_cccf_test(
            &FIRFILT_CCCF_DATA_H23X64_H,
            &FIRFILT_CCCF_DATA_H23X64_X,
            &FIRFILT_CCCF_DATA_H23X64_Y
        );
    }
}