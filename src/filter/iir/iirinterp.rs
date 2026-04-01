use crate::filter::iir::iirfilt::IirFilter;
use crate::filter::iir::design::{IirFilterShape, IirBandType, IirFormat};
use crate::error::{Error, Result};
use num_complex::{ComplexFloat, Complex32};
use crate::dotprod::DotProd;

#[derive(Debug, Clone)]
pub struct IirInterpolationFilter<T, Coeff = T> {
    m: usize,     // interpolation factor
    iirfilt: IirFilter<T, Coeff>,  // filter object
}

impl<T, Coeff> IirInterpolationFilter<T, Coeff>
where
    T: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + From<Coeff>,
    Coeff: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<T, Output = T> + Into<Complex32>,
    std::collections::VecDeque<T>: DotProd<Coeff, Output = T>,
    f32: Into<Coeff>,
{
    /// create interpolator from external coefficients
    pub fn new(m: usize, b: &[Coeff], a: &[Coeff]) -> Result<Self> {
        // validate input
        if m < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }

        // create filter
        let iirfilt = IirFilter::new(b, a)?;

        Ok(IirInterpolationFilter { m, iirfilt })
    }

    /// create interpolator with default Butterworth prototype
    pub fn new_default(m: usize, order: usize) -> Result<Self> {
        Self::new_prototype(
            m,
            IirFilterShape::Cheby2,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            order,
            0.5 / (m as f32),  // fc
            0.0,               // f0
            0.1,               // pass-band ripple
            60.0,              // stop-band attenuation
        )
    }

    /// create interpolator from prototype
    pub fn new_prototype(
        m: usize,
        ftype: IirFilterShape,
        btype: IirBandType,
        format: IirFormat,
        order: usize,
        fc: f32,
        f0: f32,
        ap: f32,
        as_: f32,
    ) -> Result<Self> {
        // validate input
        if m < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }

        // create filter
        let mut iirfilt = IirFilter::new_prototype(ftype, btype, format, order, fc, f0, ap, as_)?;

        // set appropriate scale
        iirfilt.set_scale(Coeff::from(m).unwrap());

        Ok(IirInterpolationFilter { m, iirfilt })
    }

    // copy object
    // pub fn copy(&self) -> Self {
    //     IirInterp {
    //         m: self.m,
    //         iirfilt: self.iirfilt.copy(),
    //     }
    // }

    // print interpolator state
    // pub fn print(&self) {
    //     println!("<liquid.iirinterp, interp={}>", self.m);
    // }

    /// clear internal state
    pub fn reset(&mut self) {
        self.iirfilt.reset();
    }

    /// execute interpolator
    pub fn execute(&mut self, x: T, y: &mut [T]) -> Result<()> {
        if y.len() != self.m {
            return Err(Error::Config("output array must be of length m".into()));
        }

        // TODO: use iirpfb
        for i in 0..self.m {
            y[i] = self.iirfilt.execute(if i == 0 { x } else { T::default() });
        }
        Ok(())
    }

    /// execute interpolation on block of input samples
    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if y.len() != x.len() * self.m {
            return Err(Error::Config("output array must be of length n * m".into()));
        }

        for (i, &xi) in x.iter().enumerate() {
            self.execute(xi, &mut y[i * self.m..(i + 1) * self.m])?;
        }
        Ok(())
    }

    /// get system group delay at frequency fc
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        Ok(self.iirfilt.groupdelay(fc)? / (self.m as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use crate::random::randnf;
    use crate::math::WindowType;
    use crate::fft::spgram::Spgram;
    use crate::filter::FirFilterShape;
    use crate::modem::modem::ModulationScheme;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_spectrum};
    use crate::framing::symstreamr::SymStreamR;

    fn test_iirinterp_crcf(_method: &str, interp_factor: usize, order: usize) {
        // options
        let n = 800000; // number of output samples to analyze
        let bw = 0.2f32; // target output bandwidth
        let nfft = 800; // number of bins in transform
        let as_ = 60.0f32; // error tolerance [dB]
        let tol = 0.5f32; // error tolerance [dB]

        // create resampler with rate interp/decim
        let mut interp = IirInterpolationFilter::<Complex32, f32>::new_default(interp_factor, order).unwrap();

        // create and configure objects
        let mut q = Spgram::<Complex32>::new(nfft, WindowType::Hann, nfft/2, nfft/4).unwrap();
        let mut gen = SymStreamR::new_linear(
            FirFilterShape::Kaiser,
            bw * interp_factor as f32,
            25,
            0.2,
            ModulationScheme::Qpsk
        ).unwrap();
        gen.set_gain((bw as f32).sqrt());

        // generate samples and push through spgram object
        let block_size = 10;
        let mut buf_0 = vec![Complex32::default(); block_size];
        let mut buf_1 = vec![Complex32::default(); block_size * interp_factor];
        while q.get_num_samples_total() < n {
            // generate block of samples
            gen.write_samples(&mut buf_0).unwrap();

            // interpolate
            interp.execute_block(&buf_0, &mut buf_1).unwrap();

            // run samples through the spgram object
            q.write(&buf_1);
        }

        // verify result
        let psd = q.get_psd();
        let regions = vec![
            PsdRegion{ fmin: -0.5,    fmax: -0.6*bw, pmin: 0.0,     pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion{ fmin: -0.4*bw, fmax: 0.4*bw,  pmin: 0.0-tol, pmax: 0.0+tol, test_lo: true, test_hi: true },
            PsdRegion{ fmin: 0.6*bw,  fmax: 0.5,     pmin: 0.0,     pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_spectrum(&psd, nfft, &regions).unwrap());
    }

    // baseline tests using create_kaiser() method
    #[test]
    #[autotest_annotate(autotest_iirinterp_crcf_M2_O9)]
    fn test_iirinterp_crcf_m2_o9() { test_iirinterp_crcf("baseline", 2, 9); }

    #[test]
    #[autotest_annotate(autotest_iirinterp_crcf_M3_O9)]
    fn test_iirinterp_crcf_m3_o9() { test_iirinterp_crcf("baseline", 3, 9); }

    #[test]
    #[autotest_annotate(autotest_iirinterp_crcf_M4_O9)]
    fn test_iirinterp_crcf_m4_o9() { test_iirinterp_crcf("baseline", 4, 9); }

    #[test]
    #[autotest_annotate(autotest_iirinterp_copy)]
    fn test_iirinterp_copy() {
        // create base object
        let mut q0 = IirInterpolationFilter::<Complex32, f32>::new_default(3, 7).unwrap();
        //q0.set_scale(0.12345f32);

        // run samples through filter
        let mut buf_0 = [Complex32::default(); 3];
        for _ in 0..20 {
            let v = Complex32::new(randnf(), randnf());
            q0.execute(v, &mut buf_0).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters in parallel
        let mut buf_1 = [Complex32::default(); 3];
        for _ in 0..60 {
            let v = Complex32::new(randnf(), randnf());
            q0.execute(v, &mut buf_0).unwrap();
            q1.execute(v, &mut buf_1).unwrap();

            assert_eq!(buf_0, buf_1);
        }

        // objects are automatically destroyed when they go out of scope
    }
}