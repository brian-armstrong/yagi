use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::iir::design::{IirBandType, IirFilterShape, IirFormat};
use crate::filter::iir::filter::IirFilter;
use num_complex::{Complex32, ComplexFloat};

#[derive(Debug, Clone)]
pub struct IirInterpolationFilter<T, Coeff = T> {
    m: usize,                     // interpolation factor
    iirfilt: IirFilter<T, Coeff>, // filter object
}

impl<T, Coeff> IirInterpolationFilter<T, Coeff>
where
    T: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + From<Coeff>,
    Coeff: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<T, Output = T> + Into<Complex32>,
    [T]: DotProd<Coeff, Output = T>,
    f32: Into<Coeff>,
{
    /// create interpolator from external coefficients
    pub fn new(
        interpolation_factor: usize,
        numerator_coefficients: &[Coeff],
        denominator_coefficients: &[Coeff],
    ) -> Result<Self> {
        // validate input
        if interpolation_factor < 2 {
            return Err(Error::Config("interpolation factor must be greater than 1".into()));
        }

        // create filter
        let iirfilt = IirFilter::new(numerator_coefficients, denominator_coefficients)?;

        Ok(IirInterpolationFilter { m: interpolation_factor, iirfilt })
    }

    /// Create an interpolator with a Chebyshev Type II prototype.
    pub fn new_chebyshev2(interpolation_factor: usize, order: usize) -> Result<Self> {
        Self::new_prototype(
            interpolation_factor,
            IirFilterShape::Cheby2,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            order,
            0.5 / interpolation_factor as f32,
            0.0,  // center frequency
            0.1,  // pass-band ripple
            60.0, // stop-band attenuation
        )
    }

    /// create interpolator from prototype
    pub fn new_prototype(
        interpolation_factor: usize,
        filter_shape: IirFilterShape,
        band_type: IirBandType,
        format: IirFormat,
        order: usize,
        cutoff_frequency: f32,
        center_frequency: f32,
        passband_ripple: f32,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        // validate input
        if interpolation_factor < 2 {
            return Err(Error::Config("interpolation factor must be greater than 1".into()));
        }

        // create filter
        let mut iirfilt = IirFilter::new_prototype(
            filter_shape,
            band_type,
            format,
            order,
            cutoff_frequency,
            center_frequency,
            passband_ripple,
            stopband_attenuation,
        )?;

        // set appropriate scale
        iirfilt.set_scale(Coeff::from(interpolation_factor).unwrap());

        Ok(IirInterpolationFilter { m: interpolation_factor, iirfilt })
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
    pub fn execute(&mut self, input: T, output: &mut [T]) -> Result<()> {
        if output.len() != self.m {
            return Err(Error::Config("output length must equal the interpolation factor".into()));
        }

        // TODO: use iirpfb
        for (i, yi) in output.iter_mut().enumerate() {
            *yi = self.iirfilt.execute(if i == 0 { input } else { T::default() });
        }
        Ok(())
    }

    /// execute interpolation on block of input samples
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        if output.len() != input.len() * self.m {
            return Err(Error::Config("output length must equal input length times the interpolation factor".into()));
        }

        for (i, &xi) in input.iter().enumerate() {
            self.execute(xi, &mut output[i * self.m..(i + 1) * self.m])?;
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
    use crate::fft::SpectralPeriodogram;
    use crate::filter::FirFilterShape;
    use crate::framing::ArbitraryRateSymbolStream;
    use crate::math::WindowType;
    use crate::modem::ModulationScheme;
    use crate::random::randnf;
    use crate::utility::test_helpers::{validate_psd_spectrum, PsdRegion};
    use test_macro::autotest_annotate;

    fn test_iirinterp_crcf(_method: &str, interp_factor: usize, order: usize) {
        // options
        let n = 800000; // number of output samples to analyze
        let bw = 0.2f32; // target output bandwidth
        let nfft = 800; // number of bins in transform
        let as_ = 60.0f32; // error tolerance [dB]
        let tol = 0.5f32; // error tolerance [dB]

        // create resampler with rate interp/decim
        let mut interp = IirInterpolationFilter::<Complex32, f32>::new_chebyshev2(interp_factor, order).unwrap();

        // create and configure objects
        let mut q = SpectralPeriodogram::<Complex32>::new(nfft, WindowType::Hann, nfft / 2, nfft / 4).unwrap();
        let mut gen = ArbitraryRateSymbolStream::new_linear(
            FirFilterShape::Kaiser,
            bw * interp_factor as f32,
            25,
            0.2,
            ModulationScheme::Qpsk,
        )
        .unwrap();
        gen.set_gain((bw as f32).sqrt());

        // generate samples and push through spgram object
        let block_size = 10;
        let mut buf_0 = vec![Complex32::default(); block_size];
        let mut buf_1 = vec![Complex32::default(); block_size * interp_factor];
        while q.num_samples_total() < n {
            // generate block of samples
            gen.write_samples(&mut buf_0).unwrap();

            // interpolate
            interp.execute_block(&buf_0, &mut buf_1).unwrap();

            // run samples through the spgram object
            q.write(&buf_1);
        }

        // verify result
        let psd = q.psd();
        #[rustfmt::skip]
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
    fn test_iirinterp_crcf_m2_o9() {
        test_iirinterp_crcf("baseline", 2, 9);
    }

    #[test]
    #[autotest_annotate(autotest_iirinterp_crcf_M3_O9)]
    fn test_iirinterp_crcf_m3_o9() {
        test_iirinterp_crcf("baseline", 3, 9);
    }

    #[test]
    #[autotest_annotate(autotest_iirinterp_crcf_M4_O9)]
    fn test_iirinterp_crcf_m4_o9() {
        test_iirinterp_crcf("baseline", 4, 9);
    }

    #[test]
    #[autotest_annotate(autotest_iirinterp_copy)]
    fn test_iirinterp_copy() {
        // create base object
        let mut q0 = IirInterpolationFilter::<Complex32, f32>::new_chebyshev2(3, 7).unwrap();
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
