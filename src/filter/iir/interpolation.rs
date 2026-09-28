use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::{IirBandType, IirFilter, IirFilterShape, IirFormat, Sample};
use num_complex::Complex32;

/// Infinite impulse response filter that interpolates by a fixed factor.
///
/// Each input sample produces `interpolation_factor` output samples. The
/// filter runs at the output rate and sees each input sample followed by
/// `interpolation_factor - 1` zeros. A new filter starts with zero-valued
/// history.
///
/// Inserting zeros lowers the output level. With a unit-gain filter, a
/// constant input produces an output of about `1 / interpolation_factor`
/// times its level. [`Self::new_prototype`] compensates by scaling the
/// output, and [`Self::new`] leaves the scaling to the caller.
///
/// Use [`Self::execute`] to process one input sample at a time, or
/// [`Self::execute_block`] to process many.
#[derive(Debug, Clone)]
pub struct IirInterpolationFilter<T, Coeff = T> {
    m: usize,                     // interpolation factor
    iirfilt: IirFilter<T, Coeff>, // filter object
}

impl<T, Coeff> IirInterpolationFilter<T, Coeff>
where
    T: Sample + std::ops::Mul<Coeff, Output = T> + From<Coeff>,
    Coeff: Sample + std::ops::Mul<T, Output = T> + Into<Complex32>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates an interpolation filter from transfer-function coefficients.
    ///
    /// The coefficients follow the conventions of [`IirFilter::new`] and run at
    /// the output rate. The output is not scaled, so unit-gain coefficients
    /// produce about `1 / interpolation_factor` of the input level. Multiply
    /// the numerator coefficients by `interpolation_factor` to restore it.
    ///
    /// # Errors
    ///
    /// Returns an error when `interpolation_factor` is less than two or either
    /// coefficient slice is empty.
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

    /// Creates a low-pass Chebyshev type II interpolation filter.
    ///
    /// The stopband begins at `0.5 / interpolation_factor` cycles per output
    /// sample, which is the Nyquist frequency of the input. Attenuation there
    /// is 60 dB. The passband ends below that edge by an amount that depends
    /// on `order`, so the top of the input band can be attenuated.
    ///
    /// The output is scaled by `interpolation_factor` to offset the level lost
    /// to zero insertion.
    ///
    /// # Errors
    ///
    /// Returns an error when `interpolation_factor` is less than two or
    /// `order` is zero.
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

    /// Creates an interpolation filter from a prototype design.
    ///
    /// Frequencies are in cycles per output sample. The design parameters are
    /// passed to [`IirFilter::new_prototype`], which describes their meaning.
    /// The output is scaled by `interpolation_factor` to offset the level lost
    /// to zero insertion.
    ///
    /// # Errors
    ///
    /// Returns an error when `interpolation_factor` is less than two or the
    /// prototype design rejects the parameters.
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
        iirfilt.set_scale(<Coeff as num_traits::NumCast>::from(interpolation_factor).unwrap());

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

    /// Clears the filter history to zeros.
    pub fn reset(&mut self) {
        self.iirfilt.reset();
    }

    /// Consumes one input sample and writes `interpolation_factor` outputs.
    ///
    /// `output` must hold exactly `interpolation_factor` samples.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` has any other length. The filter and
    /// `output` are unchanged on error.
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

    /// Consumes a block of input samples and writes `interpolation_factor`
    /// outputs for each.
    ///
    /// `output` must hold at least `input.len() * interpolation_factor`
    /// samples. Extra output is left unchanged. The results match calling
    /// [`Self::execute`] on each input sample. Returns the number of outputs
    /// written.
    ///
    /// # Errors
    ///
    /// Returns an error when `input.len() * interpolation_factor` overflows
    /// `usize` or `output` is too short. The filter and `output` are unchanged
    /// on error.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let required_output = input
            .len()
            .checked_mul(self.m)
            .ok_or_else(|| Error::Range("interpolation output length overflow".into()))?;
        if output.len() < required_output {
            return Err(Error::Config(
                "output length must be at least input length times the interpolation factor".into(),
            ));
        }

        for (i, &xi) in input.iter().enumerate() {
            self.execute(xi, &mut output[i * self.m..(i + 1) * self.m])?;
        }
        Ok(required_output)
    }

    /// Estimates the group delay at `fc` cycles per output sample.
    ///
    /// The delay is measured in input samples. The filter runs at the output
    /// rate, so its delay in output samples is divided by
    /// `interpolation_factor`. [`IirFilter::groupdelay`] describes the limits
    /// of the estimate.
    ///
    /// # Errors
    ///
    /// Returns an error when `fc` is outside `[-0.5, 0.5]`.
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

    #[test]
    fn test_iirinterp_execute_block_capacity() {
        let mut q = IirInterpolationFilter::<f32, f32>::new(3, &[1.0], &[1.0]).unwrap();
        let mut q_ref = q.clone();
        let input = [1.0, -0.5];
        assert!(q.execute_block(&input, &mut [0.0; 5]).is_err());

        let mut actual = [-9.0; 8];
        let mut expected = [0.0; 6];
        assert_eq!(q.execute_block(&input, &mut actual).unwrap(), 6);
        q_ref.execute_block(&input, &mut expected).unwrap();
        assert_eq!(&actual[..6], &expected);
        assert_eq!(&actual[6..], &[-9.0, -9.0]);
        assert_eq!(q.execute_block(&[], &mut actual).unwrap(), 0);
    }
}
