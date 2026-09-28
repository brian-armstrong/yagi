use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::{IirBandType, IirFilter, IirFilterShape, IirFormat, Sample};
use num_complex::Complex32;

/// Infinite impulse response filter that decimates by a fixed factor.
///
/// The filter consumes input in groups of `decimation_factor` samples and
/// produces one output per group. Each output depends on the first sample of
/// its group and on earlier input. The remaining samples in the group only
/// affect later outputs. A new filter starts with zero-valued history.
///
/// Use [`Self::execute`] to process one group at a time, or
/// [`Self::execute_block`] to process many.
#[derive(Clone, Debug)]
pub struct IirDecimationFilter<T, Coeff = T> {
    decimation_factor: usize,
    iirfilt: IirFilter<T, Coeff>,
}

impl<T, Coeff> IirDecimationFilter<T, Coeff>
where
    T: Sample + std::ops::Mul<Coeff, Output = T> + From<Coeff>,
    Coeff: Sample + std::ops::Mul<T, Output = T> + Into<Complex32>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates a decimation filter from transfer-function coefficients.
    ///
    /// The coefficients follow the conventions of [`IirFilter::new`] and run at
    /// the input rate.
    ///
    /// # Errors
    ///
    /// Returns an error when `decimation_factor` is less than two or either
    /// coefficient slice is empty.
    pub fn new(
        decimation_factor: usize,
        numerator_coefficients: &[Coeff],
        denominator_coefficients: &[Coeff],
    ) -> Result<Self> {
        if decimation_factor < 2 {
            return Err(Error::Config("decimation factor must be greater than 1".into()));
        }

        let iirfilt = IirFilter::new(numerator_coefficients, denominator_coefficients)?;

        Ok(Self { decimation_factor, iirfilt })
    }

    /// Creates a low-pass Butterworth decimation filter.
    ///
    /// The -3 dB point is `0.5 / decimation_factor` cycles per input sample,
    /// which is the Nyquist frequency of the output. Attenuation rises only
    /// gradually above that edge, at a rate set by `order`, so input near the
    /// edge can alias. When alias rejection matters, use
    /// [`Self::new_prototype`] with a lower cutoff or a sharper shape.
    ///
    /// # Errors
    ///
    /// Returns an error when `decimation_factor` is less than two or `order`
    /// is zero.
    pub fn new_butterworth(decimation_factor: usize, order: usize) -> Result<Self> {
        Self::new_prototype(
            decimation_factor,
            IirFilterShape::Butter,
            IirBandType::Lowpass,
            IirFormat::SecondOrderSections,
            order,
            0.5 / decimation_factor as f32,
            0.0,
            0.1,
            60.0,
        )
    }

    /// Creates a decimation filter from a prototype design.
    ///
    /// Frequencies are in cycles per input sample. The design parameters are
    /// passed to [`IirFilter::new_prototype`], which describes their meaning.
    ///
    /// # Errors
    ///
    /// Returns an error when `decimation_factor` is less than two or the
    /// prototype design rejects the parameters.
    pub fn new_prototype(
        decimation_factor: usize,
        filter_shape: IirFilterShape,
        band_type: IirBandType,
        format: IirFormat,
        order: usize,
        cutoff_frequency: f32,
        center_frequency: f32,
        passband_ripple: f32,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        if decimation_factor < 2 {
            return Err(Error::Config("decimation factor must be greater than 1".into()));
        }

        let iirfilt = IirFilter::new_prototype(
            filter_shape,
            band_type,
            format,
            order,
            cutoff_frequency,
            center_frequency,
            passband_ripple,
            stopband_attenuation,
        )?;

        Ok(Self { decimation_factor, iirfilt })
    }

    /// Clears the filter history to zeros.
    pub fn reset(&mut self) {
        self.iirfilt.reset();
    }

    /// Consumes one group of input samples and returns one output sample.
    ///
    /// A group should hold `decimation_factor` samples, but the length is not
    /// checked. Every sample in `input` passes through the filter, and the
    /// result is the response to the first one. Empty input returns zero and
    /// leaves the filter unchanged. Use [`Self::execute_block`] when groups
    /// must be complete.
    pub fn execute(&mut self, input: &[T]) -> T {
        let mut y = T::default();
        for (i, &xi) in input.iter().enumerate() {
            let v = self.iirfilt.execute(xi);
            if i == 0 {
                y = v;
            }
        }
        y
    }

    /// Consumes complete groups of input samples and writes one output per
    /// group.
    ///
    /// `input.len()` must be a multiple of `decimation_factor`, and `output`
    /// must hold at least `input.len() / decimation_factor` samples. Extra
    /// output is left unchanged. The results match calling [`Self::execute`]
    /// on each group. Returns the number of outputs written.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` does not hold a whole number of groups or
    /// `output` is too short. The filter and `output` are unchanged on error.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if !input.len().is_multiple_of(self.decimation_factor) {
            return Err(Error::Config("input length must be a multiple of the decimation factor".into()));
        }
        let n = input.len() / self.decimation_factor;
        if output.len() < n {
            return Err(Error::Config("output length must be at least the number of output samples".into()));
        }
        for (i, xi) in input.chunks(self.decimation_factor).enumerate() {
            output[i] = self.execute(xi);
        }
        Ok(n)
    }

    /// Estimates the group delay at `fc` cycles per input sample.
    ///
    /// The delay is measured in input samples. [`IirFilter::groupdelay`]
    /// describes the limits of the estimate.
    ///
    /// # Errors
    ///
    /// Returns an error when `fc` is outside `[-0.5, 0.5]`.
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        self.iirfilt.groupdelay(fc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_iirdecim_copy)]
    fn test_iirdecim_copy() {
        // create base object
        let mut q0 = IirDecimationFilter::<Complex32, f32>::new_butterworth(3, 7).unwrap();

        // run samples through filter
        let mut buf = [Complex32::default(); 3];
        for _ in 0..20 {
            for j in 0..3 {
                buf[j] = Complex32::new(randnf(), randnf());
            }
            let _y0 = q0.execute(&buf);
        }

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters in parallel
        for _ in 0..60 {
            for j in 0..3 {
                buf[j] = Complex32::new(randnf(), randnf());
            }
            let y0 = q0.execute(&buf);
            let y1 = q1.execute(&buf);

            assert_eq!(y0, y1);
        }

        // objects are automatically destroyed when they go out of scope
    }

    #[test]
    fn test_iirdecim_execute_block_output_capacity() {
        let mut q = IirDecimationFilter::<Complex32, f32>::new_butterworth(3, 7).unwrap();
        let x = vec![Complex32::new(1.0, 0.0); 12]; // 4 output samples

        // input not a multiple of the decimation factor is an error
        assert!(q.execute_block(&x[..11], &mut [Complex32::default(); 8]).is_err());

        // undersized output is an error
        assert!(q.execute_block(&x, &mut [Complex32::default(); 3]).is_err());

        // oversized output is accepted
        let mut y = vec![Complex32::new(-9.0, -9.0); 8];
        assert_eq!(q.execute_block(&x, &mut y).unwrap(), 4);
        assert_eq!(y[4], Complex32::new(-9.0, -9.0));
        assert_eq!(y[7], Complex32::new(-9.0, -9.0));
    }
}
