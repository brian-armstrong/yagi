use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::{IirBandType, IirFilter, IirFilterShape, IirFormat, Sample};
use num_complex::Complex32;

/// Infinite impulse response (IIR) decimation filter
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
    /// Create a new IIR decimation filter from external coefficients
    ///
    /// # Notes
    ///
    /// The number of feed-forward and feed-back coefficients do not need to be equal, but they do
    ///  need to be non-zero. Furthermore, the first feed-back coefficient \(a_0\) cannot be equal to
    ///  zero, otherwise the filter will be invalid as this value is factored out from all
    ///  coefficients.
    /// For stability reasons the number of coefficients should reasonably not exceed about 8 for
    ///  single-precision floating-point.
    ///
    /// # Arguments
    ///
    /// * `decimation_factor` - The decimation factor
    /// * `numerator_coefficients` - The feed-forward coefficients
    /// * `denominator_coefficients` - The feed-back coefficients
    ///
    /// # Returns
    ///
    /// A new IIR decimation filter
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

    /// Create a new IIR decimation filter with a default Butterworth prototype
    ///
    /// # Arguments
    ///
    /// * `decimation_factor` - The decimation factor
    /// * `order` - The filter order
    ///
    /// # Returns
    ///
    /// A new IIR decimation filter
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

    /// Create a new IIR decimation filter from a prototype
    ///
    /// # Arguments
    ///
    /// * `decimation_factor` - The decimation factor
    /// * `filter_shape` - The filter shape
    /// * `band_type` - The band type
    /// * `format` - The coefficients format
    /// * `order` - The filter order
    /// * `cutoff_frequency` - The low-pass prototype cut-off frequency
    /// * `center_frequency` - The center frequency
    /// * `passband_ripple` - The pass-band ripple
    /// * `stopband_attenuation` - The stop-band attenuation
    ///
    /// # Returns
    ///
    /// A new IIR decimation filter
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

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.iirfilt.reset();
    }

    /// Execute the filter on `decimation_factor` input samples
    ///
    /// # Arguments
    ///
    /// * `input` - The input samples
    ///
    /// # Returns
    ///
    /// The output sample
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

    /// Execute the filter on a block of input samples
    ///
    /// # Arguments
    ///
    /// * `input` - The input samples (size: `n * decimation_factor`)
    /// * `output` - The output samples (size: `n`)
    ///
    /// Returns the number of output samples written, `n`.
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

    /// Get the group delay at a given frequency
    ///
    /// # Arguments
    ///
    /// * `fc` - The frequency
    ///
    /// # Returns
    ///
    /// The group delay
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
