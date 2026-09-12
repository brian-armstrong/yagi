use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::iir::design::{IirBandType, IirFilterShape, IirFormat};
use crate::filter::iir::iirfilt::IirFilter;
use num_complex::{Complex32, ComplexFloat};

/// Infinite impulse response (IIR) decimation filter
#[derive(Clone, Debug)]
pub struct IirDecimationFilter<T, Coeff = T> {
    decimation_factor: usize,
    iirfilt: IirFilter<T, Coeff>,
}

impl<T, Coeff> IirDecimationFilter<T, Coeff>
where
    T: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + From<Coeff>,
    Coeff: Copy + Default + ComplexFloat<Real = f32> + std::ops::Mul<T, Output = T> + Into<Complex32>,
    [T]: DotProd<Coeff, Output = T>,
    f32: Into<Coeff>,
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
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) {
        for (i, xi) in input.chunks(self.decimation_factor).enumerate() {
            output[i] = self.execute(xi);
        }
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
}
