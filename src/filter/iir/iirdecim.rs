use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter::iir::iirfilt::IirFilter;
use crate::filter::iir::design::{IirFilterShape, IirBandType, IirFormat};
use num_complex::{ComplexFloat, Complex32};

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
    /// * `b` - The feed-forward coefficients
    /// * `a` - The feed-back coefficients
    /// 
    /// # Returns
    /// 
    /// A new IIR decimation filter
    pub fn new(decimation_factor: usize, b: &[Coeff], a: &[Coeff]) -> Result<Self> {
        if decimation_factor < 2 {
            return Err(Error::Config("decimation factor must be greater than 1".into()));
        }

        let iirfilt = IirFilter::new(b, a)?;

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
    pub fn new_default(decimation_factor: usize, order: usize) -> Result<Self> {
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
    /// * `ftype` - The filter type
    /// * `btype` - The band type
    /// * `format` - The coefficients format
    /// * `order` - The filter order
    /// * `fc` - The low-pass prototype cut-off frequency
    /// * `f0` - The center frequency
    /// * `ap` - The pass-band ripple
    /// * `as_` - The stop-band ripple
    /// 
    /// # Returns
    /// 
    /// A new IIR decimation filter
    pub fn new_prototype(
        decimation_factor: usize,
        ftype: IirFilterShape,
        btype: IirBandType,
        format: IirFormat,
        order: usize,
        fc: f32,
        f0: f32,
        ap: f32,
        as_: f32,
    ) -> Result<Self> {
        if decimation_factor < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }

        let iirfilt = IirFilter::new_prototype(ftype, btype, format, order, fc, f0, ap, as_)?;

        Ok(Self { decimation_factor, iirfilt })
    }

    /// Reset the filter state
    pub fn reset(&mut self) -> () {
        self.iirfilt.reset();
    }

    /// Execute the filter on `decimation_factor` input samples
    /// 
    /// # Arguments
    /// 
    /// * `x` - The input samples
    /// 
    /// # Returns
    /// 
    /// The output sample
    pub fn execute(&mut self, x: &[T]) -> T {
        let mut y = T::default();
        for (i, &xi) in x.iter().enumerate() {
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
    /// * `x` - The input samples (size: `n * decimation_factor`)
    /// * `y` - The output samples (size: `n`)
    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> () {
        for (i, xi) in x.chunks(self.decimation_factor).enumerate() {
            y[i] = self.execute(xi);
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
    use test_macro::autotest_annotate;
    use crate::random::randnf;

    #[test]
    #[autotest_annotate(autotest_iirdecim_copy)]
    fn test_iirdecim_copy() {
        // create base object
        let mut q0 = IirDecimationFilter::<Complex32, f32>::new_default(3, 7).unwrap();

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