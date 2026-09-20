//! Slice-based vector operations.
//!
//! These functions operate on caller-provided slices, with separate in-place
//! variants where appropriate. Elementwise arithmetic is generic over its
//! required scalar operations. Norm, magnitude, and phase operations support
//! `f32` and `Complex32` samples.

use num_complex::{Complex32, ComplexFloat};
use std::f32::consts::PI;
use std::ops::{Add, Mul};

mod private {
    use num_complex::Complex32;

    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for Complex32 {}
}

/// Element type supported by vector operations whose result type differs from
/// their input type.
///
/// This trait is sealed. Yagi currently supports `f32` and `Complex32`.
pub trait VectorElement: Copy + private::Sealed {
    /// One element of [`expj`]: complex phase rotation `exp{ j theta }`.
    fn expj(theta: f32) -> Self;

    /// One element of [`arg`]: angle `arg{ x }`.
    fn arg(self) -> f32;

    /// One element of [`abs`]: absolute value `|x|`.
    fn abs(self) -> f32;
}

impl VectorElement for Complex32 {
    fn expj(theta: f32) -> Self {
        Complex32::from_polar(1.0, theta)
    }

    fn arg(self) -> f32 {
        ComplexFloat::arg(self)
    }

    fn abs(self) -> f32 {
        ComplexFloat::abs(self)
    }
}

impl VectorElement for f32 {
    fn expj(theta: f32) -> Self {
        if theta > 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    fn arg(self) -> f32 {
        if self > 0.0 {
            0.0
        } else {
            PI
        }
    }

    fn abs(self) -> f32 {
        f32::abs(self)
    }
}

/// Fill a vector with a scalar: `values[i] = value`.
///
/// # Arguments
/// * `value` - scalar value
/// * `values` - vector to fill
pub fn fill<T>(value: T, values: &mut [T])
where
    T: Copy,
{
    for sample in values.iter_mut() {
        *sample = value;
    }
}

/// Add each element pointwise: `output[i] = left[i] + right[i]`.
///
/// # Arguments
/// * `left` - first input vector
/// * `right` - second input vector
/// * `output` - output vector
///
/// # Panics
/// Panics if the vectors do not all have the same length.
pub fn add<T>(left: &[T], right: &[T], output: &mut [T])
where
    T: Copy + Add<Output = T>,
{
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), output.len());

    for i in 0..left.len() {
        output[i] = left[i] + right[i];
    }
}

/// Add each element of `addend` to `values` in place.
///
/// `values[i] = values[i] + addend[i]`.
///
/// # Panics
///
/// Panics if `values` and `addend` do not have the same length.
pub fn add_in_place<T>(values: &mut [T], addend: &[T])
where
    T: Copy + Add<Output = T>,
{
    assert_eq!(values.len(), addend.len());

    for i in 0..values.len() {
        values[i] = values[i] + addend[i];
    }
}

/// Add a scalar to each element: `output[i] = input[i] + value`.
///
/// # Arguments
/// * `input` - input vector
/// * `value` - scalar value
/// * `output` - output vector
///
/// # Panics
/// Panics if `input` and `output` do not have the same length.
pub fn add_scalar<T>(input: &[T], value: T, output: &mut [T])
where
    T: Copy + Add<Output = T>,
{
    assert_eq!(input.len(), output.len());

    for i in 0..input.len() {
        output[i] = input[i] + value;
    }
}

/// Add a scalar to every element of `values` in place.
pub fn add_scalar_in_place<T>(values: &mut [T], value: T)
where
    T: Copy + Add<Output = T>,
{
    for sample in values.iter_mut() {
        *sample = *sample + value;
    }
}

/// Multiply each element pointwise: `output[i] = left[i] * right[i]`.
///
/// # Arguments
/// * `left` - first input vector
/// * `right` - second input vector
/// * `output` - output vector
///
/// # Panics
/// Panics if the vectors do not all have the same length.
pub fn multiply<T>(left: &[T], right: &[T], output: &mut [T])
where
    T: Copy + Mul<Output = T>,
{
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), output.len());

    for i in 0..left.len() {
        output[i] = left[i] * right[i];
    }
}

/// Multiply every element of `values` by the corresponding `factor` in place.
///
/// `values[i] = values[i] * factor[i]`.
///
/// # Panics
///
/// Panics if `values` and `factor` do not have the same length.
pub fn multiply_in_place<T>(values: &mut [T], factor: &[T])
where
    T: Copy + Mul<Output = T>,
{
    assert_eq!(values.len(), factor.len());

    for i in 0..values.len() {
        values[i] = values[i] * factor[i];
    }
}

/// Multiply each element by a scalar: `output[i] = input[i] * value`.
///
/// # Arguments
/// * `input` - input vector
/// * `value` - scalar value
/// * `output` - output vector
///
/// # Panics
/// Panics if `input` and `output` do not have the same length.
pub fn multiply_scalar<T>(input: &[T], value: T, output: &mut [T])
where
    T: Copy + Mul<Output = T>,
{
    assert_eq!(input.len(), output.len());

    for i in 0..input.len() {
        output[i] = input[i] * value;
    }
}

/// Multiply every element of `values` by a scalar in place.
pub fn multiply_scalar_in_place<T>(values: &mut [T], value: T)
where
    T: Copy + Mul<Output = T>,
{
    for sample in values.iter_mut() {
        *sample = *sample * value;
    }
}

/// Compute the l2-norm of a vector: `sqrt{ sum{ |values[i]|^2 } }`.
///
/// # Arguments
/// * `values` - input vector
pub fn norm<T>(values: &[T]) -> f32
where
    T: Copy + ComplexFloat<Real = f32>,
{
    // initialize accumulator
    let mut norm = 0.0f32;

    for &xi in values {
        norm += (xi * xi.conj()).re();
    }

    // return square root of accumulation
    norm.sqrt()
}

/// Scale a vector to its l2-norm: `output[i] = input[i] / norm(input)`.
///
/// # Arguments
/// * `input` - input vector
/// * `output` - output vector
///
/// # Panics
/// Panics if `input` and `output` do not have the same length.
pub fn normalize<T>(input: &[T], output: &mut [T])
where
    T: Copy + ComplexFloat<Real = f32> + From<f32>,
{
    assert_eq!(input.len(), output.len());

    // compute l2-norm on vector
    let norm = norm(input);

    // compute inverse
    let norm_inv = <T as From<f32>>::from(1.0 / norm);

    // scale by inverse
    for i in 0..input.len() {
        output[i] = input[i] * norm_inv;
    }
}

/// Scale `values` to unit l2-norm in place.
pub fn normalize_in_place<T>(values: &mut [T])
where
    T: Copy + ComplexFloat<Real = f32> + From<f32>,
{
    let norm_inv = <T as From<f32>>::from(1.0 / norm(values));

    for sample in values.iter_mut() {
        *sample = *sample * norm_inv;
    }
}

/// Compute the sum of squares: `sum{ |values[i]|^2 }`.
///
/// # Arguments
/// * `values` - input vector
pub fn sum_squares<T>(values: &[T]) -> f32
where
    T: Copy + ComplexFloat<Real = f32>,
{
    let mut sum = 0.0f32;

    for &xi in values {
        sum += (xi * xi.conj()).re();
    }

    sum
}

/// Compute the l-p norm: `{ sum{ |values[i]|^p } }^(1/p)`.
///
/// # Arguments
/// * `values` - input vector
/// * `order` - finite, positive norm order
///
/// # Panics
///
/// Panics if `order` is not finite and positive.
pub fn p_norm<T>(values: &[T], order: f32) -> f32
where
    T: Copy + ComplexFloat<Real = f32>,
{
    assert!(order.is_finite() && order > 0.0, "norm order must be finite and positive");

    let mut sum = 0.0f32;

    for &xi in values {
        sum += xi.abs().powf(order);
    }

    sum.powf(1.0 / order)
}

/// Compute complex phase rotation: `output[i] = exp{ j angles[i] }`.
///
/// For the real type this degenerates to the sign of the angle:
/// `output[i] = angles[i] > 0 ? 1 : -1`.
///
/// # Arguments
/// * `angles` - input angle vector, in radians
/// * `output` - output vector
///
/// # Panics
/// Panics if `angles` and `output` do not have the same length.
pub fn expj<T: VectorElement>(angles: &[f32], output: &mut [T]) {
    assert_eq!(angles.len(), output.len());

    for i in 0..angles.len() {
        output[i] = T::expj(angles[i]);
    }
}

/// Compute angle of each element: `angles[i] = arg{ input[i] }`.
///
/// For the real type this degenerates to `angles[i] = input[i] > 0 ? 0 : pi`.
///
/// # Arguments
/// * `input` - input vector
/// * `angles` - output angle vector, in radians
///
/// # Panics
/// Panics if `input` and `angles` do not have the same length.
pub fn arg<T: VectorElement>(input: &[T], angles: &mut [f32]) {
    assert_eq!(input.len(), angles.len());

    for i in 0..input.len() {
        angles[i] = input[i].arg();
    }
}

/// Compute absolute value of each element: `output[i] = |input[i]|`.
///
/// # Arguments
/// * `input` - input vector
/// * `output` - output vector
///
/// # Panics
/// Panics if `input` and `output` do not have the same length.
pub fn abs<T: VectorElement>(input: &[T], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());

    for i in 0..input.len() {
        output[i] = input[i].abs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;

    #[test]
    fn test_add_f32() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut z = [0.0f32; 5];
        add(&x, &y, &mut z);
        assert_eq!(z, [11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_add_cf32() {
        let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let y = [Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];
        let mut z = [Complex32::default(); 2];
        add(&x, &y, &mut z);
        assert_eq!(z, [Complex32::new(6.0, 8.0), Complex32::new(10.0, 12.0)]);
    }

    #[test]
    fn test_add_scalar_f32() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [0.0f32; 3];
        add_scalar(&x, 0.5, &mut y);
        assert_eq!(y, [1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_add_scalar_cf32() {
        let x = [Complex32::new(1.0, 1.0), Complex32::new(2.0, 2.0)];
        let mut y = [Complex32::default(); 2];
        add_scalar(&x, Complex32::new(0.0, 1.0), &mut y);
        assert_eq!(y, [Complex32::new(1.0, 2.0), Complex32::new(2.0, 3.0)]);
    }

    #[test]
    fn test_add_in_place() {
        let mut values = [1.0f32, 2.0, 3.0];
        add_in_place(&mut values, &[4.0, 5.0, 6.0]);
        assert_eq!(values, [5.0, 7.0, 9.0]);

        add_scalar_in_place(&mut values, 0.5);
        assert_eq!(values, [5.5, 7.5, 9.5]);
    }

    #[test]
    fn test_multiply_f32() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut z = [0.0f32; 5];
        multiply(&x, &y, &mut z);
        assert_eq!(z, [10.0, 40.0, 90.0, 160.0, 250.0]);
    }

    #[test]
    fn test_multiply_cf32() {
        // (1+2j)(5+6j) = 5+6j+10j-12 = -7+16j
        // (3+4j)(7+8j) = 21+24j+28j-32 = -11+52j
        let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let y = [Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];
        let mut z = [Complex32::default(); 2];
        multiply(&x, &y, &mut z);
        assert_eq!(z, [Complex32::new(-7.0, 16.0), Complex32::new(-11.0, 52.0)]);
    }

    #[test]
    fn test_multiply_scalar_f32() {
        let x = [1.0f32, 2.0, 3.0];
        let mut y = [0.0f32; 3];
        multiply_scalar(&x, 2.0, &mut y);
        assert_eq!(y, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_multiply_scalar_cf32() {
        // multiply by j rotates by 90 degrees: (a+bj)*j = -b+aj
        let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let mut y = [Complex32::default(); 2];
        multiply_scalar(&x, Complex32::new(0.0, 1.0), &mut y);
        assert_eq!(y, [Complex32::new(-2.0, 1.0), Complex32::new(-4.0, 3.0)]);
    }

    #[test]
    fn test_multiply_in_place() {
        let mut values = [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        multiply_in_place(&mut values, &[Complex32::new(0.0, 1.0), Complex32::new(0.0, 1.0)]);
        assert_eq!(values, [Complex32::new(-2.0, 1.0), Complex32::new(-4.0, 3.0)]);

        multiply_scalar_in_place(&mut values, Complex32::new(0.0, -1.0));
        assert_eq!(values, [Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)]);
    }

    #[test]
    fn test_norm_f32() {
        // sqrt(3^2 + 4^2) = 5
        let x = [3.0f32, 4.0];
        assert_abs_diff_eq!(norm(&x), 5.0, epsilon = 1e-6);

        // sqrt(1+1+1+1) = 2
        let x = [1.0f32, 1.0, 1.0, 1.0];
        assert_abs_diff_eq!(norm(&x), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_norm_cf32() {
        // |3+4j| = 5, |0+0j| = 0, so norm = 5
        let x = [Complex32::new(3.0, 4.0), Complex32::new(0.0, 0.0)];
        assert_abs_diff_eq!(norm(&x), 5.0, epsilon = 1e-6);

        let x = [Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)];
        assert_abs_diff_eq!(norm(&x), 2.0f32.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_normalize_f32() {
        let x = [3.0f32, 4.0, 12.0];
        let mut y = [0.0f32; 3];
        normalize(&x, &mut y);
        assert_abs_diff_eq!(norm(&y), 1.0, epsilon = 1e-6);
        let n = norm(&x);
        for i in 0..x.len() {
            assert_abs_diff_eq!(y[i], x[i] / n, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_normalize_cf32() {
        let x = [Complex32::new(1.0, 2.0), Complex32::new(3.0, -4.0)];
        let mut y = [Complex32::default(); 2];
        normalize(&x, &mut y);
        assert_abs_diff_eq!(norm(&y), 1.0, epsilon = 1e-6);
        let n = norm(&x);
        for i in 0..x.len() {
            assert_abs_diff_eq!((y[i] - x[i] / n).norm(), 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_normalize_in_place() {
        let mut values = [3.0f32, 4.0];
        normalize_in_place(&mut values);
        assert_abs_diff_eq!(norm(&values), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(values[0], 0.6, epsilon = 1e-6);
        assert_abs_diff_eq!(values[1], 0.8, epsilon = 1e-6);
    }

    #[test]
    fn test_expj_cf32() {
        let theta = [0.0f32, PI / 2.0, PI, -PI / 2.0];
        let mut x = [Complex32::default(); 4];
        expj(&theta, &mut x);
        let expected = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, -1.0),
        ];
        for i in 0..theta.len() {
            assert_abs_diff_eq!((x[i] - expected[i]).norm(), 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_expj_f32() {
        let theta = [1.0f32, -1.0, 0.0, 100.0];
        let mut x = [0.0f32; 4];
        expj(&theta, &mut x);
        assert_eq!(x, [1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_arg_cf32() {
        let x = [
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(-1.0, 0.0),
            Complex32::new(0.0, -1.0),
        ];
        let mut theta = [0.0f32; 4];
        arg(&x, &mut theta);
        assert_abs_diff_eq!(theta[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(theta[1], PI / 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(theta[2], PI, epsilon = 1e-6);
        assert_abs_diff_eq!(theta[3], -PI / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_arg_f32() {
        let x = [2.0f32, -3.0, 0.0];
        let mut theta = [0.0f32; 3];
        arg(&x, &mut theta);
        assert_eq!(theta, [0.0, PI, PI]);
    }

    #[test]
    fn test_expj_arg_roundtrip() {
        let theta = [0.1f32, -0.7, 1.3, 2.9, -2.9];
        let mut x = [Complex32::default(); 5];
        let mut out = [0.0f32; 5];
        expj(&theta, &mut x);
        arg(&x, &mut out);
        for i in 0..theta.len() {
            assert_abs_diff_eq!(out[i], theta[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_abs_cf32() {
        let x = [Complex32::new(3.0, 4.0), Complex32::new(0.0, 0.0), Complex32::new(-5.0, 12.0)];
        let mut y = [0.0f32; 3];
        abs(&x, &mut y);
        assert_abs_diff_eq!(y[0], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y[2], 13.0, epsilon = 1e-6);
    }

    #[test]
    fn test_abs_f32() {
        let x = [-1.0f32, 2.0, -3.5, 0.0];
        let mut y = [99.0f32; 4];
        abs(&x, &mut y);
        assert_eq!(y, [1.0, 2.0, 3.5, 0.0]);
        assert_eq!(x, [-1.0, 2.0, -3.5, 0.0]);
    }

    #[test]
    fn test_fill_f32() {
        let mut x = [0.0f32; 4];
        fill(3.5, &mut x);
        assert_eq!(x, [3.5, 3.5, 3.5, 3.5]);
    }

    #[test]
    fn test_fill_cf32() {
        let mut x = [Complex32::default(); 3];
        fill(Complex32::new(1.0, -2.0), &mut x);
        assert_eq!(x, [Complex32::new(1.0, -2.0); 3]);
    }

    #[test]
    fn test_sum_squares_f32() {
        let x = [3.0f32, 4.0];
        // 9 + 16 = 25
        assert_abs_diff_eq!(sum_squares(&x), 25.0, epsilon = 1e-6);
        // sumsq == norm^2
        assert_abs_diff_eq!(sum_squares(&x), norm(&x).powi(2), epsilon = 1e-5);
    }

    #[test]
    fn test_sum_squares_cf32() {
        let x = [Complex32::new(3.0, 4.0), Complex32::new(0.0, 5.0)];
        // |3+4j|^2 + |5j|^2 = 25 + 25 = 50
        assert_abs_diff_eq!(sum_squares(&x), 50.0, epsilon = 1e-5);
        // sumsq == norm^2
        assert_abs_diff_eq!(sum_squares(&x), norm(&x).powi(2), epsilon = 1e-4);
    }

    #[test]
    fn test_p_norm_f32() {
        let x = [3.0f32, -4.0];
        // p=2 matches the l2-norm
        assert_abs_diff_eq!(p_norm(&x, 2.0), norm(&x), epsilon = 1e-5);
        // p=1 is the sum of magnitudes: 3 + 4 = 7
        assert_abs_diff_eq!(p_norm(&x, 1.0), 7.0, epsilon = 1e-5);
    }

    #[test]
    fn test_p_norm_cf32() {
        let x = [Complex32::new(3.0, 4.0), Complex32::new(0.0, -5.0)];
        // p=2 matches the l2-norm
        assert_abs_diff_eq!(p_norm(&x, 2.0), norm(&x), epsilon = 1e-4);
        // p=1 is the sum of magnitudes: 5 + 5 = 10
        assert_abs_diff_eq!(p_norm(&x, 1.0), 10.0, epsilon = 1e-5);
    }

    #[test]
    #[should_panic(expected = "norm order must be finite and positive")]
    fn test_p_norm_rejects_non_positive_order() {
        p_norm(&[1.0f32, 2.0], 0.0);
    }

    #[test]
    #[should_panic]
    fn test_add_length_mismatch() {
        let x = [1.0f32, 2.0];
        let y = [1.0f32, 2.0, 3.0];
        let mut z = [0.0f32; 2];
        add(&x, &y, &mut z);
    }
}
