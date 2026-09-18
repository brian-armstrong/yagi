use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, Zero};

use crate::error::Result;
use crate::matrix::view::{MatrixMut, MatrixRef};

/// Identify whether a scalar type has an imaginary component.
pub trait IsComplex {
    /// Return true for complex scalar types.
    fn is_complex() -> bool;
}

impl IsComplex for f32 {
    fn is_complex() -> bool {
        false
    }
}

impl IsComplex for f64 {
    fn is_complex() -> bool {
        false
    }
}

impl IsComplex for Complex<f32> {
    fn is_complex() -> bool {
        true
    }
}

impl IsComplex for Complex<f64> {
    fn is_complex() -> bool {
        true
    }
}

/// Scalar types accepted by the dense matrix routines.
pub trait FloatComplex:
    ComplexFloat<Real: Float + Zero + From<f32> + std::fmt::LowerExp>
    + std::fmt::Display
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + Default
    + std::iter::Sum
    + IsComplex
{
}

impl<
        T: ComplexFloat<Real: Float + Zero + From<f32> + std::fmt::LowerExp>
            + std::fmt::Display
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + Default
            + std::iter::Sum
            + IsComplex,
    > FloatComplex for T
{
}

/// Print a dense matrix to standard output.
///
/// The output is intended for inspection and debugging. Its format is not a
/// stable serialization format.
pub fn matrix_print<T>(x: MatrixRef<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (rows, cols) = x.shape();
    println!("matrix [{} x {}] :", rows, cols);
    for row in x.rows() {
        for value in row {
            print!("{:12}", value);
        }
        println!();
    }
    Ok(())
}

/// Replace a square matrix with the identity matrix.
///
/// # Errors
///
/// Returns an error if `x` is not square. `x` is unchanged on error.
pub fn matrix_eye<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (rows, cols) = x.shape();
    if rows != cols {
        return Err(crate::error::Error::Range("matrix_eye(), matrix must be square".to_string()));
    }
    x.as_mut_slice().fill(T::zero());
    for (row_index, row) in x.rows_mut().enumerate() {
        row[row_index] = T::one();
    }
    Ok(())
}

/// Replace every element of a matrix with one.
pub fn matrix_ones<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    x.as_mut_slice().fill(T::one());
    Ok(())
}

/// Replace every element of a matrix with zero.
pub fn matrix_zeros<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    x.as_mut_slice().fill(T::zero());
    Ok(())
}
