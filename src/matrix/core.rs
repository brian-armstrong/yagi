use num_complex::ComplexFloat;
use num_traits::{Float, Zero};

use crate::error::Result;
use crate::matrix::{MatrixMut, MatrixRef};

/// Scalar types accepted by the dense matrix routines.
///
/// Implemented for every type that meets the bounds, which includes `f32`,
/// `f64`, `Complex32` and `Complex64`.
pub trait FloatComplex:
    ComplexFloat<Real: Float + Zero + From<f32> + std::fmt::LowerExp>
    + std::fmt::Display
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + Default
    + std::iter::Sum
{
}

impl<
        T: ComplexFloat<Real: Float + Zero + From<f32> + std::fmt::LowerExp>
            + std::fmt::Display
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + Default
            + std::iter::Sum,
    > FloatComplex for T
{
}

/// Prints a dense matrix to standard output.
///
/// The output is meant for inspection and debugging. Its format may change.
/// This never returns an error.
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

/// Overwrites a square matrix with the identity matrix.
///
/// # Errors
///
/// Returns an error when `x` is not square. `x` is unchanged on error.
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

/// Sets every element of a matrix to one.
///
/// This never returns an error.
pub fn matrix_ones<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    x.as_mut_slice().fill(T::one());
    Ok(())
}

/// Sets every element of a matrix to zero.
///
/// This never returns an error.
pub fn matrix_zeros<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    x.as_mut_slice().fill(T::zero());
    Ok(())
}
