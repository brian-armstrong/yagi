use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, Zero};

use crate::error::Result;
use crate::matrix::view::{MatrixMut, MatrixRef};

/// Extension trait for complex number support
pub trait IsComplex {
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

/// Print matrix to stdout
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

/// Initialize square matrix to the identity matrix
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

/// Initialize matrix to ones
pub fn matrix_ones<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    x.as_mut_slice().fill(T::one());
    Ok(())
}

/// Initialize matrix to zeros
pub fn matrix_zeros<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    x.as_mut_slice().fill(T::zero());
    Ok(())
}
