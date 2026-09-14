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
pub fn matrix_print<T>(x: &[T], rows: usize, cols: usize) -> Result<()>
where
    T: FloatComplex,
{
    let matrix = MatrixRef::new(x, rows, cols)?;
    println!("matrix [{} x {}] :", rows, cols);
    for row in matrix.rows() {
        for value in row {
            print!("{:12}", value);
        }
        println!();
    }
    Ok(())
}

/// Initialize square matrix to the identity matrix
pub fn matrix_eye<T>(x: &mut [T], n: usize) -> Result<()>
where
    T: FloatComplex,
{
    let mut matrix = MatrixMut::new(x, n, n)?;
    matrix.as_mut_slice().fill(T::zero());
    for (row_index, row) in matrix.rows_mut().enumerate() {
        row[row_index] = T::one();
    }
    Ok(())
}

/// Initialize matrix to ones
pub fn matrix_ones<T>(x: &mut [T], rows: usize, cols: usize) -> Result<()>
where
    T: FloatComplex,
{
    MatrixMut::new(x, rows, cols)?.as_mut_slice().fill(T::one());
    Ok(())
}

/// Initialize matrix to zeros
pub fn matrix_zeros<T>(x: &mut [T], rows: usize, cols: usize) -> Result<()>
where
    T: FloatComplex,
{
    MatrixMut::new(x, rows, cols)?.as_mut_slice().fill(T::zero());
    Ok(())
}

/// Helper function to access matrix elements
#[inline]
pub fn matrix_access<T>(matrix: &[T], _rows: usize, cols: usize, r: usize, c: usize) -> T
where
    T: FloatComplex,
{
    matrix[r * cols + c]
}

/// Helper function to mutably access matrix elements
#[inline]
pub fn matrix_access_mut<T>(matrix: &mut [T], _rows: usize, cols: usize, r: usize, c: usize, value: T) {
    matrix[r * cols + c] = value;
}
