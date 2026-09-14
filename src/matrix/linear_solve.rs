use crate::error::{Error, Result};
use crate::matrix::view::{MatrixMut, MatrixRef};
use crate::matrix::{inverse::matrix_gjelim_view, FloatComplex};

/// Solve linear system of n equations: Ax = b
///
/// # Arguments
///
/// * `a` - System matrix [size: n x n]
/// * `n` - System size
/// * `b` - Equality vector [size: n x 1]
/// * `x` - Solution vector [size: n x 1]
/// * `scratch` - Optional augmented-matrix storage, at least `n * (n + 1)`
///   elements. When omitted, the solver allocates temporary storage.
///
/// # Returns
///
/// `Ok(())` if successful, `Err(...)` otherwise
pub fn matrix_linsolve<T>(a: &[T], n: usize, b: &[T], x: &mut [T], scratch: Option<&mut [T]>) -> Result<()>
where
    T: FloatComplex,
{
    let cols = n
        .checked_add(1)
        .ok_or_else(|| Error::Config("matrix_linsolve(), augmented matrix size overflow".into()))?;
    let scratch_len = n
        .checked_mul(cols)
        .ok_or_else(|| Error::Config("matrix_linsolve(), augmented matrix size overflow".into()))?;

    let a = MatrixRef::new(a, n, n)?;
    let b = MatrixRef::new(b, n, 1)?;
    let mut x = MatrixMut::new(x, n, 1)?;

    let mut owned = Vec::new();
    let scratch = match scratch {
        Some(scratch) => {
            if scratch.len() < scratch_len {
                return Err(Error::Config(format!(
                    "matrix_linsolve(), scratch length {} must be at least {}",
                    scratch.len(),
                    scratch_len
                )));
            }
            &mut scratch[..scratch_len]
        }
        None => {
            owned.resize(scratch_len, T::default());
            owned.as_mut_slice()
        }
    };

    let mut augmented = MatrixMut::new(scratch, n, cols)?;

    // Compute augmented matrix M [size: n x n+1]
    for ((a_row, b_row), augmented_row) in a.rows().zip(b.rows()).zip(augmented.rows_mut()) {
        augmented_row[..n].copy_from_slice(a_row);
        augmented_row[n] = b_row[0];
    }

    // Run Gauss-Jordan elimination on M
    matrix_gjelim_view(&mut augmented)?;

    // Copy result from right-most column of M
    for row_index in 0..n {
        x.row_mut(row_index)[0] = augmented.row(row_index)[n];
    }

    Ok(())
}
