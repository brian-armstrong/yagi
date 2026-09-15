use crate::error::{Error, Result};
use crate::matrix::view::{MatrixMut, MatrixRef};
use crate::matrix::{matrix_gjelim, FloatComplex};

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
pub fn matrix_linsolve<T>(a: MatrixRef<'_, T>, b: &[T], x: &mut [T], scratch: Option<&mut [T]>) -> Result<()>
where
    T: FloatComplex,
{
    let (n, columns) = a.shape();
    if n != columns || b.len() != n || x.len() != n {
        return Err(Error::Range("matrix_linsolve(), matrix and vectors must form an n by n system".to_string()));
    }
    let cols = n
        .checked_add(1)
        .ok_or_else(|| Error::Config("matrix_linsolve(), augmented matrix size overflow".into()))?;
    let scratch_len = n
        .checked_mul(cols)
        .ok_or_else(|| Error::Config("matrix_linsolve(), augmented matrix size overflow".into()))?;

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
    for ((a_row, &b_value), augmented_row) in a.rows().zip(b).zip(augmented.rows_mut()) {
        augmented_row[..n].copy_from_slice(a_row);
        augmented_row[n] = b_value;
    }

    // Run Gauss-Jordan elimination on M
    matrix_gjelim(&mut augmented)?;

    // Copy result from right-most column of M
    for (output, row_index) in x.iter_mut().zip(0..n) {
        *output = augmented.row(row_index)[n];
    }

    Ok(())
}
