use crate::error::{Error, Result};
use crate::matrix::{matrix_gjelim, FloatComplex, MatrixMut, MatrixRef};

/// Solves the square linear system `a * x = b`.
///
/// `a` must be square, and `b` and `x` must each have one element per row of
/// `a`. The system is solved by [`matrix_gjelim`] on `a` with `b` appended as
/// an extra column.
///
/// That augmented matrix needs `n * (n + 1)` elements for an `n` by `n` system.
/// Pass `scratch` to supply them and avoid an allocation. Extra elements are
/// ignored, and the contents of `scratch` are unspecified after the call.
/// With `None`, the storage is allocated for the call.
///
/// Only an exactly zero pivot is detected. A nearly singular system gives an
/// inaccurate result with no error. A very small pivot can also give an
/// infinite `x`, as described on [`matrix_gjelim`].
///
/// # Errors
///
/// Returns an error when the shapes do not match, `scratch` is too short, or
/// elimination meets a zero pivot. `x` is unchanged on error.
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
