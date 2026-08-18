use crate::matrix::{matrix_gjelim, FloatComplex};
use crate::error::{Error, Result};

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
pub fn matrix_linsolve<T>(
    a: &[T],
    n: usize,
    b: &[T],
    x: &mut [T],
    scratch: Option<&mut [T]>,
) -> Result<()>
where
    T: FloatComplex,
{
    let cols = n.checked_add(1).ok_or_else(|| {
        Error::Config("matrix_linsolve(), augmented matrix size overflow".into())
    })?;
    let scratch_len = n.checked_mul(cols).ok_or_else(|| {
        Error::Config("matrix_linsolve(), augmented matrix size overflow".into())
    })?;

    let mut owned = Vec::new();
    let m = match scratch {
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

    // Compute augmented matrix M [size: n x n+1]
    for r in 0..n {
        let row = r * cols;
        m[row..row + n].copy_from_slice(&a[r * n..(r + 1) * n]);
        m[row + n] = b[r];
    }

    // Run Gauss-Jordan elimination on M
    matrix_gjelim(m, n, cols)?;

    // Copy result from right-most column of M
    for r in 0..n {
        x[r] = m[cols * (r + 1) - 1];
    }

    Ok(())
}
