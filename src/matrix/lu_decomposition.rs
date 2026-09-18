use crate::error::{Error, Result};
use crate::matrix::view::{MatrixMut, MatrixRef};
use crate::matrix::FloatComplex;

/// Factor a square matrix with Crout's L/U decomposition.
///
/// `l`, `u`, and `p` must have the same square shape as `x`. `l` receives the
/// lower-triangular factor, `u` receives a unit-diagonal upper-triangular
/// factor, and `p` receives the identity matrix.
///
/// This retained compatibility routine does not pivot or detect singular
/// matrices. Its permutation output is always the identity. Use a dedicated
/// linear-algebra library for a general-purpose LU factorization.
///
/// # Errors
///
/// Returns an error if any matrix is not square with the shape of `x`. Outputs
/// are unchanged on that error.
pub fn matrix_ludecomp_crout<T>(
    x: MatrixRef<'_, T>,
    l: &mut MatrixMut<'_, T>,
    u: &mut MatrixMut<'_, T>,
    p: &mut MatrixMut<'_, T>,
) -> Result<()>
where
    T: FloatComplex,
{
    let (n, columns) = x.shape();
    if n != columns || l.shape() != (n, n) || u.shape() != (n, n) || p.shape() != (n, n) {
        return Err(Error::Range("matrix_ludecomp_crout(), input matrix not square".to_string()));
    }

    // Reset L, U
    l.as_mut_slice().fill(T::zero());
    u.as_mut_slice().fill(T::zero());
    p.as_mut_slice().fill(T::zero());

    for k in 0..n {
        for i in k..n {
            let mut l_ik = x.row(i)[k];
            let l_row = l.row_mut(i);
            for t in 0..k {
                l_ik -= l_row[t] * u.row(t)[k];
            }
            l_row[k] = l_ik;
        }

        for j in k..n {
            if j == k {
                u.row_mut(k)[j] = T::one();
                continue;
            }

            let mut u_kj = x.row(k)[j];
            for t in 0..k {
                u_kj -= l.row(k)[t] * u.row(t)[j];
            }
            u_kj = u_kj / l.row(k)[k];
            u.row_mut(k)[j] = u_kj;
        }
    }

    // Set output permutation matrix to identity matrix
    for row_index in 0..n {
        p.row_mut(row_index)[row_index] = T::one();
    }
    Ok(())
}

/// Factor a square matrix with Doolittle's L/U decomposition.
///
/// `l`, `u`, and `p` must have the same square shape as `x`. `l` receives a
/// unit-diagonal lower-triangular factor, `u` receives the upper-triangular
/// factor, and `p` receives the identity matrix.
///
/// This retained compatibility routine does not pivot or detect singular
/// matrices. Its permutation output is always the identity. Use a dedicated
/// linear-algebra library for a general-purpose LU factorization.
///
/// # Errors
///
/// Returns an error if any matrix is not square with the shape of `x`. Outputs
/// are unchanged on that error.
pub fn matrix_ludecomp_doolittle<T>(
    x: MatrixRef<'_, T>,
    l: &mut MatrixMut<'_, T>,
    u: &mut MatrixMut<'_, T>,
    p: &mut MatrixMut<'_, T>,
) -> Result<()>
where
    T: FloatComplex,
{
    let (n, columns) = x.shape();
    if n != columns || l.shape() != (n, n) || u.shape() != (n, n) || p.shape() != (n, n) {
        return Err(Error::Range("matrix_ludecomp_doolittle(), input matrix not square".to_string()));
    }

    // Reset L, U
    l.as_mut_slice().fill(T::zero());
    u.as_mut_slice().fill(T::zero());
    p.as_mut_slice().fill(T::zero());

    for k in 0..n {
        // Compute upper triangular matrix
        for j in k..n {
            let mut u_kj = x.row(k)[j];
            for t in 0..k {
                u_kj -= l.row(k)[t] * u.row(t)[j];
            }
            u.row_mut(k)[j] = u_kj;
        }

        // Compute lower triangular matrix
        for i in k..n {
            if i == k {
                l.row_mut(i)[k] = T::one();
                continue;
            }

            let mut l_ik = x.row(i)[k];
            let l_row = l.row_mut(i);
            for t in 0..k {
                l_ik -= l_row[t] * u.row(t)[k];
            }
            l_ik = l_ik / u.row(k)[k];
            l_row[k] = l_ik;
        }
    }

    // Set output permutation matrix to identity matrix
    for row_index in 0..n {
        p.row_mut(row_index)[row_index] = T::one();
    }
    Ok(())
}
