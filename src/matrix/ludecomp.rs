use crate::error::{Error, Result};
use crate::matrix::{matrix_access, matrix_access_mut, matrix_eye, FloatComplex};

/// L/U/P decomposition, Crout's method
pub fn matrix_ludecomp_crout<T>(x: &[T], rx: usize, cx: usize, l: &mut [T], u: &mut [T], p: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    // Validate input
    if rx != cx {
        return Err(Error::Range("matrix_ludecomp_crout(), input matrix not square".to_string()));
    }

    let n = rx;

    // Reset L, U
    for i in 0..(n * n) {
        l[i] = T::zero();
        u[i] = T::zero();
        p[i] = T::zero();
    }

    for k in 0..n {
        for i in k..n {
            let mut l_ik = matrix_access(x, n, n, i, k);
            for t in 0..k {
                l_ik -= matrix_access(l, n, n, i, t) * matrix_access(u, n, n, t, k);
            }
            matrix_access_mut(l, n, n, i, k, l_ik);
        }

        for j in k..n {
            if j == k {
                matrix_access_mut(u, n, n, k, j, T::one());
                continue;
            }

            let mut u_kj = matrix_access(x, n, n, k, j);
            for t in 0..k {
                u_kj -= matrix_access(l, n, n, k, t) * matrix_access(u, n, n, t, j);
            }
            u_kj = u_kj / matrix_access(l, n, n, k, k);
            matrix_access_mut(u, n, n, k, j, u_kj);
        }
    }

    // Set output permutation matrix to identity matrix
    matrix_eye(p, n);
    Ok(())
}

/// L/U/P decomposition, Doolittle's method
pub fn matrix_ludecomp_doolittle<T>(x: &[T], rx: usize, cx: usize, l: &mut [T], u: &mut [T], p: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    // Validate input
    if rx != cx {
        return Err(Error::Range("matrix_ludecomp_doolittle(), input matrix not square".to_string()));
    }

    let n = rx;

    // Reset L, U
    for i in 0..(n * n) {
        l[i] = T::zero();
        u[i] = T::zero();
        p[i] = T::zero();
    }

    for k in 0..n {
        // Compute upper triangular matrix
        for j in k..n {
            let mut u_kj = matrix_access(x, n, n, k, j);
            for t in 0..k {
                u_kj -= matrix_access(l, n, n, k, t) * matrix_access(u, n, n, t, j);
            }
            matrix_access_mut(u, n, n, k, j, u_kj);
        }

        // Compute lower triangular matrix
        for i in k..n {
            if i == k {
                matrix_access_mut(l, n, n, i, k, T::one());
                continue;
            }

            let mut l_ik = matrix_access(x, n, n, i, k);
            for t in 0..k {
                l_ik -= matrix_access(l, n, n, i, t) * matrix_access(u, n, n, t, k);
            }
            l_ik = l_ik / matrix_access(u, n, n, k, k);
            matrix_access_mut(l, n, n, i, k, l_ik);
        }
    }

    // Set output permutation matrix to identity matrix
    matrix_eye(p, n);
    Ok(())
}
