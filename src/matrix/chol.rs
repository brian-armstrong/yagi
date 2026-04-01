use num_traits::{Float, Zero};

use crate::matrix::{matrix_access, matrix_access_mut, FloatComplex};
use crate::error::{Result, Error};

/// Compute Cholesky decomposition of a symmetric/Hermitian positive-
/// definite matrix as A = L * L^T
///  _a      :   input square matrix [size: _n x _n]
///  _n      :   input matrix dimension
///  _l      :   output lower-triangular matrix
pub fn matrix_chol<T>(a: &[T], n: usize, l: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    // Reset L
    l.iter_mut().for_each(|x| *x = T::zero());

    for j in 0..n {
        // Assert that a_jj is real, positive
        let a_jj = matrix_access(a, n, n, j, j);
        if a_jj.re() < T::Real::zero() {
            return Err(Error::Value(format!("matrix_chol(), matrix is not positive definite (real{{A[{},{}]}} = {:e} < 0)", j, j, a_jj.re())));
        }
        if T::is_complex() && Float::abs(a_jj.im()) > T::Real::zero() {
            return Err(Error::Value(format!("matrix_chol(), matrix is not positive definite (|imag{{A[{},{}]}}| = {:e} > 0)", j, j, a_jj.im().abs())));
        }

        // Compute l_jj and store it in output matrix
        let mut t0 = T::zero();
        for k in 0..j {
            let l_jk = matrix_access(l, n, n, j, k);
            t0 = t0 + l_jk * l_jk.conj();
        }
        // Test to ensure a_jj > t0
        if a_jj.re() < t0.re() {
            return Err(Error::Value(format!("matrix_chol(), matrix is not positive definite (real{{A[{},{}]}} = {:e} < {:e})", j, j, a_jj.re(), t0.re())));
        }

        let l_jj = (a_jj - t0).sqrt();
        matrix_access_mut(l, n, n, j, j, l_jj);

        for i in j + 1..n {
            let mut t1 = matrix_access(a, n, n, i, j);
            for k in 0..j {
                let l_ik = matrix_access(l, n, n, i, k);
                let l_jk = matrix_access(l, n, n, j, k);
                t1 = t1 - l_ik * l_jk.conj();
            }
            // TODO: store inverse of l_jj to reduce number of divisions
            matrix_access_mut(l, n, n, i, j, t1 / l_jj);
        }
    }

    Ok(())
}
