use num_traits::{Float, Zero};

use crate::error::{Error, Result};
use crate::matrix::view::{MatrixMut, MatrixRef};
use crate::matrix::FloatComplex;

/// Compute Cholesky decomposition of a symmetric/Hermitian positive-
/// definite matrix as A = L * L^T
///  _a      :   input square matrix [size: _n x _n]
///  _n      :   input matrix dimension
///  _l      :   output lower-triangular matrix
pub fn matrix_chol<T>(a: &[T], n: usize, l: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let a = MatrixRef::new(a, n, n)?;
    let mut l = MatrixMut::new(l, n, n)?;

    // Reset L
    l.as_mut_slice().fill(T::zero());

    for j in 0..n {
        // Assert that a_jj is real, positive
        let a_jj = a.row(j)[j];
        if a_jj.re() < T::Real::zero() {
            return Err(Error::Value(format!(
                "matrix_chol(), matrix is not positive definite (real{{A[{},{}]}} = {:e} < 0)",
                j,
                j,
                a_jj.re()
            )));
        }
        if T::is_complex() && Float::abs(a_jj.im()) > T::Real::zero() {
            return Err(Error::Value(format!(
                "matrix_chol(), matrix is not positive definite (|imag{{A[{},{}]}}| = {:e} > 0)",
                j,
                j,
                a_jj.im().abs()
            )));
        }

        // Compute l_jj and store it in output matrix
        let mut t0 = T::zero();
        let l_j_row = l.row(j);
        for k in 0..j {
            let l_jk = l_j_row[k];
            t0 += l_jk * l_jk.conj();
        }
        // Test to ensure a_jj > t0
        if a_jj.re() < t0.re() {
            return Err(Error::Value(format!(
                "matrix_chol(), matrix is not positive definite (real{{A[{},{}]}} = {:e} < {:e})",
                j,
                j,
                a_jj.re(),
                t0.re()
            )));
        }

        let l_jj = (a_jj - t0).sqrt();
        l.row_mut(j)[j] = l_jj;

        for i in j + 1..n {
            let mut t1 = a.row(i)[j];
            let (l_i_row, l_j_row) = l.two_rows_mut(i, j).expect("distinct rows below and on the diagonal");
            for k in 0..j {
                let l_ik = l_i_row[k];
                let l_jk = l_j_row[k];
                t1 -= l_ik * l_jk.conj();
            }
            // TODO: store inverse of l_jj to reduce number of divisions
            l_i_row[j] = t1 / l_jj;
        }
    }

    Ok(())
}
