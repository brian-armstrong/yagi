use num_traits::{Float, Zero};

use crate::error::{Error, Result};
use crate::matrix::{FloatComplex, MatrixMut, MatrixRef};

/// Factor a positive-definite matrix as `A = L * L^H`.
///
/// `l` receives the lower-triangular factor. `a` and `l` must have the same
/// square shape.
///
/// This retained compatibility routine does not verify that `a` is Hermitian.
/// Use a dedicated linear-algebra library for a general-purpose Cholesky
/// factorization.
///
/// # Errors
///
/// Returns an error if the matrices are not equally sized and square, or a
/// diagonal term shows that `a` is not positive definite. `l` can be partly
/// written when factorization fails.
pub fn matrix_chol<T>(a: MatrixRef<'_, T>, l: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (n, columns) = a.shape();
    if n != columns || l.shape() != (n, n) {
        return Err(Error::Range("matrix_chol(), input and output must be equally sized square matrices".to_string()));
    }

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
        if a_jj.im().abs() > T::Real::zero() {
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
