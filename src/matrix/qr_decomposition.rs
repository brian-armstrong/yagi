use num_traits::{Float, Zero};

use crate::error::{Error, Result};
use crate::matrix::view::{MatrixMut, MatrixRef};
use crate::matrix::FloatComplex;

/// Factor a square matrix as `X = Q * R` with classical Gram-Schmidt.
///
/// `q` and `r` must have the same square shape as `x`. `q` receives the
/// orthonormal columns and `r` receives the upper-triangular factor.
///
/// This retained compatibility routine does not detect rank-deficient input
/// and can lose accuracy for nearly dependent columns. Use a dedicated
/// linear-algebra library for a general-purpose QR factorization.
///
/// # Errors
///
/// Returns an error if `x` is not square or either output shape differs from
/// `x`. Outputs are unchanged on those errors.
pub fn matrix_qrdecomp_gramschmidt<T>(
    x: MatrixRef<'_, T>,
    q: &mut MatrixMut<'_, T>,
    r: &mut MatrixMut<'_, T>,
) -> Result<()>
where
    T: FloatComplex,
{
    let (m, n) = x.shape();
    if m != n {
        return Err(Error::Range("matrix_qrdecomp_gramschmidt(), input matrix not square".to_string()));
    }
    if q.shape() != (n, n) || r.shape() != (n, n) {
        return Err(Error::Range("matrix_qrdecomp_gramschmidt(), output shapes must match the input".to_string()));
    }

    // generate and initialize matrices
    let mut e_storage = vec![T::zero(); x.as_slice().len()];
    let mut e = MatrixMut::new(&mut e_storage, n, n)?;

    for k in 0..n {
        // e(i,k) <- _x(i,k)
        for (e_row, &x_value) in e.rows_mut().zip(x.column(k)) {
            e_row[k] = x_value;
        }

        // subtract...
        for i in 0..k {
            // compute dot product _x(:,k) * e(:,i)
            let mut g = T::zero();
            for (&x_value, &e_value) in x.column(k).zip(e.column(i)) {
                g += x_value * e_value.conj();
            }
            for e_row in e.rows_mut() {
                e_row[k] -= e_row[i] * g;
            }
        }

        // compute e_k = e_k / |e_k|
        let mut ek: T::Real = T::Real::zero();
        for &ak in e.column(k) {
            let ak2: T::Real = ak.abs();
            ek = ek + ak2 * ak2;
        }
        ek = Float::sqrt(ek);

        // normalize e
        for e_row in e.rows_mut() {
            e_row[k] = e_row[k] / T::from(ek).unwrap();
        }
    }

    // move Q
    q.as_mut_slice().copy_from_slice(e.as_slice());

    // compute R
    // j : row
    // k : column
    for (j, r_row) in r.rows_mut().enumerate() {
        for k in 0..n {
            if k < j {
                r_row[k] = T::zero();
            } else {
                // compute dot product between and Q(:,j) and _x(:,k)
                let mut g = T::zero();
                for (&q_value, &x_value) in q.column(j).zip(x.column(k)) {
                    g += q_value.conj() * x_value;
                }
                r_row[k] = g;
            }
        }
    }

    Ok(())
}
