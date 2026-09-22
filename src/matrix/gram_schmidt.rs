const DEBUG_MATRIX_GRAMSCHMIDT: bool = false;

use crate::error::{Error, Result};
use crate::matrix::{FloatComplex, MatrixMut, MatrixRef};

/// Project `v` onto `u` and write the result to `e`.
///
/// All three vectors must have the same length.
///
/// This retained compatibility routine does not reject a zero or nearly zero
/// `u`. That input can produce non-finite output.
///
/// # Errors
///
/// Returns an error if the vector lengths differ. `e` is unchanged on error.
pub fn matrix_proj<T>(u: &[T], v: &[T], e: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let n = u.len();
    if n != v.len() || n != e.len() {
        return Err(Error::Range("matrix_proj: input vectors must have the same length".to_owned()));
    }

    // Compute dot-product between _u and _v
    let mut uv = T::zero();
    let mut uu = T::zero();
    for (&u_value, &v_value) in u.iter().zip(v) {
        uv += u_value * v_value.conj();
        uu += u_value * u_value.conj();
    }

    // TODO: check magnitude of _uu
    let g = uv / uu;
    for (output, &input) in e.iter_mut().zip(u) {
        *output = input * g;
    }

    Ok(())
}

/// Orthonormalize the columns of `x` with classical Gram-Schmidt.
///
/// `v` receives a matrix with the same shape as `x`. The matrix must have at
/// least one row and one column.
///
/// This retained compatibility routine does not detect linearly dependent or
/// nearly dependent columns. Use a dedicated linear-algebra library for a
/// general-purpose orthonormalization.
///
/// # Errors
///
/// Returns an error for zero dimensions or a mismatched output shape. `v` is
/// unchanged on those errors. It can be partly written on a later failure.
pub fn matrix_gramschmidt<T>(x: MatrixRef<'_, T>, v: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (rx, cx) = x.shape();
    if rx == 0 || cx == 0 || v.shape() != (rx, cx) {
        return Err(Error::Range("matrix_gramschmidt: input matrix cannot have zero-length dimensions".to_owned()));
    }

    // Copy _x to _v
    v.as_mut_slice().copy_from_slice(x.as_slice());

    let n = rx; // Dimensionality of each vector
    let mut proj_ij = vec![T::zero(); n];

    for j in 0..cx {
        for i in 0..j {
            // v_j <- v_j - proj(v_i, v_j)

            if DEBUG_MATRIX_GRAMSCHMIDT {
                println!("computing proj(v_{}, v_{})", i, j);
            }

            // Compute proj(v_i, v_j)
            let mut vij = T::zero(); // dotprod(v_i, v_j)
            let mut vii = T::zero(); // dotprod(v_i, v_i)
            for (&ti, &tj) in v.column(i).zip(v.column(j)) {
                let prodij = ti * tj.conj();
                vij += prodij;

                let prodii = ti * ti.conj();
                vii += prodii;
            }
            // TODO: vii should be 1.0 from normalization step below
            let g = vij / vii;

            // Complete projection
            for (projection, &value) in proj_ij.iter_mut().zip(v.column(i)) {
                *projection = value * g;
            }

            // Subtract projection from v_j
            for (row, &projection) in v.rows_mut().zip(&proj_ij) {
                row[j] -= projection;
            }
        }

        // Normalize v_j
        let mut vjj = T::zero(); // dotprod(v_j, v_j)
        for &tj in v.column(j) {
            let prodjj = tj * tj.conj();
            vjj += prodjj;
        }
        // TODO: check magnitude of vjj
        let g = T::one() / vjj.sqrt();
        for row in v.rows_mut() {
            row[j] *= g;
        }

        if DEBUG_MATRIX_GRAMSCHMIDT {
            // TODO: Implement matrix print function
            // MATRIX(_print)(v, rx, cx);
        }
    }

    Ok(())
}
