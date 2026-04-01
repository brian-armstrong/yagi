use num_traits::NumCast;
const DEBUG_MATRIX_GRAMSCHMIDT: bool = false;

use crate::matrix::FloatComplex;
use crate::error::{Result, Error};

/// Compute projection of _u onto _v, store in _e
pub fn matrix_proj<T>(u: &[T], v: &[T], e: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let n = u.len();
    if n != v.len() || n != e.len() {
        return Err(Error::Range("matrix_proj: input vectors must have the same length".to_owned()));
    }

    // Compute dot-product between _u and _v
    let mut uv = <T as NumCast>::from(0.0).unwrap();
    let mut uu = <T as NumCast>::from(0.0).unwrap();
    for i in 0..n {
        uv = uv + u[i] * v[i].conj();
        uu = uu + u[i] * u[i].conj();
    }

    // TODO: check magnitude of _uu
    let g = uv / uu;
    for i in 0..n {
        e[i] = u[i] * g;
    }

    Ok(())
}

/// Orthonormalization using the Gram-Schmidt algorithm
pub fn matrix_gramschmidt<T>(x: &[T], rx: usize, cx: usize, v: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    // Validate input
    if rx == 0 || cx == 0 {
        return Err(Error::Range("matrix_gramschmidt: input matrix cannot have zero-length dimensions".to_owned()));
    }

    // Copy _x to _v
    v.copy_from_slice(x);

    let n = rx; // Dimensionality of each vector
    let mut proj_ij = vec![<T as NumCast>::from(0.0).unwrap(); n];

    for j in 0..cx {
        for i in 0..j {
            // v_j <- v_j - proj(v_i, v_j)

            if DEBUG_MATRIX_GRAMSCHMIDT {
                println!("computing proj(v_{}, v_{})", i, j);
            }

            // Compute proj(v_i, v_j)
            let mut vij = <T as NumCast>::from(0.0).unwrap(); // dotprod(v_i, v_j)
            let mut vii = <T as NumCast>::from(0.0).unwrap(); // dotprod(v_i, v_i)
            for k in 0..n {
                let ti = v[k * cx + i];
                let tj = v[k * cx + j];

                let prodij = ti * tj.conj();
                vij = vij + prodij;

                let prodii = ti * ti.conj();
                vii = vii + prodii;
            }
            // TODO: vii should be 1.0 from normalization step below
            let g = vij / vii;

            // Complete projection
            for k in 0..n {
                proj_ij[k] = v[k * cx + i] * g;
            }

            // Subtract projection from v_j
            for k in 0..n {
                v[k * cx + j] = v[k * cx + j] - proj_ij[k];
            }
        }

        // Normalize v_j
        let mut vjj = <T as NumCast>::from(0.0).unwrap(); // dotprod(v_j, v_j)
        for k in 0..n {
            let tj = v[k * cx + j];
            let prodjj = tj * tj.conj();
            vjj = vjj + prodjj;
        }
        // TODO: check magnitude of vjj
        let g = <T as NumCast>::from(1.0).unwrap() / vjj.sqrt();
        for k in 0..n {
            v[k * cx + j] = v[k * cx + j] * g;
        }

        if DEBUG_MATRIX_GRAMSCHMIDT {
            // TODO: Implement matrix print function
            // MATRIX(_print)(v, rx, cx);
        }
    }

    Ok(())
}
