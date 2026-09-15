use num_complex::Complex;
use num_traits::{Float, Zero};

use crate::error::{Error, Result};
use crate::matrix::view::{MatrixMut, MatrixRef};
use crate::matrix::FloatComplex;

/// Q/R decomposition using the Gram-Schmidt algorithm
pub fn matrix_qrdecomp_gramschmidt<T>(x: &[T], m: usize, n: usize, q: &mut [T], r: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    // validate input
    if m != n {
        return Err(Error::Range("matrix_qrdecomp_gramschmidt(), input matrix not square".to_string()));
    }

    let x = MatrixRef::new(x, n, n)?;
    let mut q = MatrixMut::new(q, n, n)?;
    let mut r = MatrixMut::new(r, n, n)?;

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

#[macro_export]
macro_rules! matrix_qrdecomp_gramschmidt {
    ($name:tt, $T:ty) => {
        pub fn $name(x: &[$T], m: usize, n: usize, q: &mut [$T], r: &mut [$T]) -> $crate::error::Result<()> {
            $crate::matrix::matrix_qrdecomp_gramschmidt(x, m, n, q, r)
        }
    };
}

matrix_qrdecomp_gramschmidt!(matrix_qrdecomp_gramschmidt_f32, f32);
matrix_qrdecomp_gramschmidt!(matrix_qrdecomp_gramschmidt_c32, Complex<f32>);
