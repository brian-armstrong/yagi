use num_traits::{Float, Zero};
use num_complex::{Complex, ComplexFloat};

use crate::matrix::{matrix_access, matrix_access_mut, FloatComplex};
use crate::error::{Error, Result};

/// Q/R decomposition using the Gram-Schmidt algorithm
pub fn matrix_qrdecomp_gramschmidt<T>(x: &[T], m: usize, n: usize, q: &mut [T], r: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    // validate input
    if m != n {
        return Err(Error::Range("matrix_qrdecomp_gramschmidt(), input matrix not square".to_string()));
    }

    // generate and initialize matrices
    let mut e = vec![T::zero(); n * n];

    for k in 0..n {
        // e(i,k) <- _x(i,k)
        for i in 0..n {
            matrix_access_mut(&mut e, n, n, i, k, matrix_access(x, n, n, i, k));
        }

        // subtract...
        for i in 0..k {
            // compute dot product _x(:,k) * e(:,i)
            let mut g = T::zero();
            for j in 0..n {
                g += matrix_access(x, n, n, j, k) * matrix_access(&e, n, n, j, i).conj();
            }
            for j in 0..n {
                let v = matrix_access(&e, n, n, j, k) - matrix_access(&e, n, n, j, i) * g;
                matrix_access_mut(&mut e, n, n, j, k, v);
            }
        }

        // compute e_k = e_k / |e_k|
        let mut ek: T::Real = T::Real::zero();
        for i in 0..n {
            let ak = matrix_access(&e, n, n, i, k);
            let ak2: T::Real = ak.abs();
            ek = ek + ak2 * ak2;
        }
        ek = Float::sqrt(ek);

        // normalize e
        for i in 0..n {
            let v = matrix_access(&e, n, n, i, k) / T::from(ek).unwrap();
            matrix_access_mut(&mut e, n, n, i, k, v);
        }
    }

    // move Q
    q.copy_from_slice(&e);

    // compute R
    // j : row
    // k : column
    for j in 0..n {
        for k in 0..n {
            if k < j {
                matrix_access_mut(r, n, n, j, k, T::zero());
            } else {
                // compute dot product between and Q(:,j) and _x(:,k)
                let mut g = T::zero();
                for i in 0..n {
                    g += matrix_access(q, n, n, i, j).conj() * matrix_access(x, n, n, i, k);
                }
                matrix_access_mut(r, n, n, j, k, g);
            }
        }
    }

    Ok(())
}


#[macro_export]
macro_rules! matrix_qrdecomp_gramschmidt {
    ($name:tt, $T:ty) => {
        pub fn $name(x: &[$T], m: usize, n: usize, q: &mut [$T], r: &mut [$T]) ->  Result<()> {
            // validate input
            if m != n {
                return Err(Error::Range("matrix_qrdecomp_gramschmidt(), input matrix not square".to_string()));
            }

            // generate and initialize matrices
            let mut e = vec![<$T>::zero(); n * n];

            for k in 0..n {
                // e(i,k) <- _x(i,k)
                for i in 0..n {
                    matrix_access_mut(&mut e, n, n, i, k, matrix_access(x, n, n, i, k));
                }

                // subtract...
                for i in 0..k {
                    // compute dot product _x(:,k) * e(:,i)
                    let mut g = <$T>::zero();
                    for j in 0..n {
                        g += matrix_access(x, n, n, j, k) * matrix_access(&e, n, n, j, i).conj();
                    }
                    for j in 0..n {
                        let v = matrix_access(&e, n, n, j, k) - matrix_access(&e, n, n, j, i) * g;
                        matrix_access_mut(&mut e, n, n, j, k, v);
                    }
                }

                // compute e_k = e_k / |e_k|
                let mut ek = <$T>::zero();
                for i in 0..n {
                    let ak = matrix_access(&e, n, n, i, k);
                    let ak2 = ak.abs();
                    ek = ek + ak2 * ak2;
                }
                ek = ek.sqrt();

                // normalize e
                for i in 0..n {
                    let v = matrix_access(&e, n, n, i, k) / ek;
                    matrix_access_mut(&mut e, n, n, i, k, v);
                }
            }

            // move Q
            q.copy_from_slice(&e);

            // compute R
            // j : row
            // k : column
            for j in 0..n {
                for k in 0..n {
                    if k < j {
                        matrix_access_mut(r, n, n, j, k, <$T>::zero());
                    } else {
                        // compute dot product between and Q(:,j) and _x(:,k)
                        let mut g = <$T>::zero();
                        for i in 0..n {
                            g += matrix_access(q, n, n, i, j).conj() * matrix_access(x, n, n, i, k);
                        }
                        matrix_access_mut(r, n, n, j, k, g);
                    }
                }
            }

            Ok(())
        }
    };
}

matrix_qrdecomp_gramschmidt!(matrix_qrdecomp_gramschmidt_f32, f32);
matrix_qrdecomp_gramschmidt!(matrix_qrdecomp_gramschmidt_c32, Complex<f32>);