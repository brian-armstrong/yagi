use num_traits::{Float, Zero};

use crate::error::{Error, Result};
use crate::matrix::view::{MatrixMut, MatrixRef};
use crate::matrix::{matrix_mul_vec, FloatComplex};

const DEBUG_CGSOLVE: bool = false;

/// Compute the Hermitian inner product, preserving left-to-right accumulation.
fn vector_dot_hermitian<T>(left: &[T], right: &[T]) -> T
where
    T: FloatComplex,
{
    debug_assert_eq!(left.len(), right.len());

    let mut sum = T::default();
    for (&left_value, &right_value) in left.iter().zip(right) {
        sum += left_value.conj() * right_value;
    }
    sum
}

/// Solve linear system of equations using conjugate gradient method
///  _A      :   symmetric positive definite matrix [size: _n x _n]
///  _n      :   system dimension
///  _b      :   equality [size: _n x 1]
///  _x      :   solution estimate [size: _n x 1]
///  _opts   :   options (ignored for now)
pub fn matrix_cgsolve<T>(a: &[T], n: usize, b: &[T], x: &mut [T], _opts: Option<()>) -> Result<()>
where
    T: FloatComplex,
{
    // validate input
    if n == 0 {
        return Err(Error::Range("matrix_cgsolve(), system dimension cannot be zero".to_owned()));
    }

    let max_iterations = n
        .checked_mul(4)
        .ok_or_else(|| Error::Range("matrix_cgsolve(), iteration count overflows usize".to_owned()))?;
    let a = MatrixRef::new(a, n, n)?;
    let b = MatrixRef::new(b, n, 1)?;
    let mut x = MatrixMut::new(x, n, 1)?;

    // options
    let tol = T::Real::from(1e-6); // error tolerance

    // TODO : check options
    //  1. set initial _x0
    //  2. max number of iterations
    //  3. residual tolerance

    // allocate memory for arrays
    // initialize x0 to {0, 0, ... 0}
    let mut x0 = vec![T::zero(); n];
    let mut x1 = vec![T::zero(); n];
    let mut d0 = vec![T::zero(); n];
    let mut d1 = vec![T::zero(); n];
    let mut r0 = vec![T::zero(); n];
    let mut r1 = vec![T::zero(); n];
    let mut q = vec![T::zero(); n];
    let mut ax1 = vec![T::zero(); n];

    // d0 = b - A*x0 (assume x0 = {0, 0, 0, ...0})
    d0.copy_from_slice(b.as_slice());

    // r0 = d0
    r0.copy_from_slice(&d0);

    // delta_init = b^T * b
    let delta_init = vector_dot_hermitian(b.as_slice(), b.as_slice());

    // delta0 = r0^T * r0
    let mut delta0 = vector_dot_hermitian(&r0, &r0);

    // save best solution
    x.as_mut_slice().copy_from_slice(&x0);
    let mut i = 0; // iteration counter
    let mut res_opt = T::Real::zero();

    while i < max_iterations && delta0.re() > tol * tol * delta_init.re() {
        if DEBUG_CGSOLVE {
            println!("*********** {} / {} (max) **************", i, max_iterations);
            println!("  comparing {:e} > {:e}", delta0.re(), tol * tol * delta_init.re());
        }

        // q = A*d0
        matrix_mul_vec(a.as_slice(), n, n, &d0, &mut q)?;

        // gamma = d0^T * q
        let gamma = vector_dot_hermitian(&d0, &q);

        // step size: alpha = (r0^T * r0) / (d0^T * A * d0)
        //                  = delta0 / gamma
        let alpha = delta0 / gamma;

        if DEBUG_CGSOLVE {
            println!("  alpha  = {:e}", alpha.re());
            println!("  delta0 = {:e}", delta0.re());
        }

        // update x
        for ((x1_value, &x0_value), &d0_value) in x1.iter_mut().zip(&x0).zip(&d0) {
            *x1_value = x0_value + alpha * d0_value;
        }

        if DEBUG_CGSOLVE {
            println!("  x:");
            // TODO: Implement matrix print function
        }

        // update r
        if (i + 1) % 50 == 0 {
            // periodically re-compute: r = b - A*x1
            matrix_mul_vec(a.as_slice(), n, n, &x1, &mut ax1)?;
            for ((r1_value, &b_value), &ax1_value) in r1.iter_mut().zip(b.as_slice()).zip(&ax1) {
                *r1_value = b_value - ax1_value;
            }
        } else {
            for ((r1_value, &r0_value), &q_value) in r1.iter_mut().zip(&r0).zip(&q) {
                *r1_value = r0_value - alpha * q_value;
            }
        }

        // delta1 = r1^T * r1
        let delta1 = vector_dot_hermitian(&r1, &r1);

        // update beta
        let beta = delta1 / delta0;

        // d1 = r + beta*d0
        for ((d1_value, &r1_value), &d0_value) in d1.iter_mut().zip(&r1).zip(&d0) {
            *d1_value = r1_value + beta * d0_value;
        }

        // compute residual
        let res = Float::sqrt(delta1.abs() / delta_init.abs());
        if i == 0 || res < res_opt {
            // save best solution
            res_opt = res;
            x.as_mut_slice().copy_from_slice(&x1);
        }

        if DEBUG_CGSOLVE {
            println!("  res    = {:e}", res);
        }

        // copy old x, d, r, delta
        x0.copy_from_slice(&x1);
        d0.copy_from_slice(&d1);
        r0.copy_from_slice(&r1);
        delta0 = delta1;

        // increment counter
        i += 1;
    }

    Ok(())
}
