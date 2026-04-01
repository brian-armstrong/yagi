use std::slice;
use num_traits::{Float, Zero};

use crate::error::{Result, Error};
use crate::matrix::{matrix_mul, matrix_transpose_mul, FloatComplex};

const DEBUG_CGSOLVE: bool = false;

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

    // options
    let max_iterations = 4 * n; // maximum number of iterations
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
    for j in 0..n {
        d0[j] = b[j];
    }

    // r0 = d0
    r0.copy_from_slice(&d0);

    let mut delta_init = T::zero();
    let mut delta0 = T::zero();

    // delta_init = b^T * b
    matrix_transpose_mul(b, n, 1, slice::from_mut(&mut delta_init));

    // delta0 = r0^T * r0
    matrix_transpose_mul(&r0, n, 1, slice::from_mut(&mut delta0));

    // save best solution
    x.copy_from_slice(&x0);
    let mut i = 0; // iteration counter
    let mut res_opt = T::Real::zero();

    while i < max_iterations && delta0.re() > tol * tol * delta_init.re() {
        if DEBUG_CGSOLVE {
            println!("*********** {} / {} (max) **************", i, max_iterations);
            println!("  comparing {:e} > {:e}", delta0.re(), tol * tol * delta_init.re());
        }

        // q = A*d0
        matrix_mul(a, n, n, &d0, n, 1, &mut q, n, 1)?;

        // gamma = d0^T * q
        let gamma: T = d0.iter().zip(q.iter()).map(|(&d, &q)| d.conj() * q).sum::<T>();

        // step size: alpha = (r0^T * r0) / (d0^T * A * d0)
        //                  = delta0 / gamma
        let alpha = delta0 / gamma;

        if DEBUG_CGSOLVE {
            println!("  alpha  = {:e}", alpha.re());
            println!("  delta0 = {:e}", delta0.re());
        }

        // update x
        for j in 0..n {
            x1[j] = x0[j] + alpha * d0[j];
        }

        if DEBUG_CGSOLVE {
            println!("  x:");
            // TODO: Implement matrix print function
        }

        // update r
        if (i + 1) % 50 == 0 {
            // periodically re-compute: r = b - A*x1
            matrix_mul(a, n, n, &x1, n, 1, &mut ax1, n, 1)?;
            for j in 0..n {
                r1[j] = b[j] - ax1[j];
            }
        } else {
            for j in 0..n {
                r1[j] = r0[j] - alpha * q[j];
            }
        }

        // delta1 = r1^T * r1
        let mut delta1 = T::zero();
        matrix_transpose_mul(&r1, n, 1, slice::from_mut(&mut delta1));

        // update beta
        let beta = delta1 / delta0;

        // d1 = r + beta*d0
        for j in 0..n {
            d1[j] = r1[j] + beta * d0[j];
        }

        // compute residual
        let res = Float::sqrt(delta1.abs() / delta_init.abs());
        if i == 0 || res < res_opt {
            // save best solution
            res_opt = res;
            x.copy_from_slice(&x1);
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
