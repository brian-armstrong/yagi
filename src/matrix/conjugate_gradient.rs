use num_traits::{Float, Zero};

use crate::error::{Error, Result};
use crate::matrix::{matrix_mul_vec, FloatComplex, MatrixRef};

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

/// Solves `a * x = b` by conjugate gradient iteration.
///
/// `a` must be square, Hermitian and positive definite, and `b` and `x` must
/// each have one element per row of `a`. The matrix properties are not
/// checked. `_opts` is reserved and ignored.
///
/// The iteration starts from zero and stops once the residual norm is at most
/// `1e-6` times the norm of `b`, or after `4 * n` iterations for an `n` by `n`
/// system. The residual is recomputed from `a` every 50 iterations. `x`
/// receives the iterate with the smallest residual. Hitting the iteration
/// limit is not reported, so check the residual when accuracy matters. A zero
/// `b` gives a zero `x`.
///
/// The stopping test compares squared residual norms. When `b^H * b`
/// underflows to zero, the test is met before the first step and `x` stays
/// zero. For `f32` `a = [1e-30]` and `b = [1e-30]`, the result is `[0]`
/// rather than `[1]`.
///
/// # Errors
///
/// Returns an error when `a` is empty or the shapes do not match. It also
/// returns an error when a residual or step becomes infinite or NaN, or when
/// `d^H * a * d` is zero or negative for a search direction `d`, which
/// happens when `a` is not positive definite. `x` is unchanged on error.
pub fn matrix_cgsolve<T>(a: MatrixRef<'_, T>, b: &[T], x: &mut [T], _opts: Option<()>) -> Result<()>
where
    T: FloatComplex,
{
    let (n, columns) = a.shape();
    if n == 0 || n != columns || b.len() != n || x.len() != n {
        return Err(Error::Range("matrix_cgsolve(), system dimension cannot be zero".to_owned()));
    }

    let max_iterations = n
        .checked_mul(4)
        .ok_or_else(|| Error::Range("matrix_cgsolve(), iteration count overflows usize".to_owned()))?;
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
    let mut best_x = vec![T::zero(); n];

    // d0 = b - A*x0 (assume x0 = {0, 0, 0, ...0})
    d0.copy_from_slice(b);

    // r0 = d0
    r0.copy_from_slice(&d0);

    // delta_init = b^T * b
    let delta_init = vector_dot_hermitian(b, b);
    if !delta_init.is_finite() {
        return Err(Error::NoConvergence("matrix_cgsolve(), initial residual is not finite".to_string()));
    }

    // delta0 = r0^T * r0
    let mut delta0 = vector_dot_hermitian(&r0, &r0);
    if !delta0.is_finite() {
        return Err(Error::NoConvergence("matrix_cgsolve(), initial residual is not finite".to_string()));
    }

    // save best solution
    let mut i = 0; // iteration counter
    let mut res_opt = T::Real::zero();

    while i < max_iterations && delta0.re() > tol * tol * delta_init.re() {
        if DEBUG_CGSOLVE {
            println!("*********** {} / {} (max) **************", i, max_iterations);
            println!("  comparing {:e} > {:e}", delta0.re(), tol * tol * delta_init.re());
        }

        // q = A*d0
        matrix_mul_vec(a, &d0, &mut q)?;

        // gamma = d0^T * q
        let gamma = vector_dot_hermitian(&d0, &q);
        if !gamma.is_finite() || gamma.re() <= T::Real::zero() {
            return Err(Error::NoConvergence(
                "matrix_cgsolve(), search direction has non-positive or non-finite curvature".to_string(),
            ));
        }

        // step size: alpha = (r0^T * r0) / (d0^T * A * d0)
        //                  = delta0 / gamma
        let alpha = delta0 / gamma;
        if !alpha.is_finite() {
            return Err(Error::NoConvergence("matrix_cgsolve(), step size is not finite".to_string()));
        }

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
            matrix_mul_vec(a, &x1, &mut ax1)?;
            for ((r1_value, &b_value), &ax1_value) in r1.iter_mut().zip(b).zip(&ax1) {
                *r1_value = b_value - ax1_value;
            }
        } else {
            for ((r1_value, &r0_value), &q_value) in r1.iter_mut().zip(&r0).zip(&q) {
                *r1_value = r0_value - alpha * q_value;
            }
        }

        // delta1 = r1^T * r1
        let delta1 = vector_dot_hermitian(&r1, &r1);
        if !delta1.is_finite() {
            return Err(Error::NoConvergence("matrix_cgsolve(), residual is not finite".to_string()));
        }

        // update beta
        let beta = delta1 / delta0;
        if !beta.is_finite() {
            return Err(Error::NoConvergence("matrix_cgsolve(), search-direction update is not finite".to_string()));
        }

        // d1 = r + beta*d0
        for ((d1_value, &r1_value), &d0_value) in d1.iter_mut().zip(&r1).zip(&d0) {
            *d1_value = r1_value + beta * d0_value;
        }

        // compute residual
        let res = Float::sqrt(delta1.abs() / delta_init.abs());
        if !res.is_finite() {
            return Err(Error::NoConvergence("matrix_cgsolve(), relative residual is not finite".to_string()));
        }
        if i == 0 || res < res_opt {
            // save best solution
            res_opt = res;
            best_x.copy_from_slice(&x1);
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

    x.copy_from_slice(&best_x);
    Ok(())
}
