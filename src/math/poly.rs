use num_complex::{Complex, ComplexFloat};
use std::cmp::Ordering;

use crate::error::{Error, Result};
use crate::matrix::{FloatComplex, matrix_access_mut, matrix_mul, matrix_trans, matrix_inv};

/// Evaluate polynomial
///
/// Computes the value `y` of polynomial `p` at `x` using Horner's method.
///
/// # Arguments
///
/// * `p` - Polynomial coefficients (length: _k)
/// * `k` - Polynomial length
/// * `x` - Input value
///
/// # Returns
///
/// Polynomial evaluated at `x`
pub fn poly_val<T>(p: &[T], k: usize, x: T) -> T
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + From<f32>,
{
    let mut xk = T::from(1.0);
    let mut y = T::from(0.0);
    for i in 0..k {
        y = y + p[i] * xk;
        xk = xk * x;
    }
    y
}

/// Fit polynomial to set of data points
///
/// # Arguments
///
/// * `x` - Independent variable (length: _n)
/// * `y` - Dependent variable (length: _n)
/// * `n` - Number of samples
/// * `p` - Output polynomial coefficients (length: _k)
/// * `k` - Polynomial length
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn poly_fit<T>(x: &[T], y: &[T], n: usize, p: &mut [T], k: usize) -> Result<()>
where
    T: FloatComplex
{
    // TODO: Check input dimensions

    let mut x_matrix = vec![T::default(); n * k];
    for r in 0..n {
        let mut v = T::one();
        for c in 0..k {
            matrix_access_mut(&mut x_matrix, n, k, r, c, v);
            v = v * x[r];
        }
    }

    // Compute transpose of X
    let mut xt_matrix = x_matrix.clone();
    matrix_trans(&mut xt_matrix, n, k);

    // Compute [X']*y
    let mut xty = vec![T::default(); k];
    matrix_mul(&xt_matrix, k, n, y, n, 1, &mut xty, k, 1)?;

    // Compute [X']*X
    let mut x2 = vec![T::default(); k * k];
    matrix_mul(&xt_matrix, k, n, &x_matrix, n, k, &mut x2, k, k)?;

    // Compute inv([X']*X)
    let mut g = x2.clone();
    matrix_inv(&mut g, k, k)?;

    // Compute coefficients
    matrix_mul(&g, k, k, &xty, k, 1, p, k, 1)?;

    Ok(())
}


/// Expands the polynomial:
///  P_n(x) = (1+x)^n
/// as
///  P_n(x) = p`[0]` + p`[1]`*x + p`[2]`*x^2 + ... + p`[n]`x^n
///
/// # Arguments
///
/// * `n` - Polynomial order
/// * `c` - Output polynomial coefficients (length: n+1)
pub fn poly_expandbinomial<T>(n: usize, c: &mut [T]) -> ()
where 
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + From<f32>,
{
    if n == 0 {
        c[0] = T::from(0.0);
        return;
    }

    // Initialize coefficients array to [1,0,0,....0]
    c[0] = T::from(1.0);
    for i in 1..=n {
        c[i] = T::from(0.0);
    }

    // Iterative polynomial multiplication
    for i in 0..n {
        for j in (1..=i+1).rev() {
            c[j] = c[j] + c[j-1];
        }
    }
}

/// Expands the polynomial:
///  P_n(x) = (1+x)^m * (1-x)^k
/// as
///  P_n(x) = p`[0]` + p`[1]`*x + p`[2]`*x^2 + ... + p`[n]`x^n
///
/// # Arguments
///
/// * `m` - Polynomial order (positive term)
/// * `k` - Polynomial order (negative term)
/// * `c` - Output polynomial coefficients (length: m+k+1)
pub fn poly_expandbinomial_pm<T>(m: usize, k: usize, c: &mut [T]) -> ()
where 
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + From<f32>,
{
    let n = m + k;

    if n == 0 {
        c[0] = T::from(0.0);
        return;
    }

    // Initialize coefficients array to [1,0,0,....0]
    c[0] = T::from(1.0);
    for i in 1..=n {
        c[i] = T::from(0.0);
    }

    // Iterative polynomial multiplication (1+x)
    for i in 0..m {
        for j in (1..=i+1).rev() {
            c[j] = c[j] + c[j-1];
        }
    }

    // Iterative polynomial multiplication (1-x)
    for i in m..n {
        for j in (1..=i+1).rev() {
            c[j] = c[j] - c[j-1];
        }
    }
}

/// Perform root expansion on the polynomial:
///  P_n(x) = (x-r`[0]`) * (x-r`[1]`) * ... * (x-r[n-1])
/// as
///  P_n(x) = p`[0]` + p`[1]`*x + ... + p`[n]`*x^n
/// where r`[0]`,r`[1]`,...,r[n-1] are the roots of P_n(x)
/// 
/// # Arguments
///
/// * `r` - Roots of polynomial (length: n)
/// * `n` - Number of roots
/// * `p` - Output polynomial coefficients (length: n+1)
pub fn poly_expandroots<T>(r: &[T], n: usize, p: &mut [T]) -> ()
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Neg<Output = T> + From<f32>,
{
    if n == 0 {
        p[0] = T::from(0.0);
        return;
    }

    // Initialize coefficients array to [1,0,0,....0]
    p[0] = T::from(1.0);
    for i in 1..=n {
        p[i] = T::from(0.0);
    }

    // Iterative polynomial multiplication
    for i in 0..n {
        for j in (1..=i+1).rev() {
            p[j] = -r[i] * p[j] + p[j-1];
        }
        p[0] = -r[i] * p[0];
    }
}

/// Perform root expansion on the polynomial:
///  P_n(x) = (x*b`[0]`-a`[0]`) * (x*b`[1]`-a`[1]`) * ... * (x*b[n-1]-a[n-1])
/// as
///  P_n(x) = p`[0]` + p`[1]`*x + ... + p`[n]`*x^n
///
/// # Arguments
///
/// * `a` - Subtractant of polynomial roots (length: n)
/// * `b` - Multiplicant of polynomial roots (length: n)
/// * `n` - Number of roots
/// * `p` - Output polynomial coefficients (length: n+1)
pub fn poly_expandroots2<T>(a: &[T], b: &[T], n: usize, p: &mut [T]) -> ()
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Neg<Output = T> + std::ops::Div<Output = T> + From<f32>,
{
    // Factor b[i] from each root : (x*b - a) = (x - a/b)*b
    let mut g = T::from(1.0);
    let mut r = vec![T::from(0.0); n];
    for i in 0..n {
        g = g * -b[i];
        r[i] = a[i] / b[i];
    }

    // Expand new root set
    poly_expandroots(&r, n, p);

    // Multiply by gain
    for i in 0..=n {
        p[i] = g * p[i];
    }
}

/// Expands the multiplication of two polynomials
///
/// (a`[0]` + a`[1]`*x + a`[2]`*x^2 + ...) * (b`[0]` + b`[1]`*x + b`[2]`*x^2 + ...)
/// as
/// c`[0]` + c`[1]`*x + c`[2]`*x^2 + ... + c`[n]`*x^n
///
/// where order(c)  = order(a)  + order(b) + 1
///    :: length(c) = length(a) + length(b) - 1
///
/// # Arguments
///
/// * `a` - First polynomial coefficients (length: order_a+1)
/// * `order_a` - First polynomial order
/// * `b` - Second polynomial coefficients (length: order_b+1)
/// * `order_b` - Second polynomial order
/// * `c` - Output polynomial coefficients (length: order_a + order_b + 1)
pub fn poly_mul<T>(a: &[T], order_a: usize, b: &[T], order_b: usize, c: &mut [T]) -> ()
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + From<f32>,
{
    let na = order_a + 1;
    let nb = order_b + 1;
    let nc = na + nb - 1;

    for i in 0..nc {
        c[i] = T::from(0.0);
    }

    for i in 0..na {
        for j in 0..nb {
            c[i+j] = c[i+j] + a[i] * b[j];
        }
    }
}

/// Lagrange polynomial exact interpolation
///
/// # Arguments
///
/// * `x` - Input array (known) [size: `n` x 1]
/// * `y` - Output array (known) [size: `n` x 1]
/// * `n` - Number of input/output pairs
/// * `x0` - Evaluation point
///
/// # Returns
///
/// Interpolated value `y0`
pub fn poly_interp_lagrange<T>(x: &[T], y: &[T], n: usize, x0: T) -> T
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + From<f32>,
{
    let mut y0 = T::from(0.0);

    for i in 0..n {
        let mut g = T::from(1.0);
        for j in 0..n {
            if i == j {
                continue;
            }
            g = g * (x0 - x[j]) / (x[i] - x[j]);
        }
        y0 = y0 + y[i] * g;
    }

    y0
}

/// Lagrange polynomial interpolation
///
/// # Arguments
///
/// * `x` - Input array (known) [size: `n` x 1]
/// * `y` - Output array (known) [size: `n` x 1]
/// * `n` - Number of input/output pairs
/// * `p` - Output polynomial coefficients [size: `n` x 1]
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn poly_fit_lagrange<T>(x: &[T], y: &[T], n: usize, p: &mut [T]) -> ()
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T> + From<f32>,
{
    let k = n - 1;
    // Clear output array
    for i in 0..n {
        p[i] = T::from(0.0);
    }

    // compute roots, gain
    let mut roots = vec![T::from(0.0); k];
    let mut c = vec![T::from(0.0); n];
    for i in 0..n {
        let mut idx = 0;
        let mut g = T::from(1.0);
        for j in 0..n {
            if i == j {
                continue;
            }
            roots[idx] = x[j];
            g = g * (x[i] - x[j]);
            idx += 1;
        }
        g = y[i] / g;

        // Expand roots
        poly_expandroots(&roots, k, &mut c);

        // Scale by gain
        for j in 0..n {
            p[j] = p[j] + g * c[j];
        }
    }
}

/// Lagrange polynomial fit (barycentric form)
///
/// # Arguments
///
/// * `x` - Input array (known) [size: `n` x 1]
/// * `n` - Number of input/output pairs
/// * `w` - Output barycentric weights [size: `n` x 1]
pub fn poly_fit_lagrange_barycentric<T>(x: &[T], n: usize, w: &mut [T]) -> ()
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T> + From<f32> + PartialEq,
{
    for j in 0..n {
        w[j] = T::from(1.0);
        for k in 0..n {
            if j == k {
                continue;
            }
            w[j] = w[j] * (x[j] - x[k]);
        }

        if w[j] == T::from(0.0) {
            w[j] = w[j] + T::from(1.0e-9);
        }
        w[j] = T::from(1.0) / w[j];
    }

    let w0 = w[0] + T::from(1.0e-9);
    for j in 0..n {
        w[j] = w[j] / w0;
    }
}

/// Lagrange polynomial interpolation (barycentric form)
///
/// # Arguments
///
/// * `x` - Input array (known) [size: `n` x 1]
/// * `y` - Output array (known) [size: `n` x 1]
/// * `w` - Barycentric weights [size: `n` x 1]
/// * `x0` - Evaluation point
/// * `n` - Number of input/output pairs
///
/// # Returns
///
/// Interpolated value `y0`
pub fn poly_val_lagrange_barycentric<T>(x: &[T], y: &[T], w: &[T], x0: T, n: usize) -> T
where
    T: FloatComplex,
{
    let mut t0 = T::from(0.0).unwrap();
    let mut t1 = T::from(0.0).unwrap();
    let tol = T::Real::from(1.0e-6);

    for j in 0..n {
        let g = x0 - x[j];

        if g.abs() < tol {
            return y[j];
        }

        t0 = t0 + w[j] * y[j] / g;
        t1 = t1 + w[j] / g;
    }

    t0 / t1
}


/// Finds the complex roots of the polynomial using the Durand-Kerner method
///
/// # Arguments
///
/// * `p` - Polynomial coefficients (length: `k`)
/// * `k` - Polynomial length (poly order = `k` - 1)
/// * `roots` - Resulting complex roots (length: `k` - 1)
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn poly_findroots_durandkerner(
    p: &[f64],
    k: usize,
    roots: &mut [Complex<f64>],
) -> Result<()> {
    if k < 2 {
        return Err(Error::Range("order must be greater than 0".to_owned()));
    }
    if p[k - 1] != 1.0 {
        return Err(Error::Range("p[k-1] must be equal to 1".to_owned()));
    }

    let num_roots = k - 1;
    let mut r0 = vec![0.0; num_roots];
    let mut r1 = vec![0.0; num_roots];

    // Find initial magnitude
    let mut g: f64;
    let mut gmax = 0.0;
    for i in 0..k {
        g = p[i].abs();
        if i == 0 || g > gmax {
            gmax = g;
        }
    }

    // Initialize roots
    let t0 = 0.9 * (1.0 + gmax) * Complex::new(0.0, 1.1526).exp().re();
    let mut t = 1.0;
    for i in 0..num_roots {
        r0[i] = t;
        t *= t0;
    }

    let max_num_iterations = 50;
    let mut continue_iterating = true;
    let mut i = 0;
    let tol = 1e-6;
    while continue_iterating {
        for j in 0..num_roots {
            let f = poly_val(p, k, r0[j]);
            let mut fp = 1.0;
            for l in 0..num_roots {
                if l == j {
                    continue;
                }
                fp *= r0[j] - r0[l];
            }
            r1[j] = r0[j] - f / fp;
        }

        // Stop iterating if roots have settled
        let mut delta = 0.0;
        for j in 0..num_roots {
            let e = r0[j] - r1[j];
            delta += (e * e.conj()).re();
        }
        delta /= num_roots as f64 * gmax;

        if delta < tol || i == max_num_iterations {
            continue_iterating = false;
        }

        r0.copy_from_slice(&r1);
        i += 1;
    }

    for i in 0..k {
        roots[i] = Complex::new(r1[i], 0.0);
    }
    Ok(())
}

/// Finds the complex roots of the polynomial using Bairstow's method
///
/// # Arguments
///
/// * `p` - Polynomial coefficients (length: `k`)
/// * `k` - Polynomial length (poly order = `k` - 1)
/// * `roots` - Resulting complex roots (length: `k` - 1)
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn poly_findroots_bairstow(
    p: &[f64],
    k: usize,
    roots: &mut [Complex<f64>],
) -> Result<()> {
    let mut p0 = vec![0.0; k];
    let mut p1 = vec![0.0; k];
    let mut p_buf;
    let mut pr_buf = &mut p1;

    p0.copy_from_slice(p);

    let mut n = k;
    let r = k % 2;
    let l = (k - r) / 2;
    let mut root_index = 0;
    for i in 0..l - 1 + r {
        // Set polynomial and reduced polynomial buffer pointers
        if i % 2 == 0 {
            p_buf = &mut p0;
            pr_buf = &mut p1;
        } else {
            p_buf = &mut p1;
            pr_buf = &mut p0;
        }

        // Initial estimates for u, v
        if p_buf[n - 1] == 0.0 {
            // TODO maybe error case
            p_buf[n - 1] = 1e-12;
        }
        let mut u = p_buf[n - 2] / p_buf[n - 1];
        let mut v = p_buf[n - 3] / p_buf[n - 1];

        // Compute factor using Bairstow's recursion
        if n > 3 {
            poly_findroots_bairstow_persistent(p_buf, n, pr_buf, &mut u, &mut v)?;
        }

        // Compute complex roots of x^2 + u*x + v
        let r0 = Complex::new(-0.5 * u, 0.0) + 0.5 * Complex::new(u * u - 4.0 * v, 0.0).sqrt();
        let r1 = Complex::new(-0.5 * u, 0.0) - 0.5 * Complex::new(u * u - 4.0 * v, 0.0).sqrt();

        roots[root_index] = r0;
        roots[root_index + 1] = r1;
        root_index += 2;

        n -= 2;
    }

    if r == 0 {
        roots[root_index] = Complex::new(-pr_buf[0] / pr_buf[1], 0.0);
    }

    Ok(())
}

/// Iterate over Bairstow's method, finding quadratic factor x^2 + u*x + v
///
/// # Arguments
///
/// * `p` - Polynomial coefficients (length: `k`)
/// * `k` - Polynomial length (poly order = `k` - 1)
/// * `p1` - Reduced polynomial (output) (length: `k` - 2)
/// * `u` - Input: initial estimate for u; output: resulting u
/// * `v` - Input: initial estimate for v; output: resulting v
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
fn poly_findroots_bairstow_recursion(
    p: &[f64],
    k: usize,
    p1: &mut [f64],
    u: &mut f64,
    v: &mut f64,
) -> Result<()> {
    if k < 3 {
        return Err(Error::Range(format!("invalid polynomial length: {}", k)));
    }

    let tol = 1e-12;
    let num_iterations_max = 20;

    let n = k - 1;
    let mut b = vec![0.0; k];
    let mut f = vec![0.0; k];
    b[n] = 0.0;
    b[n - 1] = 0.0;
    f[n] = 0.0;
    f[n - 1] = 0.0;

    let mut num_iterations = 0;
    loop {
        if num_iterations == num_iterations_max {
            return Err(Error::NoConvergence(format!("failed to converge after {} iterations", num_iterations_max)));
        }
        num_iterations += 1;

        // Update reduced polynomial coefficients
        for i in (0..=n - 2).rev() {
            b[i] = p[i + 2] - *u * b[i + 1] - *v * b[i + 2];
            f[i] = b[i + 2] - *u * f[i + 1] - *v * f[i + 2];
        }
        let c = p[1] - *u * b[0] - *v * b[1];
        let g = b[1] - *u * f[0] - *v * f[1];
        let d = p[0] - *v * b[0];
        let h = b[0] - *v * f[0];

        // Compute scaling factor
        let q0 = *v * g * g;
        let q1 = h * (h - *u * g);
        let metric = (q0 + q1).abs();
        if metric < tol {
            *u *= 0.5;
            *v *= 0.5;
            continue;
        }

        let q = 1.0 / (*v * g * g + h * (h - *u * g));

        // Compute u, v steps
        let du = -q * (-h * c + g * d);
        let dv = -q * (-g * *v * c + (g * *u - h) * d);

        let step = du.abs() + dv.abs();

        // Adjust u, v by step size
        *u += du;
        *v += dv;

        // Exit conditions
        if step < tol {
            break;
        }
    }

    // Set resulting reduced polynomial
    for i in 0..k - 2 {
        p1[i] = b[i];
    }

    Ok(())
}

/// Run multiple iterations of Bairstow's method with different starting
/// conditions looking for convergence
///
/// # Arguments
///
/// * `p` - Polynomial coefficients (length: `k`)
/// * `k` - Polynomial length (poly order = `k` - 1)
/// * `p1` - Reduced polynomial (output) (length: `k` - 2)
/// * `u` - Input: initial estimate for u; output: resulting u
/// * `v` - Input: initial estimate for v; output: resulting v
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
fn poly_findroots_bairstow_persistent(
    p: &[f64],
    k: usize,
    p1: &mut [f64],
    u: &mut f64,
    v: &mut f64,
) -> Result<()> {
    let num_iterations_max = 10;
    for i in 0..num_iterations_max {
        if poly_findroots_bairstow_recursion(p, k, p1, u, v).is_ok() {
            return Ok(());
        } else if i < num_iterations_max - 1 {
            // Didn't converge; adjust starting point using consistent and
            // reproduceable starting point
            *u = (i as f64 * 1.1).cos() * (i as f64 * 0.2).exp();
            *v = (i as f64 * 1.1).sin() * (i as f64 * 0.2).exp();
        }
    }

    // Could not converge
    Err(Error::NoConvergence(format!("failed to converge after {} iterations", num_iterations_max)))
}

/// Compare roots for sorting
pub fn poly_sort_roots_compare(a: &Complex<f64>, b: &Complex<f64>) -> Ordering {
    let ar = a.re;
    let br = b.re;
    let ai = a.im;
    let bi = b.im;

    if ar == br {
        if ai > bi {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    } else if ar > br {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

/// Finds the complex roots of the polynomial
///
/// # Arguments
///
/// * `p` - Polynomial coefficients, ascending powers [size: `k` x 1]
/// * `k` - Polynomials length (poly order = `k` - 1)
/// * `roots` - Resulting complex roots [size: `k-1` x 1]
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn polyf_findroots(p: &[f32], k: usize, roots: &mut [Complex<f32>]) -> Result<()> {
    if k < 2 {
        return Err(Error::Range(format!("invalid polynomial length: {}", k)));
    }

    // Copy to temporary double-precision array
    let p_double: Vec<f64> = p.iter().map(|&x| x as f64).collect();

    // Find roots of polynomial using Bairstow's method (more
    // accurate and reliable than Durand-Kerner)
    let mut roots_double = vec![Complex::new(0.0, 0.0); k - 1];
    poly_findroots_bairstow(&p_double, k, &mut roots_double)?;

    // Sort roots for consistent ordering
    roots_double.sort_by(poly_sort_roots_compare);

    // Copy back to original
    for i in 0..k - 1 {
        roots[i] = Complex::new(roots_double[i].re as f32, roots_double[i].im as f32);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use test_macro::autotest_annotate;

    // TODO tests for poly_fit_lagrange_barycentric and poly_interp_lagrange_barycentric

    #[test]
    #[autotest_annotate(autotest_polyf_fit_q3n3)]
    fn test_polyf_fit_q3n3() {
        let x = [-1.0, 0.0, 1.0];
        let y = [1.0, 0.0, 1.0];
        let p_test = [0.0, 0.0, 1.0];
        let n = 3;
        let mut p = [0.0; 3];
        let k = 3;
        let tol = 1e-3;

        poly_fit(&x, &y, n, &mut p, k).unwrap();

        assert_relative_eq!(p[0], p_test[0], epsilon = tol);
        assert_relative_eq!(p[1], p_test[1], epsilon = tol);
        assert_relative_eq!(p[2], p_test[2], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_lagrange_issue165)]
    fn test_polyf_lagrange_issue165() {
        let x = [-1.0, 0.0, 1.0];
        let y = [7.059105, 24.998369, 14.365907];
        let mut p = [0.0f32; 3];
        let n = 3;
        let tol = 1e-3;

        poly_fit_lagrange(&x, &y, n, &mut p);

        let mut y_out = vec![0.0; n];
        for j in 0..n {
            y_out[j] = poly_val(&p, n, x[j]);
            assert_relative_eq!(y[j], y_out[j], epsilon = tol);
        }

        poly_fit(&x, &y, n, &mut p, n).unwrap();

        for j in 0..n {
            y_out[j] = poly_val(&p, n, x[j]);
            assert_relative_eq!(y[j], y_out[j], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_polyf_lagrange)]
    fn test_polyf_lagrange() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 8.0, 27.0];
        let mut p = [0.0f32; 3];
        let n = 3;
        let tol = 1e-3f32;

        poly_fit_lagrange(&x, &y, n, &mut p);

        let mut y_out = vec![0.0; n];
        for j in 0..n {
            y_out[j] = poly_val(&p, n, x[j]);
            assert_relative_eq!(y[j], y_out[j], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_polyf_expandroots_4)]
    fn test_polyf_expandroots_4() {
        let roots = [-2.0, -1.0, -4.0, 5.0, 3.0];
        let c_test = [120.0, 146.0, 1.0, -27.0, -1.0, 1.0];
        let n = 5;
        let mut p = vec![0.0; n+1];
        let tol = 1e-3;

        poly_expandroots(&roots, n, &mut p);
        assert_relative_eq!(p[0], c_test[0], epsilon = tol);
        assert_relative_eq!(p[1], c_test[1], epsilon = tol);
        assert_relative_eq!(p[2], c_test[2], epsilon = tol);
        assert_relative_eq!(p[3], c_test[3], epsilon = tol);
        assert_relative_eq!(p[4], c_test[4], epsilon = tol);
        assert_relative_eq!(p[5], c_test[5], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_expandroots_11)]
    fn test_polyf_expandroots_11() {
        let roots = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0];
        let c_test = [39916800.0, 120543840.0, 150917976.0, 105258076.0, 45995730.0, 13339535.0, 2637558.0, 357423.0, 32670.0, 1925.0, 66.0, 1.0];
        let n = 11;
        let mut p = vec![0.0; n+1];
        let tol = 1e-6f32;

        poly_expandroots(&roots, n, &mut p);
        assert_relative_eq!(p[0], c_test[0], epsilon = (tol*c_test[0]).abs());
        assert_relative_eq!(p[1], c_test[1], epsilon = (tol*c_test[1]).abs());
        assert_relative_eq!(p[2], c_test[2], epsilon = (tol*c_test[2]).abs());
        assert_relative_eq!(p[3], c_test[3], epsilon = (tol*c_test[3]).abs());
        assert_relative_eq!(p[4], c_test[4], epsilon = (tol*c_test[4]).abs());
        assert_relative_eq!(p[5], c_test[5], epsilon = (tol*c_test[5]).abs());
        assert_relative_eq!(p[6], c_test[6], epsilon = (tol*c_test[6]).abs());
        assert_relative_eq!(p[7], c_test[7], epsilon = (tol*c_test[7]).abs());
        assert_relative_eq!(p[8], c_test[8], epsilon = (tol*c_test[8]).abs());
        assert_relative_eq!(p[9], c_test[9], epsilon = (tol*c_test[9]).abs());
        assert_relative_eq!(p[10], c_test[10], epsilon = (tol*c_test[10]).abs());
        assert_relative_eq!(p[11], c_test[11], epsilon = (tol*c_test[11]).abs());
    }

    #[test]
    #[autotest_annotate(autotest_polycf_expandroots_4)]
    fn test_polycf_expandroots_4() {
        use num_complex::Complex32;

        let theta = 1.7f32;
        let a = [
            -Complex32::new(0.0, theta).exp(),
            -Complex32::new(0.0, -theta).exp()
        ];
        let mut c = vec![Complex32::new(0.0, 0.0); 3];
        let c_test = [
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0 * theta.cos(), 0.0),
            Complex32::new(1.0, 0.0)
        ];
        let tol = 1e-3f32;

        poly_expandroots(&a, 2, &mut c);

        assert_relative_eq!(c[0].re, c_test[0].re, epsilon = tol);
        assert_relative_eq!(c[0].im, c_test[0].im, epsilon = tol);

        assert_relative_eq!(c[1].re, c_test[1].re, epsilon = tol);
        assert_relative_eq!(c[1].im, c_test[1].im, epsilon = tol);

        assert_relative_eq!(c[2].re, c_test[2].re, epsilon = tol);
        assert_relative_eq!(c[2].im, c_test[2].im, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_expandroots2_3)]
    fn test_polyf_expandroots2_3() {
        let n = 3;
        let a = [2.0, 3.0, -1.0];
        let b = [5.0, -2.0, -3.0];
        let mut c = vec![0.0f32; n + 1];
        let c_test = [-6.0, 29.0, -23.0, -30.0];
        let tol = 1e-3f32;

        poly_expandroots2(&a, &b, n, &mut c);

        assert_relative_eq!(c[0], c_test[0], epsilon = tol);
        assert_relative_eq!(c[1], c_test[1], epsilon = tol);
        assert_relative_eq!(c[2], c_test[2], epsilon = tol);
        assert_relative_eq!(c[3], c_test[3], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_mul_2_3)]
    fn test_polyf_mul_2_3() {
        let a = [2.0, -4.0, 3.0];
        let b = [-9.0, 3.0, -2.0, 5.0];
        let mut c = vec![0.0f32; 6];
        let c_test = [-18.0, 42.0, -43.0, 27.0, -26.0, 15.0];
        let tol = 1e-3f32;

        poly_mul(&a, 2, &b, 3, &mut c);

        assert_relative_eq!(c[0], c_test[0], epsilon = tol);
        assert_relative_eq!(c[1], c_test[1], epsilon = tol);
        assert_relative_eq!(c[2], c_test[2], epsilon = tol);
        assert_relative_eq!(c[3], c_test[3], epsilon = tol);
        assert_relative_eq!(c[4], c_test[4], epsilon = tol);
        assert_relative_eq!(c[5], c_test[5], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_poly_expandbinomial_n6)]
    fn test_poly_expandbinomial_n6() {
        let n = 6;
        let mut c = vec![0.0f32; n + 1];
        let c_test = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0];
        let tol = 1e-3f32;

        poly_expandbinomial(n, &mut c);

        assert_relative_eq!(c[0], c_test[0], epsilon = tol);
        assert_relative_eq!(c[1], c_test[1], epsilon = tol);
        assert_relative_eq!(c[2], c_test[2], epsilon = tol);
        assert_relative_eq!(c[3], c_test[3], epsilon = tol);
        assert_relative_eq!(c[4], c_test[4], epsilon = tol);
        assert_relative_eq!(c[5], c_test[5], epsilon = tol);
        assert_relative_eq!(c[6], c_test[6], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_poly_binomial_expand_pm_m6_k1)]
    fn test_poly_expandbinomial_pm_m6_k1() {
        let m = 5;
        let k = 1;
        let n = m + k;
        let mut c = vec![0.0f32; n + 1];
        let c_test = [1.0, 4.0, 5.0, 0.0, -5.0, -4.0, -1.0];
        let tol = 1e-3f32;

        poly_expandbinomial_pm(m, k, &mut c);

        assert_relative_eq!(c[0], c_test[0], epsilon = tol);
        assert_relative_eq!(c[1], c_test[1], epsilon = tol);
        assert_relative_eq!(c[2], c_test[2], epsilon = tol);
        assert_relative_eq!(c[3], c_test[3], epsilon = tol);
        assert_relative_eq!(c[4], c_test[4], epsilon = tol);
        assert_relative_eq!(c[5], c_test[5], epsilon = tol);
        assert_relative_eq!(c[6], c_test[6], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_poly_expandbinomial_pm_m5_k2)]
    fn test_poly_expandbinomial_pm_m5_k2() {
        let m = 5;
        let k = 2;
        let n = m + k;
        let mut c = vec![0.0f32; n + 1];
        let c_test = [1.0, 3.0, 1.0, -5.0, -5.0, 1.0, 3.0, 1.0];
        let tol = 1e-3f32;

        poly_expandbinomial_pm(m, k, &mut c);

        assert_relative_eq!(c[0], c_test[0], epsilon = tol);
        assert_relative_eq!(c[1], c_test[1], epsilon = tol);
        assert_relative_eq!(c[2], c_test[2], epsilon = tol);
        assert_relative_eq!(c[3], c_test[3], epsilon = tol);
        assert_relative_eq!(c[4], c_test[4], epsilon = tol);
        assert_relative_eq!(c[5], c_test[5], epsilon = tol);
        assert_relative_eq!(c[6], c_test[6], epsilon = tol);
        assert_relative_eq!(c[7], c_test[7], epsilon = tol);
    }

    /// Polynomial root-finding testbench
    ///
    /// # Arguments
    ///
    /// * `p` - Polynomial coefficients (length: _order+1)
    /// * `r` - Roots (sorted) (length: _order)
    /// * `order` - Polynomial order
    /// * `tol` - Error tolerance
    fn polyf_findroots_testbench(p: &[f32], r: &[Complex<f32>], order: usize, tol: f32) {
        let mut roots = vec![Complex::new(0.0, 0.0); order];

        // Run Durand-Kerner method
        assert!(polyf_findroots(p, order + 1, &mut roots).is_ok());

        // Verify roots
        for i in 0..order {
            let e = (roots[i] - r[i]).norm();
            assert!(e < tol, "root {} : {:e} > {:e}", i, e, tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_polyf_findroots_real)]
    fn test_polyf_findroots_real() {
        // Basic roots, no complex values
        let p = [6.0, 11.0, -33.0, -33.0, 11.0, 6.0];
        let r = [
            Complex::new(-3.0, 0.0),
            Complex::new(-1.0, 0.0),
            Complex::new(-1.0 / 3.0, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(2.0, 0.0),
        ];
        polyf_findroots_testbench(&p, &r, 5, 1e-6);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_findroots_complex)]
    fn test_polyf_findroots_complex() {
        // Complex roots
        let p = [3.0, 2.0, 1.0];
        let r = [
            Complex::new(-1.0, std::f32::consts::SQRT_2),
            Complex::new(-1.0, -std::f32::consts::SQRT_2),
        ];
        polyf_findroots_testbench(&p, &r, 2, 1e-6);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_findroots_mix)]
    fn test_polyf_findroots_mix() {
        // Complex roots
        let p = [-1.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let r = [
            Complex::new(-1.544928106217380, 0.0),
            Complex::new(-0.8438580445415772, 1.251293921227189),
            Complex::new(-0.8438580445415772, -1.251293921227189),
            Complex::new(0.1464465720078399, 0.0),
            Complex::new(0.5430988116463471, 1.282747429218130),
            Complex::new(0.5430988116463471, -1.282747429218130),
        ];
        polyf_findroots_testbench(&p, &r, 6, 1e-6);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_findroots_mix2)]
    fn test_polyf_findroots_mix2() {
        // Complex roots, longer polynomial
        let p = [
            -2.1218292415142059326171875000e-02,
            1.6006522178649902343750000000e+00,
            -1.2054302543401718139648437500e-01,
            -8.4453743696212768554687500000e-01,
            -1.1174567937850952148437500000e+00,
            8.2108253240585327148437500000e-01,
            2.2316795587539672851562500000e-01,
            1.4220994710922241210937500000e+00,
            -8.4215706586837768554687500000e-01,
            1.3681684434413909912109375000e-01,
            1.0689756833016872406005859375e-02,
        ];
        let r = [
            Complex::new(-17.67808709752869, 0.0),
            Complex::new(-0.7645511425850682, 0.4932343666704793),
            Complex::new(-0.7645511425850682, -0.4932343666704793),
            Complex::new(-0.2764509491715267, 1.058805768356938),
            Complex::new(-0.2764509491715267, -1.058805768356938),
            Complex::new(0.01327054605125156, 0.0),
            Complex::new(0.9170364272114475, 0.3838341863217226),
            Complex::new(0.9170364272114475, -0.3838341863217226),
            Complex::new(2.556937242081334, 1.448576080447611),
            Complex::new(2.556937242081334, -1.448576080447611),
        ];
        polyf_findroots_testbench(&p, &r, 10, 4e-6);
    }
    
    #[test]
    fn test_poly_val_f32() {
        let p = [1.0, -4.0, 6.0, 2.0];
        let k = 4;
        let x = 3.0;
        let y = poly_val(&p, k, x);
        assert_relative_eq!(y, 97.0, epsilon = 1e-6);
    }

}