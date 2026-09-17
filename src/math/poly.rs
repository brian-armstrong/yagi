use num_complex::Complex;
use num_traits::{Float, One, Zero};
use std::cmp::Ordering;

use crate::error::{Error, Result};
use crate::matrix::{ColMatrixMut, FloatComplex, LeastSquaresWorkspace};

/// Evaluate polynomial
///
/// Computes the value `y` of polynomial `p` at `x` using successive powers.
///
/// # Arguments
///
/// * `p` - Polynomial coefficients
/// * `x` - Input value
///
/// # Returns
///
/// Polynomial evaluated at `x`
pub fn poly_val<T>(p: &[T], x: T) -> T
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + One + Zero,
{
    let mut xk = T::one();
    let mut y = T::zero();
    for &pi in p {
        y = y + pi * xk;
        xk = xk * x;
    }
    y
}

/// Fit a polynomial to a set of data points with Householder QR.
///
/// The output is left unchanged when the inputs are invalid, non-finite,
/// or numerically rank deficient.
///
/// # Arguments
///
/// * `x` - Independent variable
/// * `y` - Dependent variable
/// * `p` - Output polynomial coefficients
///
/// # Returns
///
/// `Ok(())` on success, or an error when the dimensions, values, or numerical
/// rank do not allow a finite fit.
pub fn poly_fit<T>(x: &[T], y: &[T], p: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    if x.len() != y.len() {
        return Err(Error::Range("independent and dependent variable slices must have equal lengths".to_owned()));
    }
    if y.iter().any(|value| !value.abs().is_finite()) {
        return Err(Error::Value("polynomial fit inputs must be finite".to_owned()));
    }

    let mut workspace = vandermonde_workspace(x, p.len())?;
    workspace.solve(y, p)
}

/// Build and factor the Vandermonde matrix for a polynomial fit over `x`.
///
/// Column `k` holds `x[row]^k`, so a solve against this workspace fits a
/// polynomial with `coefficient_count` coefficients through samples at `x`.
///
/// # Arguments
///
/// * `x` - Independent variable, one sample per row
/// * `coefficient_count` - Number of polynomial coefficients (columns)
pub(crate) fn vandermonde_workspace<T>(x: &[T], coefficient_count: usize) -> Result<LeastSquaresWorkspace<T>>
where
    T: FloatComplex,
{
    if coefficient_count == 0 {
        return Err(Error::Range("coefficient slice must not be empty".to_owned()));
    }
    if x.len() < coefficient_count {
        return Err(Error::Range("sample count must be at least the coefficient count".to_owned()));
    }
    if x.iter().any(|value| !value.abs().is_finite()) {
        return Err(Error::Value("polynomial fit inputs must be finite".to_owned()));
    }

    let rows = x.len();
    let matrix_len = rows
        .checked_mul(coefficient_count)
        .ok_or_else(|| Error::Range("Vandermonde matrix dimensions overflow".to_owned()))?;

    // construct a column-major view of the Vandermonde matrix
    let mut matrix_storage = vec![T::zero(); matrix_len];
    let mut matrix = ColMatrixMut::new(&mut matrix_storage, rows, coefficient_count)?;

    // first column: all 1s
    matrix.column_mut(0).fill(T::one());

    for column in 1..coefficient_count {
        let (previous, current) =
            matrix.two_columns_mut(column - 1, column).expect("adjacent Vandermonde columns are in bounds");
        for ((value, &previous_power), &sample) in current.iter_mut().zip(previous.iter()).zip(x) {
            // matrix[(row, column)] = matrix[(row, column - 1)] * x[row]
            let power = previous_power * sample;
            if !power.abs().is_finite() {
                return Err(Error::Value("Vandermonde matrix contains a non-finite value".to_owned()));
            }
            *value = power;
        }
    }

    LeastSquaresWorkspace::from_design_matrix(matrix.as_ref())
}

/// Expands the polynomial:
///  P_n(x) = (1+x)^n
/// as
///  P_n(x) = p`[0]` + p`[1]`*x + p`[2]`*x^2 + ... + p`[n]`x^n
///
/// # Arguments
///
/// * `n` - Polynomial order
/// * `c` - Output polynomial coefficients, with length `n + 1`
pub fn poly_expandbinomial<T>(n: usize, c: &mut [T]) -> Result<()>
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + From<f32>,
{
    let length = n.checked_add(1).ok_or_else(|| Error::Range("binomial coefficient length overflows".to_owned()))?;
    if c.len() != length {
        return Err(Error::Range("coefficient slice length must equal n + 1".to_owned()));
    }
    if n == 0 {
        c[0] = T::from(0.0);
        return Ok(());
    }

    // Initialize coefficients array to [1,0,0,....0]
    c[0] = T::from(1.0);
    for ci in &mut c[1..] {
        *ci = T::from(0.0);
    }

    // Iterative polynomial multiplication
    for i in 0..n {
        for j in (1..=i + 1).rev() {
            c[j] = c[j] + c[j - 1];
        }
    }
    Ok(())
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
pub fn poly_expandbinomial_pm<T>(m: usize, k: usize, c: &mut [T]) -> Result<()>
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + From<f32>,
{
    let n = m.checked_add(k).ok_or_else(|| Error::Range("binomial order overflows".to_owned()))?;
    let length = n.checked_add(1).ok_or_else(|| Error::Range("binomial coefficient length overflows".to_owned()))?;
    if c.len() != length {
        return Err(Error::Range("coefficient slice length must equal m + k + 1".to_owned()));
    }
    if n == 0 {
        c[0] = T::from(0.0);
        return Ok(());
    }

    // Initialize coefficients array to [1,0,0,....0]
    c[0] = T::from(1.0);
    for ci in &mut c[1..] {
        *ci = T::from(0.0);
    }

    // Iterative polynomial multiplication (1+x)
    for i in 0..m {
        for j in (1..=i + 1).rev() {
            c[j] = c[j] + c[j - 1];
        }
    }

    // Iterative polynomial multiplication (1-x)
    for i in m..n {
        for j in (1..=i + 1).rev() {
            c[j] = c[j] - c[j - 1];
        }
    }
    Ok(())
}

/// Perform root expansion on the polynomial:
///  P_n(x) = (x-r`[0]`) * (x-r`[1]`) * ... * (x-r[n-1])
/// as
///  P_n(x) = p`[0]` + p`[1]`*x + ... + p`[n]`*x^n
/// where r`[0]`,r`[1]`,...,r[n-1] are the roots of P_n(x)
///
/// # Arguments
///
/// * `r` - Roots of polynomial
/// * `p` - Output polynomial coefficients, one longer than `r`
pub fn poly_expandroots<T>(r: &[T], p: &mut [T]) -> Result<()>
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Neg<Output = T> + From<f32>,
{
    let length = r.len().checked_add(1).ok_or_else(|| Error::Range("coefficient length overflows".to_owned()))?;
    if p.len() != length {
        return Err(Error::Range("coefficient slice must be one element longer than root slice".to_owned()));
    }
    if r.is_empty() {
        p[0] = T::from(0.0);
        return Ok(());
    }

    // Initialize coefficients array to [1,0,0,....0]
    p[0] = T::from(1.0);
    for pi in &mut p[1..] {
        *pi = T::from(0.0);
    }

    // Iterative polynomial multiplication
    for (i, &root) in r.iter().enumerate() {
        for j in (1..=i + 1).rev() {
            p[j] = -root * p[j] + p[j - 1];
        }
        p[0] = -root * p[0];
    }
    Ok(())
}

/// Perform root expansion on the polynomial:
///  P_n(x) = (x*b`[0]`-a`[0]`) * (x*b`[1]`-a`[1]`) * ... * (x*b[n-1]-a[n-1])
/// as
///  P_n(x) = p`[0]` + p`[1]`*x + ... + p`[n]`*x^n
///
/// # Arguments
///
/// * `a` - Subtractant of polynomial roots
/// * `b` - Multiplicant of polynomial roots, the same length as `a`
/// * `p` - Output polynomial coefficients, one longer than `a`
pub fn poly_expandroots2<T>(a: &[T], b: &[T], p: &mut [T]) -> Result<()>
where
    T: Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Neg<Output = T>
        + std::ops::Div<Output = T>
        + From<f32>,
{
    if a.len() != b.len() {
        return Err(Error::Range("root slices must have equal lengths".to_owned()));
    }
    let length = a.len().checked_add(1).ok_or_else(|| Error::Range("coefficient length overflows".to_owned()))?;
    if p.len() != length {
        return Err(Error::Range("coefficient slice must be one element longer than root slices".to_owned()));
    }

    // Factor b[i] from each root : (x*b - a) = (x - a/b)*b
    let mut g = T::from(1.0);
    let mut roots = Vec::with_capacity(a.len());
    for (&a, &b) in a.iter().zip(b) {
        g = g * -b;
        roots.push(a / b);
    }

    // Expand new root set
    poly_expandroots(&roots, p)?;

    // Multiply by gain
    for pi in p {
        *pi = g * *pi;
    }
    Ok(())
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
/// * `a` - First polynomial coefficients
/// * `b` - Second polynomial coefficients
/// * `c` - Output polynomial coefficients, with length `a.len() + b.len() - 1`
pub fn poly_mul<T>(a: &[T], b: &[T], c: &mut [T]) -> Result<()>
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + From<f32>,
{
    if a.is_empty() || b.is_empty() {
        return Err(Error::Range("input polynomial slices must not be empty".to_owned()));
    }
    let nc = a
        .len()
        .checked_add(b.len())
        .and_then(|length| length.checked_sub(1))
        .ok_or_else(|| Error::Range("product polynomial length overflows".to_owned()))?;
    if c.len() != nc {
        return Err(Error::Range("product slice length must equal a.len() + b.len() - 1".to_owned()));
    }
    c.fill(T::from(0.0));

    for (i, &ai) in a.iter().enumerate() {
        for (ci, &bj) in c[i..i + b.len()].iter_mut().zip(b) {
            *ci = *ci + ai * bj;
        }
    }
    Ok(())
}

/// Lagrange polynomial exact interpolation
///
/// # Arguments
///
/// * `x` - Input array (known)
/// * `y` - Output array (known), the same length as `x`
/// * `x0` - Evaluation point
///
/// # Returns
///
/// Interpolated value `y0`
pub fn poly_interp_lagrange<T>(x: &[T], y: &[T], x0: T) -> Result<T>
where
    T: Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + From<f32>,
{
    if x.len() != y.len() {
        return Err(Error::Range("input and output sample slices must have equal lengths".to_owned()));
    }
    let mut y0 = T::from(0.0);

    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        let mut g = T::from(1.0);
        for (j, &xj) in x.iter().enumerate() {
            if i == j {
                continue;
            }
            g = g * (x0 - xj) / (xi - xj);
        }
        y0 = y0 + yi * g;
    }

    Ok(y0)
}

/// Lagrange polynomial interpolation
///
/// # Arguments
///
/// * `x` - Input array (known)
/// * `y` - Output array (known), the same length as `x`
/// * `p` - Output polynomial coefficients, the same length as `x`
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn poly_fit_lagrange<T>(x: &[T], y: &[T], p: &mut [T]) -> Result<()>
where
    T: Copy
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + From<f32>,
{
    if x.len() != y.len() || p.len() != x.len() {
        return Err(Error::Range("input, output, and coefficient slices must have equal lengths".to_owned()));
    }
    let Some(k) = x.len().checked_sub(1) else {
        return Ok(());
    };

    // Clear output array
    p.fill(T::from(0.0));

    // compute roots, gain
    let mut roots = vec![T::from(0.0); k];
    let mut c = vec![T::from(0.0); p.len()];
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        let mut idx = 0;
        let mut g = T::from(1.0);
        for (j, &xj) in x.iter().enumerate() {
            if i == j {
                continue;
            }
            roots[idx] = xj;
            g = g * (xi - xj);
            idx += 1;
        }
        g = yi / g;

        // Expand roots
        poly_expandroots(&roots, &mut c)?;

        // Scale by gain
        for (coefficient, &ci) in p.iter_mut().zip(&c) {
            *coefficient = *coefficient + g * ci;
        }
    }
    Ok(())
}

/// Lagrange polynomial fit (barycentric form)
///
/// # Arguments
///
/// * `x` - Input array (known)
/// * `w` - Output barycentric weights, the same length as `x`
pub fn poly_fit_lagrange_barycentric<T>(x: &[T], w: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    if x.len() != w.len() {
        return Err(Error::Range("input and weight slices must have equal lengths".to_owned()));
    }
    if x.is_empty() {
        return Err(Error::Range("input and weight slices must not be empty".to_owned()));
    }
    if !x.iter().all(|value| value.is_finite()) {
        return Err(Error::Value("interpolation nodes must be finite".to_owned()));
    }

    for (j, weight) in w.iter_mut().enumerate() {
        let xj = x[j];
        *weight = T::one();
        for (k, &xk) in x.iter().enumerate() {
            if j == k {
                continue;
            }
            *weight *= xj - xk;
        }
        if *weight == T::zero() || !weight.is_finite() {
            return Err(Error::Value("interpolation nodes must be distinct and produce finite weights".to_owned()));
        }
        *weight = T::one() / *weight;
    }

    let w0 = w[0];
    for wj in w {
        *wj = *wj / w0;
        if !wj.is_finite() {
            return Err(Error::Value("normalized barycentric weight is not finite".to_owned()));
        }
    }
    Ok(())
}

/// Lagrange polynomial interpolation (barycentric form)
///
/// # Arguments
///
/// * `x` - Distinct, finite interpolation nodes
/// * `y` - Finite values at each node, the same length as `x`
/// * `w` - Finite, nonzero barycentric weights produced for `x`
/// * `x0` - Finite evaluation point
///
/// # Returns
///
/// Interpolated value `y0`
pub fn poly_val_lagrange_barycentric<T>(x: &[T], y: &[T], w: &[T], x0: T) -> Result<T>
where
    T: FloatComplex,
{
    if x.len() != y.len() || x.len() != w.len() {
        return Err(Error::Range("input, output, and weight slices must have equal lengths".to_owned()));
    }
    if x.is_empty() {
        return Err(Error::Range("input, output, and weight slices must not be empty".to_owned()));
    }
    if !x0.is_finite() {
        return Err(Error::Value("interpolation point must be finite".to_owned()));
    }
    if x.iter().chain(y).chain(w).any(|value| !value.is_finite()) || w.iter().any(|&weight| weight == T::zero()) {
        return Err(Error::Value(
            "interpolation values and weights must be finite and weights must be nonzero".to_owned(),
        ));
    }

    let mut t0 = T::zero();
    let mut t1 = T::zero();

    for ((&xj, &yj), &wj) in x.iter().zip(y).zip(w) {
        if x0 == xj {
            return if yj.is_finite() {
                Ok(yj)
            } else {
                Err(Error::Value("barycentric interpolation result is not finite".to_owned()))
            };
        }

        let g = x0 - xj;
        let scaled_weight = wj / g;
        t0 += scaled_weight * yj;
        t1 += scaled_weight;
    }

    let value = t0 / t1;
    if !value.is_finite() {
        return Err(Error::Value("barycentric interpolation result is not finite".to_owned()));
    }
    Ok(value)
}

/// Finds the complex roots of the polynomial using the Durand-Kerner method
///
/// # Arguments
///
/// * `p` - Polynomial coefficients
/// * `roots` - Resulting complex roots, one fewer than `p`
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn poly_findroots_durandkerner(p: &[f64], roots: &mut [Complex<f64>]) -> Result<()> {
    if p.len() < 2 {
        return Err(Error::Range("order must be greater than 0".to_owned()));
    }
    if roots.len() != p.len() - 1 {
        return Err(Error::Range("root slice length must be one less than coefficient slice length".to_owned()));
    }
    if p.last() != Some(&1.0) {
        return Err(Error::Range("leading coefficient must be equal to 1".to_owned()));
    }

    let p_complex: Vec<_> = p.iter().map(|&coefficient| Complex::new(coefficient, 0.0)).collect();
    let mut r0 = vec![Complex::new(0.0, 0.0); roots.len()];
    let mut r1 = vec![Complex::new(0.0, 0.0); roots.len()];

    // Find initial magnitude
    let mut gmax = 0.0;
    for (i, &coefficient) in p.iter().enumerate() {
        let g = coefficient.abs();
        if i == 0 || g > gmax {
            gmax = g;
        }
    }

    // Initialize roots
    let t0 = Complex::from_polar(0.9 * (1.0 + gmax), 1.1526);
    let mut t = Complex::new(1.0, 0.0);
    for r0i in &mut r0 {
        *r0i = t;
        t *= t0;
    }

    let max_num_iterations = 50;
    let tol = 1e-6;
    for iteration in 0..=max_num_iterations {
        for j in 0..r0.len() {
            let f = poly_val(&p_complex, r0[j]);
            let mut fp = Complex::new(1.0, 0.0);
            for l in 0..r0.len() {
                if l == j {
                    continue;
                }
                fp *= r0[j] - r0[l];
            }
            if fp.norm_sqr() == 0.0 {
                return Err(Error::NoConvergence("encountered coincident root estimates".to_owned()));
            }
            r1[j] = r0[j] - f / fp;
        }

        // Stop iterating if roots have settled
        let mut delta = 0.0;
        for (&r0j, &r1j) in r0.iter().zip(&r1) {
            let e = r0j - r1j;
            delta += e.norm_sqr();
        }
        delta /= r0.len() as f64 * gmax;

        if delta < tol {
            roots.copy_from_slice(&r1);
            return Ok(());
        }
        r0.copy_from_slice(&r1);
        if iteration == max_num_iterations {
            return Err(Error::NoConvergence(format!("failed to converge after {} iterations", max_num_iterations)));
        }
    }

    unreachable!("the bounded Durand-Kerner iteration always returns")
}

/// Finds the complex roots of the polynomial using Bairstow's method
///
/// # Arguments
///
/// * `p` - Polynomial coefficients
/// * `roots` - Resulting complex roots, one fewer than `p`
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn poly_findroots_bairstow(p: &[f64], roots: &mut [Complex<f64>]) -> Result<()> {
    if p.len() < 2 {
        return Err(Error::Range("order must be greater than 0".to_owned()));
    }
    if roots.len() != p.len() - 1 {
        return Err(Error::Range("root slice length must be one less than coefficient slice length".to_owned()));
    }

    let mut p0 = p.to_vec();
    let mut p1 = vec![0.0; p.len()];
    let mut p_buf;
    let mut pr_buf = &mut p1;

    let mut n = p.len();
    let r = p.len() % 2;
    let l = (p.len() - r) / 2;
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
            poly_findroots_bairstow_persistent(&p_buf[..n], &mut pr_buf[..n - 2], &mut u, &mut v)?;
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
fn poly_findroots_bairstow_recursion(p: &[f64], p1: &mut [f64], u: &mut f64, v: &mut f64) -> Result<()> {
    let k = p.len();
    if k < 3 {
        return Err(Error::Range(format!("invalid polynomial length: {}", k)));
    }
    if p1.len() != k - 2 {
        return Err(Error::Range("reduced polynomial length must be two less than input length".to_owned()));
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
    p1.copy_from_slice(&b[..k - 2]);

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
fn poly_findroots_bairstow_persistent(p: &[f64], p1: &mut [f64], u: &mut f64, v: &mut f64) -> Result<()> {
    let num_iterations_max = 10;
    for i in 0..num_iterations_max {
        if poly_findroots_bairstow_recursion(p, p1, u, v).is_ok() {
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
    a.re.total_cmp(&b.re).then_with(|| b.im.total_cmp(&a.im))
}

/// Finds the complex roots of the polynomial
///
/// # Arguments
///
/// * `p` - Polynomial coefficients, ascending powers
/// * `roots` - Resulting complex roots, one fewer than `p`
///
/// # Returns
///
/// `Ok(())` on success, `Err(())` on failure
pub fn polyf_findroots(p: &[f32], roots: &mut [Complex<f32>]) -> Result<()> {
    if p.len() < 2 {
        return Err(Error::Range(format!("invalid polynomial length: {}", p.len())));
    }
    if roots.len() != p.len() - 1 {
        return Err(Error::Range("root slice length must be one less than coefficient slice length".to_owned()));
    }

    // Copy to temporary double-precision array
    let p_double: Vec<f64> = p.iter().map(|&x| x as f64).collect();

    // Find roots of polynomial using Bairstow's method (more
    // accurate and reliable than Durand-Kerner)
    let mut roots_double = vec![Complex::new(0.0, 0.0); roots.len()];
    poly_findroots_bairstow(&p_double, &mut roots_double)?;

    // Sort roots for consistent ordering
    roots_double.sort_by(poly_sort_roots_compare);

    // Copy back to original
    for (root, root_double) in roots.iter_mut().zip(roots_double) {
        *root = Complex::new(root_double.re as f32, root_double.im as f32);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    fn test_poly_lagrange_barycentric_recovers_cubic() {
        let polynomial = |x: f64| ((0.25 * x + 0.5) * x - 2.0) * x + 1.0;
        let x = [-1.5f64, -0.25, 0.5, 2.0];
        let y = x.map(polynomial);
        let mut weights = [0.0f64; 4];

        poly_fit_lagrange_barycentric(&x, &mut weights).unwrap();

        for x0 in [-1.0, -0.5, 0.0, 0.25, 1.0, 1.5] {
            let actual = poly_val_lagrange_barycentric(&x, &y, &weights, x0).unwrap();
            assert_abs_diff_eq!(actual, polynomial(x0), epsilon = 1e-12);
        }
        for (&x0, &expected) in x.iter().zip(&y) {
            assert_eq!(poly_val_lagrange_barycentric(&x, &y, &weights, x0), Ok(expected));
        }
    }

    #[test]
    fn test_poly_lagrange_barycentric_interpolates_between_close_nodes() {
        let x = [0.0f64, 5.0e-7];
        let y = [0.0f64, 1.0];
        let mut weights = [0.0f64; 2];

        poly_fit_lagrange_barycentric(&x, &mut weights).unwrap();

        let actual = poly_val_lagrange_barycentric(&x, &y, &weights, 2.5e-7).unwrap();
        assert_abs_diff_eq!(actual, 0.5, epsilon = 1e-12);
        assert_eq!(poly_val_lagrange_barycentric(&x, &y, &weights, x[0]), Ok(y[0]));
    }

    #[test]
    fn test_poly_val_lagrange_barycentric_rejects_invalid_values() {
        assert!(poly_val_lagrange_barycentric::<f64>(&[], &[], &[], 0.0).is_err());
        assert!(poly_val_lagrange_barycentric(&[0.0f64, 1.0], &[1.0, f64::NAN], &[-1.0, 1.0], 0.5).is_err());
        assert!(poly_val_lagrange_barycentric(&[0.0f64, 1.0], &[1.0, 2.0], &[0.0, 1.0], 0.5).is_err());
        assert!(poly_val_lagrange_barycentric(&[0.0f64, 1.0], &[1.0, 2.0], &[-1.0, 1.0], f64::NAN).is_err());
    }

    #[test]
    #[autotest_annotate(autotest_polyf_fit_q3n3)]
    fn test_polyf_fit_q3n3() {
        let x = [-1.0, 0.0, 1.0];
        let y = [1.0, 0.0, 1.0];
        let p_test = [0.0, 0.0, 1.0];
        let mut p = [0.0; 3];
        let tol = 1e-3;

        poly_fit(&x, &y, &mut p).unwrap();

        assert_abs_diff_eq!(p[0], p_test[0], epsilon = tol);
        assert_abs_diff_eq!(p[1], p_test[1], epsilon = tol);
        assert_abs_diff_eq!(p[2], p_test[2], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_lagrange_issue165)]
    fn test_polyf_lagrange_issue165() {
        let x = [-1.0, 0.0, 1.0];
        let y = [7.059105, 24.998369, 14.365907];
        let mut p = [0.0f32; 3];
        let n = 3;
        let tol = 1e-3;

        poly_fit_lagrange(&x, &y, &mut p).unwrap();

        let mut y_out = vec![0.0; n];
        for j in 0..n {
            y_out[j] = poly_val(&p, x[j]);
            assert_abs_diff_eq!(y[j], y_out[j], epsilon = tol);
        }

        poly_fit(&x, &y, &mut p).unwrap();

        for j in 0..n {
            y_out[j] = poly_val(&p, x[j]);
            assert_abs_diff_eq!(y[j], y_out[j], epsilon = tol);
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

        poly_fit_lagrange(&x, &y, &mut p).unwrap();

        let mut y_out = vec![0.0; n];
        for j in 0..n {
            y_out[j] = poly_val(&p, x[j]);
            assert_abs_diff_eq!(y[j], y_out[j], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_polyf_expandroots_4)]
    fn test_polyf_expandroots_4() {
        let roots = [-2.0, -1.0, -4.0, 5.0, 3.0];
        let c_test = [120.0, 146.0, 1.0, -27.0, -1.0, 1.0];
        let n = 5;
        let mut p = vec![0.0; n + 1];
        let tol = 1e-3;

        poly_expandroots(&roots, &mut p).unwrap();
        assert_abs_diff_eq!(p[0], c_test[0], epsilon = tol);
        assert_abs_diff_eq!(p[1], c_test[1], epsilon = tol);
        assert_abs_diff_eq!(p[2], c_test[2], epsilon = tol);
        assert_abs_diff_eq!(p[3], c_test[3], epsilon = tol);
        assert_abs_diff_eq!(p[4], c_test[4], epsilon = tol);
        assert_abs_diff_eq!(p[5], c_test[5], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_expandroots_11)]
    fn test_polyf_expandroots_11() {
        let roots = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0];
        let c_test = [
            39916800.0,
            120543840.0,
            150917976.0,
            105258076.0,
            45995730.0,
            13339535.0,
            2637558.0,
            357423.0,
            32670.0,
            1925.0,
            66.0,
            1.0,
        ];
        let n = 11;
        let mut p = vec![0.0; n + 1];
        let tol = 1e-6f32;

        poly_expandroots(&roots, &mut p).unwrap();
        assert_abs_diff_eq!(p[0], c_test[0], epsilon = (tol * c_test[0]).abs());
        assert_abs_diff_eq!(p[1], c_test[1], epsilon = (tol * c_test[1]).abs());
        assert_abs_diff_eq!(p[2], c_test[2], epsilon = (tol * c_test[2]).abs());
        assert_abs_diff_eq!(p[3], c_test[3], epsilon = (tol * c_test[3]).abs());
        assert_abs_diff_eq!(p[4], c_test[4], epsilon = (tol * c_test[4]).abs());
        assert_abs_diff_eq!(p[5], c_test[5], epsilon = (tol * c_test[5]).abs());
        assert_abs_diff_eq!(p[6], c_test[6], epsilon = (tol * c_test[6]).abs());
        assert_abs_diff_eq!(p[7], c_test[7], epsilon = (tol * c_test[7]).abs());
        assert_abs_diff_eq!(p[8], c_test[8], epsilon = (tol * c_test[8]).abs());
        assert_abs_diff_eq!(p[9], c_test[9], epsilon = (tol * c_test[9]).abs());
        assert_abs_diff_eq!(p[10], c_test[10], epsilon = (tol * c_test[10]).abs());
        assert_abs_diff_eq!(p[11], c_test[11], epsilon = (tol * c_test[11]).abs());
    }

    #[test]
    #[autotest_annotate(autotest_polycf_expandroots_4)]
    fn test_polycf_expandroots_4() {
        use num_complex::Complex32;

        let theta = 1.7f32;
        let a = [-Complex32::new(0.0, theta).exp(), -Complex32::new(0.0, -theta).exp()];
        let mut c = vec![Complex32::new(0.0, 0.0); 3];
        let c_test = [Complex32::new(1.0, 0.0), Complex32::new(2.0 * theta.cos(), 0.0), Complex32::new(1.0, 0.0)];
        let tol = 1e-3f32;

        poly_expandroots(&a, &mut c).unwrap();

        assert_abs_diff_eq!(c[0].re, c_test[0].re, epsilon = tol);
        assert_abs_diff_eq!(c[0].im, c_test[0].im, epsilon = tol);

        assert_abs_diff_eq!(c[1].re, c_test[1].re, epsilon = tol);
        assert_abs_diff_eq!(c[1].im, c_test[1].im, epsilon = tol);

        assert_abs_diff_eq!(c[2].re, c_test[2].re, epsilon = tol);
        assert_abs_diff_eq!(c[2].im, c_test[2].im, epsilon = tol);
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

        poly_expandroots2(&a, &b, &mut c).unwrap();

        assert_abs_diff_eq!(c[0], c_test[0], epsilon = tol);
        assert_abs_diff_eq!(c[1], c_test[1], epsilon = tol);
        assert_abs_diff_eq!(c[2], c_test[2], epsilon = tol);
        assert_abs_diff_eq!(c[3], c_test[3], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_polyf_mul_2_3)]
    fn test_polyf_mul_2_3() {
        let a = [2.0, -4.0, 3.0];
        let b = [-9.0, 3.0, -2.0, 5.0];
        let mut c = vec![0.0f32; 6];
        let c_test = [-18.0, 42.0, -43.0, 27.0, -26.0, 15.0];
        let tol = 1e-3f32;

        poly_mul(&a, &b, &mut c).unwrap();

        assert_abs_diff_eq!(c[0], c_test[0], epsilon = tol);
        assert_abs_diff_eq!(c[1], c_test[1], epsilon = tol);
        assert_abs_diff_eq!(c[2], c_test[2], epsilon = tol);
        assert_abs_diff_eq!(c[3], c_test[3], epsilon = tol);
        assert_abs_diff_eq!(c[4], c_test[4], epsilon = tol);
        assert_abs_diff_eq!(c[5], c_test[5], epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_poly_expandbinomial_n6)]
    fn test_poly_expandbinomial_n6() {
        let n = 6;
        let mut c = vec![0.0f32; n + 1];
        let c_test = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0];

        poly_expandbinomial(n, &mut c).unwrap();

        assert_eq!(c, c_test);
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

        poly_expandbinomial_pm(m, k, &mut c).unwrap();

        assert_abs_diff_eq!(c[0], c_test[0], epsilon = tol);
        assert_abs_diff_eq!(c[1], c_test[1], epsilon = tol);
        assert_abs_diff_eq!(c[2], c_test[2], epsilon = tol);
        assert_abs_diff_eq!(c[3], c_test[3], epsilon = tol);
        assert_abs_diff_eq!(c[4], c_test[4], epsilon = tol);
        assert_abs_diff_eq!(c[5], c_test[5], epsilon = tol);
        assert_abs_diff_eq!(c[6], c_test[6], epsilon = tol);
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

        poly_expandbinomial_pm(m, k, &mut c).unwrap();

        assert_abs_diff_eq!(c[0], c_test[0], epsilon = tol);
        assert_abs_diff_eq!(c[1], c_test[1], epsilon = tol);
        assert_abs_diff_eq!(c[2], c_test[2], epsilon = tol);
        assert_abs_diff_eq!(c[3], c_test[3], epsilon = tol);
        assert_abs_diff_eq!(c[4], c_test[4], epsilon = tol);
        assert_abs_diff_eq!(c[5], c_test[5], epsilon = tol);
        assert_abs_diff_eq!(c[6], c_test[6], epsilon = tol);
        assert_abs_diff_eq!(c[7], c_test[7], epsilon = tol);
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
        assert!(polyf_findroots(p, &mut roots).is_ok());

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
        let r = [Complex::new(-1.0, std::f32::consts::SQRT_2), Complex::new(-1.0, -std::f32::consts::SQRT_2)];
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
        let x = 3.0;
        let y = poly_val(&p, x);
        assert_abs_diff_eq!(y, 97.0, epsilon = 1e-6);
    }

    #[test]
    fn test_poly_mul_evaluation_identity() {
        let left = [2.0f32, -1.0, 0.5];
        let right = [-3.0f32, 4.0];
        let mut product = [0.0f32; 4];
        poly_mul(&left, &right, &mut product).unwrap();

        let eval = |coefficients: &[f32], x: f32| {
            coefficients.iter().rev().fold(0.0, |value, &coefficient| value * x + coefficient)
        };

        for x in [-2.0f32, -0.25, 0.0, 0.75, 2.0] {
            assert_abs_diff_eq!(eval(&product, x), eval(&left, x) * eval(&right, x), epsilon = 1e-5);
        }
    }

    #[test]
    fn test_poly_expandroots_vanishes_at_roots() {
        let roots = [-2.0f32, 0.5, 3.0];
        let mut coefficients = [0.0f32; 4];
        poly_expandroots(&roots, &mut coefficients).unwrap();

        for root in roots {
            let value = coefficients.iter().rev().fold(0.0, |value, &coefficient| value * root + coefficient);
            assert_abs_diff_eq!(value, 0.0, epsilon = 1e-5);
        }
        assert_eq!(coefficients[roots.len()], 1.0);
    }

    #[test]
    fn test_poly_fit_overdetermined_residual() {
        let x = [-3.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let y = [15.5f32, 10.0, 5.5, 2.0, -0.5, -2.0, -2.5];
        let expected = [2.0f32, -3.0, 0.5];
        let mut coefficients = [0.0f32; 3];

        poly_fit(&x, &y, &mut coefficients).unwrap();

        for (&actual, &expected) in coefficients.iter().zip(&expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-4);
        }
        for (&x, &expected) in x.iter().zip(&y) {
            assert_abs_diff_eq!(poly_val(&coefficients, x), expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_poly_fit_workspace_reuses_independent_variable_samples() {
        let x = [-2.0f64, -1.0, 0.0, 1.0, 2.0];
        let mut workspace = vandermonde_workspace(&x, 3).unwrap();
        let mut actual = [0.0; 3];

        for expected in [[1.0, -0.5, 0.25], [-2.0, 3.0, -1.0]] {
            let y = x.map(|sample| poly_val(&expected, sample));
            workspace.solve(&y, &mut actual).unwrap();
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_poly_fit_recovers_high_order_coefficients() {
        let expected: Vec<f64> =
            (0..12).map(|i| if i % 2 == 0 { 1.0 / (i + 1) as f64 } else { -1.0 / (i + 1) as f64 }).collect();
        let x: Vec<f32> = (0..48).map(|i| -1.0 + 2.0 * i as f32 / 47.0).collect();
        let y: Vec<f32> = x
            .iter()
            .map(|&sample| {
                expected.iter().rev().fold(0.0, |value, &coefficient| value * sample as f64 + coefficient) as f32
            })
            .collect();
        let mut coefficients = vec![0.0f32; expected.len()];

        poly_fit(&x, &y, &mut coefficients).unwrap();

        for (&actual, &expected) in coefficients.iter().zip(&expected) {
            assert_abs_diff_eq!(actual as f64, expected, epsilon = 2e-4);
        }
    }

    #[test]
    fn test_poly_fit_clustered_f64_samples() {
        let expected = [0.25f64, -0.5, 0.75, -0.2, 0.1, -0.05, 0.025, -0.0125];
        let x: Vec<f64> = (0..32).map(|i| 0.75 + 0.5 * i as f64 / 31.0).collect();
        let y: Vec<f64> = x.iter().map(|&sample| poly_val(&expected, sample)).collect();
        let mut coefficients = [0.0f64; 8];

        poly_fit(&x, &y, &mut coefficients).unwrap();

        for (&actual, &expected) in coefficients.iter().zip(&expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_poly_fit_complex_samples() {
        let expected = [Complex::new(0.5f64, -0.25), Complex::new(-0.75, 0.125), Complex::new(0.2, 0.4)];
        let x = [
            Complex::new(-1.0, 0.25),
            Complex::new(-0.5, -0.75),
            Complex::new(0.0, 0.5),
            Complex::new(0.5, -0.25),
            Complex::new(1.0, 0.75),
        ];
        let y = x.map(|sample| poly_val(&expected, sample));
        let mut coefficients = [Complex::new(0.0, 0.0); 3];

        poly_fit(&x, &y, &mut coefficients).unwrap();

        for (&actual, &expected) in coefficients.iter().zip(&expected) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_polyf_findroots_rejects_constant_polynomial() {
        let coefficients = [3.0f32];
        let mut roots = [Complex::new(42.0f32, -7.0)];

        let result = polyf_findroots(&coefficients, &mut roots);

        assert!(result.is_err());
        assert_eq!(roots, [Complex::new(42.0, -7.0)]);
    }

    #[test]
    fn test_poly_fit_rejects_mismatched_legacy_input_lengths() {
        let x = [0.0f32, 1.0];
        let y = [0.0f32];
        let mut coefficients = [42.0f32, -7.0];

        let result = poly_fit(&x, &y, &mut coefficients);

        assert!(result.is_err());
        assert_eq!(coefficients, [42.0, -7.0]);
    }

    #[test]
    fn test_poly_fit_rejects_empty_coefficients_and_underdetermined_samples() {
        assert!(poly_fit::<f32>(&[], &[], &mut []).is_err());

        let mut coefficients = [42.0f32, -7.0, 3.0];
        let result = poly_fit(&[0.0, 1.0], &[1.0, 2.0], &mut coefficients);

        assert!(result.is_err());
        assert_eq!(coefficients, [42.0, -7.0, 3.0]);
    }

    #[test]
    fn test_poly_fit_rejects_nonfinite_and_rank_deficient_inputs() {
        let original = [42.0f32, -7.0, 3.0];

        for (x, y) in [
            ([0.0, f32::INFINITY, 2.0], [1.0, 2.0, 3.0]),
            ([0.0, 1.0, 2.0], [1.0, f32::NAN, 3.0]),
            ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ] {
            let mut coefficients = original;
            let result = poly_fit(&x, &y, &mut coefficients);

            assert!(result.is_err());
            assert_eq!(coefficients, original);
        }
    }

    #[test]
    fn test_poly_findroots_bairstow_rejects_short_output() {
        let coefficients = [1.0f64, 0.0, 1.0];
        let mut roots = [Complex::new(42.0f64, -7.0)];

        let result = poly_findroots_bairstow(&coefficients, &mut roots);

        assert!(result.is_err());
        assert_eq!(roots, [Complex::new(42.0, -7.0)]);
    }

    #[test]
    fn test_polynomial_entry_points_handle_empty_or_short_slices() {
        assert_eq!(poly_val::<f32>(&[], 3.0), 0.0);
        assert_eq!(poly_interp_lagrange::<f32>(&[], &[], 3.0), Ok(0.0));

        let mut coefficients = [42.0f32; 2];
        assert!(poly_mul(&[1.0, 2.0], &[3.0, 4.0], &mut coefficients).is_err());
        assert_eq!(coefficients, [42.0, 42.0]);

        let mut roots = [42.0f32];
        poly_expandroots(&[], &mut roots).unwrap();
        assert_eq!(roots, [0.0]);

        poly_fit_lagrange::<f32>(&[], &[], &mut []).unwrap();
        assert!(poly_fit_lagrange_barycentric::<f32>(&[], &mut []).is_err());
    }

    #[test]
    fn test_fallible_polynomial_entry_points_reject_mismatched_slices() {
        assert!(poly_interp_lagrange(&[0.0f32], &[], 0.0).is_err());
        assert!(poly_val_lagrange_barycentric(&[0.0f32], &[1.0], &[], 0.0).is_err());

        let mut coefficients = [42.0f32; 2];
        assert!(poly_expandbinomial::<f32>(0, &mut []).is_err());
        assert!(poly_expandbinomial_pm(1, 1, &mut coefficients).is_err());
        assert!(poly_expandroots(&[1.0f32], &mut coefficients[..1]).is_err());
        assert!(poly_expandroots2(&[1.0f32], &[], &mut coefficients).is_err());
        assert!(poly_fit_lagrange(&[0.0f32], &[1.0], &mut []).is_err());
        assert!(poly_fit_lagrange_barycentric(&[0.0f32], &mut []).is_err());
        assert_eq!(coefficients, [42.0, 42.0]);
    }

    #[test]
    fn test_poly_findroots_durandkerner_finds_complex_roots() {
        let coefficients = [1.0f64, 0.0, 1.0];
        let mut roots = [Complex::new(0.0f64, 0.0); 2];

        poly_findroots_durandkerner(&coefficients, &mut roots).unwrap();

        for root in roots {
            let residual = root * root + Complex::new(1.0, 0.0);
            assert!(residual.norm() < 1e-5, "root {root:?} has residual {residual:?}");
        }
    }

    #[test]
    fn test_poly_findroots_durandkerner_rejects_short_slices() {
        let coefficients = [1.0f64, 0.0];
        let mut roots = [Complex::new(42.0f64, -7.0)];

        assert!(poly_findroots_durandkerner(&coefficients, &mut roots).is_err());
        assert_eq!(roots, [Complex::new(42.0, -7.0)]);
        assert!(poly_findroots_durandkerner(&[1.0, 0.0, 1.0], &mut []).is_err());
    }

    #[test]
    fn test_poly_sort_roots_compare_is_a_total_order() {
        let root = Complex::new(1.0f64, -2.0);
        assert_eq!(poly_sort_roots_compare(&root, &root), Ordering::Equal);

        let mut roots =
            [Complex::new(2.0, 0.0), Complex::new(1.0, -1.0), Complex::new(1.0, 3.0), Complex::new(-1.0, 0.0)];
        roots.sort_by(poly_sort_roots_compare);

        assert_eq!(
            roots,
            [Complex::new(-1.0, 0.0), Complex::new(1.0, 3.0), Complex::new(1.0, -1.0), Complex::new(2.0, 0.0),]
        );
    }
}
