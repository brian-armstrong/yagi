//! Special functions, number theory, polynomials, and windows.
//!
//! The special functions work in `f32` unless their name ends in `d`. They use
//! short fixed-length series and simple approximations, and several lose
//! accuracy or overflow at moderate arguments, as their docs say. Use a
//! dedicated special-functions library where accuracy over a wide range
//! matters.
//!
//! Polynomial coefficients are in ascending powers, so `[c0, c1, c2]` is
//! `c0 + c1*x + c2*x^2`. A slice of length `n + 1` has order `n`, even when
//! its last coefficient is zero.
//!
//! The number-theory functions work on `u32` by trial division and repeated
//! multiplication. Most run in time that grows with the value of the input,
//! so they suit small numbers.
//!
//! Most window functions return one sample at a time, from a zero-based
//! index and the window length. The windows are symmetric, with matching
//! samples at opposite ends, as used for filter design. They are not the
//! periodic windows often used for spectral analysis.

mod bessel;
mod complex_tests;
mod gamma;
mod modular_arithmetic;
mod poly;
mod windows;

pub use bessel::{besseli0f, besselif, besselj0f, besseljf, lnbesselif};
pub use gamma::{factorialf, gammaf, lngammaf, lnlowergammaf, lnuppergammaf, lowergammaf, uppergammaf};
pub use modular_arithmetic::{factor, gcd, is_prime, modpow, primitive_root_prime, totient, unique_factor};
pub(crate) use poly::vandermonde_workspace;
pub use poly::{
    poly_expandbinomial, poly_expandbinomial_pm, poly_expandroots, poly_expandroots2, poly_findroots_bairstow,
    poly_findroots_durandkerner, poly_fit, poly_fit_lagrange, poly_fit_lagrange_barycentric, poly_interp_lagrange,
    poly_mul, poly_val, poly_val_lagrange_barycentric, polyf_findroots,
};
pub use windows::{
    blackman_harris4, blackman_harris7, flat_top, get_window_type, hamming, hann, kaiser, kbd, kbd_window,
    print_windows, rcos_taper, triangular, window, WindowType,
};

use libm::erfcf;
use std::f32::consts::{PI, SQRT_2};

use crate::error::{Error, Result};

/// Returns the standard normal upper-tail probability `P(Z > z)`.
pub fn qf(z: f32) -> f32 {
    0.5 * erfcf(z / SQRT_2)
}

/// Returns a rough estimate of the order-`m` Marcum Q function.
///
/// `m` is the positive integer order, and `alpha` and `beta` are
/// non-negative. For order one, [`marcumq1f`] evaluates the function
/// directly.
///
/// # Warning
///
/// The result is not a usable Marcum Q probability, even for ordinary
/// arguments. It can exceed one. For example, `marcumqf(1, 0.0, 0.0)` returns
/// about `1.84` instead of `1`. Do not use it for probability calculations.
pub fn marcumqf(m: i32, alpha: f32, beta: f32) -> f32 {
    // Use approximation [Helstrom:1992] (Eq. 25)
    // Q_M(a,b) ~ erfc(x),
    //   x = (b-a-M)/sigma^2,
    //   sigma = M + 2a

    let sigma = m as f32 + 2.0 * alpha;
    let x = (beta - alpha - m as f32) / (sigma * sigma);
    erfcf(x)
}

/// Returns the order-one Marcum Q function `Q_1(alpha, beta)`.
///
/// `alpha` and `beta` must be non-negative. A `beta` of zero returns one.
///
/// # Warning
///
/// The 64-term `f32` series returns infinity or NaN for many ordinary
/// arguments. It fails when `alpha` is more than about four times `beta`,
/// because the power `(alpha / beta)^k` overflows. It also fails once
/// `alpha * beta` reaches about 90, where the Bessel terms overflow. For
/// example, `marcumq1f(9.0, 10.0)` returns infinity.
pub fn marcumq1f(alpha: f32, beta: f32) -> f32 {
    const NUM_MARCUMQ1_ITERATIONS: usize = 64;

    if beta == 0.0 {
        return 1.0;
    }

    let t0 = (-0.5 * (alpha * alpha + beta * beta)).exp();
    let mut t1 = 1.0;
    let a_div_b = alpha / beta;
    let a_mul_b = alpha * beta;

    let mut y = 0.0;
    for k in 0..NUM_MARCUMQ1_ITERATIONS {
        // accumulate y
        y += t1 * besselif(k as f32, a_mul_b);

        // update t1
        t1 *= a_div_b;
    }

    t0 * y
}

/// Returns the normalized sinc function `sin(pi * x) / (pi * x)`.
///
/// The value at zero is one. Where `abs(x) < 0.01`, a product of three
/// cosines stands in for the ratio. It runs slightly high, by up to about
/// `3e-6` relative at the edge of that range.
pub fn sincf(x: f32) -> f32 {
    if x.abs() < 0.01 {
        (PI * x / 2.0).cos() * (PI * x / 4.0).cos() * (PI * x / 8.0).cos()
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// Returns the normalized sinc function `sin(pi * x) / (pi * x)` in `f64`.
///
/// The value at zero is one. Where `abs(x) < 0.01`, a product of three
/// cosines stands in for the ratio. It runs slightly high, by up to about
/// `3e-6` relative at the edge of that range, which is far above `f64`
/// rounding.
pub fn sincd(x: f64) -> f64 {
    if x.abs() < 0.01 {
        (std::f64::consts::PI * x / 2.0).cos()
            * (std::f64::consts::PI * x / 4.0).cos()
            * (std::f64::consts::PI * x / 8.0).cos()
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
}

/// Returns the smallest exponent `p` for which `2^p >= x`.
///
/// For example, `nextpow2(9)` returns `4`.
///
/// # Errors
///
/// Returns an error when `x` is zero.
pub fn nextpow2(mut x: u32) -> Result<u32> {
    if x == 0 {
        return Err(Error::Value("nextpow2(), input must be greater than zero".to_owned()));
    }

    x -= 1;
    let mut n = 0;
    while x > 0 {
        x >>= 1;
        n += 1;
    }
    Ok(n)
}

/// Returns the binomial coefficient `n choose k`.
///
/// For `n` up to 12, the result is exact. Larger `n` uses an `f64`
/// multiplicative product for small `k` and an `f64` log-gamma difference
/// otherwise, rounded to a whole number before conversion to `f32`. A
/// coefficient above the `f32` range gives infinity. Finite results may
/// round when stored as `f32`.
///
/// # Errors
///
/// Returns an error when `k > n`.
pub fn nchoosek(n: u32, k: u32) -> Result<f32> {
    if k > n {
        return Err(Error::Value(("invalid input: k cannot exceed n").to_owned()));
    } else if k == 0 || k == n {
        return Ok(1.0);
    }

    // take advantage of symmetry and take larger value
    let k = if k < n / 2 { n - k } else { k };

    // use lngamma() function when n is large
    if n > 12 {
        let small_k = n - k;
        if small_k <= 32 {
            // For small k, the product avoids cancellation even when n is
            // near u32::MAX. These intermediate coefficients fit in f64.
            let mut value = 1.0f64;
            for i in 1..=small_k {
                value *= (k + i) as f64 / i as f64;
            }
            return Ok(value.round() as f32);
        }

        // The large log-gamma terms nearly cancel. Evaluate the static
        // coefficient in f64 so their difference retains useful precision.
        let t0 = libm::lgamma(n as f64 + 1.0);
        let t1 = libm::lgamma((n - k) as f64 + 1.0);
        let t2 = libm::lgamma(k as f64 + 1.0);

        return Ok((t0 - t1 - t2).exp().round() as f32);
    }

    // old method
    let mut rnum = 1.0;
    let mut rden = 1.0;
    for i in (k + 1)..=n {
        rnum *= i as f32;
    }
    for i in 1..=(n - k) {
        rden *= i as f32;
    }
    Ok(rnum / rden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_Q)]
    fn test_q() {
        let tol = 1e-6f32;
        assert_abs_diff_eq!(qf(-4.0), 0.999968329, epsilon = tol);
        assert_abs_diff_eq!(qf(-3.0), 0.998650102, epsilon = tol);
        assert_abs_diff_eq!(qf(-2.0), 0.977249868, epsilon = tol);
        assert_abs_diff_eq!(qf(-1.0), 0.841344746, epsilon = tol);
        assert_abs_diff_eq!(qf(0.0), 0.5, epsilon = tol);
        assert_abs_diff_eq!(qf(1.0), 0.158655254, epsilon = tol);
        assert_abs_diff_eq!(qf(2.0), 0.022750132, epsilon = tol);
        assert_abs_diff_eq!(qf(3.0), 0.001349898, epsilon = tol);
        assert_abs_diff_eq!(qf(4.0), 0.000031671, epsilon = tol);
    }

    #[test]
    fn test_q_positive_tail() {
        assert_abs_diff_eq!(qf(10.0), 7.619853e-24, epsilon = 1e-29);
    }

    #[test]
    #[autotest_annotate(autotest_sincf)]
    fn test_sincf() {
        let tol = 1e-3f32;
        assert_abs_diff_eq!(sincf(0.0), 1.0, epsilon = tol);
    }

    #[test]
    fn test_marcumq1f_zero_beta() {
        assert_eq!(marcumq1f(0.0, 0.0), 1.0);
        assert_eq!(marcumq1f(3.0, 0.0), 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_nextpow2)]
    fn test_nextpow2() {
        assert_eq!(nextpow2(1).unwrap(), 0);
        assert_eq!(nextpow2(2).unwrap(), 1);
        assert_eq!(nextpow2(3).unwrap(), 2);
        assert_eq!(nextpow2(4).unwrap(), 2);
        assert_eq!(nextpow2(5).unwrap(), 3);
        assert_eq!(nextpow2(6).unwrap(), 3);
        assert_eq!(nextpow2(7).unwrap(), 3);
        assert_eq!(nextpow2(8).unwrap(), 3);
        assert_eq!(nextpow2(9).unwrap(), 4);
        assert_eq!(nextpow2(10).unwrap(), 4);
        assert_eq!(nextpow2(11).unwrap(), 4);
        assert_eq!(nextpow2(12).unwrap(), 4);
        assert_eq!(nextpow2(13).unwrap(), 4);
        assert_eq!(nextpow2(14).unwrap(), 4);
        assert_eq!(nextpow2(15).unwrap(), 4);
        assert_eq!(nextpow2(67).unwrap(), 7);
        assert_eq!(nextpow2(179).unwrap(), 8);
        assert_eq!(nextpow2(888).unwrap(), 10);
    }

    #[test]
    #[autotest_annotate(autotest_nchoosek)]
    fn test_nchoosek() {
        const EPSILON: f32 = 1e-3;
        let test_vectors = [
            (6, 0, 1),
            (6, 1, 6),
            (6, 2, 15),
            (6, 3, 20),
            (6, 4, 15),
            (6, 5, 6),
            (6, 6, 1),
            (7, 0, 1),
            (7, 1, 7),
            (7, 2, 21),
            (7, 3, 35),
            (7, 4, 35),
            (7, 5, 21),
            (7, 6, 7),
            (7, 7, 1),
        ];

        for &(n, k, expected) in &test_vectors {
            assert_abs_diff_eq!(nchoosek(n, k).unwrap(), expected as f32, epsilon = EPSILON);
        }

        // test very large numbers
        assert_abs_diff_eq!(nchoosek(124, 5).unwrap(), 225150024.0, epsilon = 5000.0);
    }

    #[test]
    fn test_nchoosek_large_n() {
        assert_eq!(nchoosek(1_000_000, 1).unwrap(), 1_000_000.0);
        assert_eq!(nchoosek(u32::MAX, 1).unwrap(), u32::MAX as f32);
        assert_eq!(nchoosek(u32::MAX, 100).unwrap(), f32::INFINITY);
    }

    #[test]
    #[autotest_annotate(autotest_math_config)]
    fn test_math_config() {
        assert!(nextpow2(0).is_err());
        assert!(nchoosek(4, 5).is_err());
        // assert!(std::panic::catch_unwind(|| lngammaf(-1.0)).is_err());
        assert!(gcd(12, 0).is_err());
        assert!(gcd(0, 12).is_err());
        assert!(gcd(0, 0).is_err());
    }
}
