use crate::error::{Error, Result};
use crate::math::lnbesselif;
use crate::math::marcumq1f;
use crate::random::crandnf;
use num_complex::Complex;

/// Returns a Rice-K random sample.
///
/// `k` is the ratio of line-of-sight power to scattered power, and `omega`
/// is the mean-square value of the distribution. A `k` of zero gives a
/// Rayleigh distribution.
///
/// # Errors
///
/// Returns an error when `k` is negative or `omega` is zero or negative.
pub fn randricekf(k: f32, omega: f32) -> Result<f32> {
    // validate input
    if k < 0.0 {
        return Err(Error::Config("k must be non-negative".to_string()));
    } else if omega <= 0.0 {
        return Err(Error::Config("omega must be greater than zero".to_string()));
    }

    let s = ((omega * k) / (k + 1.0)).sqrt();
    let sig = (0.5 * omega / (k + 1.0)).sqrt();

    let x = crandnf();
    let y = Complex::new(x.re * sig + s, x.im * sig);

    Ok(y.norm())
}

/// Returns the probability density of a Rice-K distribution.
///
/// The parameters are as for [`randricekf`]. The density is zero for
/// negative `x`.
///
/// # Errors
///
/// Returns an error when `k` is negative or `omega` is zero or negative.
///
/// # Warning
///
/// The result is zero whenever `x * s / sigma^2` exceeds 80, where
/// `s = sqrt(omega * k / (k + 1))` and `sigma^2 = omega / (2 * (k + 1))`.
/// Near the peak that ratio is about `2 * (k + 1)`, so from `k` of about 39
/// the zero region covers the peak itself. With `k = 40`, `omega = 1` and
/// `x = 1`, the result is zero although the true density is about 3.6.
pub fn randricekf_pdf(x: f32, k: f32, omega: f32) -> Result<f32> {
    // validate input
    if k < 0.0 {
        return Err(Error::Config("k must be non-negative".to_string()));
    } else if omega <= 0.0 {
        return Err(Error::Config("omega must be greater than zero".to_string()));
    }

    if x < 0.0 {
        return Ok(0.0);
    }

    let s = ((omega * k) / (k + 1.0)).sqrt();
    let sig = (0.5 * omega / (k + 1.0)).sqrt();

    let t = x * x + s * s;
    let sig2 = sig * sig;

    // check high tail condition
    if (x * s / sig2) > 80.0 {
        return Ok(0.0);
    }

    let t0 = x.ln() - sig2.ln();
    let t1 = -t / (2.0 * sig2);
    let t2 = lnbesselif(0.0, x * s / sig2);

    Ok((t0 + t1 + t2).exp())
    // Ok((x / sig2) * (-t / (2.0 * sig2)).exp() * lnbesselif(0.0, x * s / sig2))
}

/// Returns the cumulative distribution of a Rice-K distribution.
///
/// The parameters are as for [`randricekf`]. The result is zero for `x` at or
/// below zero. A `k` of zero uses the closed-form Rayleigh distribution.
/// Otherwise the result comes from the Marcum Q function, and it is set to
/// exactly zero or one once `x` is a factor of three below or above the
/// line-of-sight amplitude `sqrt(omega * k / (k + 1))`.
///
/// # Errors
///
/// Returns an error when `x`, `k` or `omega` is not finite, `k` is negative,
/// or `omega` is zero or negative.
///
/// # Warning
///
/// The Marcum Q evaluation overflows at moderate `k`. The result is then a
/// wrong zero or one, or NaN, with no error. With `k = 40`, `omega = 1` and
/// `x = 1.2`, the result is zero although the true value is about 0.97. With
/// `k = 100` and `omega = 1`, it is NaN between the two cutoffs.
pub fn randricekf_cdf(x: f32, k: f32, omega: f32) -> Result<f32> {
    // validate input
    if !k.is_finite() || k < 0.0 {
        return Err(Error::Config("k must be non-negative".to_string()));
    } else if !omega.is_finite() || omega <= 0.0 {
        return Err(Error::Config("omega must be greater than zero".to_string()));
    }
    if !x.is_finite() {
        return Err(Error::Value("x must be finite".to_string()));
    }

    if x <= 0.0 {
        return Ok(0.0);
    }
    if k == 0.0 {
        return Ok(1.0 - (-(x * x) / omega).exp());
    }

    let s = ((omega * k) / (k + 1.0)).sqrt();
    let sig = (0.5 * omega / (k + 1.0)).sqrt();

    // test arguments of Q1 function
    let alpha = s / sig;
    let beta = x / sig;

    if (alpha / beta) > 3.0 {
        return Ok(0.0);
    }
    if (beta / alpha) > 3.0 {
        return Ok(1.0);
    }

    let f = 1.0 - marcumq1f(alpha, beta);

    // check for precision error
    if f < 0.0 {
        Ok(0.0)
    } else if f > 1.0 {
        Ok(1.0)
    } else {
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_randricekf)]
    fn test_randricekf() {
        const NUM_TRIALS: usize = 100_000;
        const TOL: f32 = 0.1;

        let k: f32 = 2.0;
        let omega: f32 = 1.0;

        let mut m1: f32 = 0.0;
        let mut m2: f32 = 0.0;

        for _ in 0..NUM_TRIALS {
            let x = randricekf(k, omega).unwrap();
            m1 += x;
            m2 += x * x;
        }

        m1 /= NUM_TRIALS as f32;
        m2 /= NUM_TRIALS as f32;

        // compute expected moments (closed-form solution)
        let m1_exp = 0.92749f32;
        let m2_exp = omega;

        println!("m1 = {:.6} (expected {:.6})", m1, m1_exp);
        println!("m2 = {:.6} (expected {:.6})", m2, m2_exp);

        assert_abs_diff_eq!(m1, m1_exp, epsilon = TOL);
        assert_abs_diff_eq!(m2, m2_exp, epsilon = TOL);
    }

    #[test]
    fn test_randricekf_cdf_zero_k() {
        assert_abs_diff_eq!(randricekf_cdf(1.0, 0.0, 1.0).unwrap(), 1.0 - (-1.0f32).exp(), epsilon = 1e-6);
    }

    #[test]
    fn test_randricekf_cdf_rejects_nonfinite_inputs() {
        assert!(randricekf_cdf(f32::NAN, 1.0, 1.0).is_err());
        assert!(randricekf_cdf(1.0, f32::INFINITY, 1.0).is_err());
        assert!(randricekf_cdf(1.0, 1.0, f32::INFINITY).is_err());
    }
}
