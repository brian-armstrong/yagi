use crate::error::{Error, Result};
use crate::math::{lngammaf, lnlowergammaf};
use crate::random::randgammaf;

/// Returns a Nakagami-m random sample.
///
/// `m` is the shape and `omega` is the spread, the mean-square value of the
/// distribution. The sample is the square root of a gamma sample from
/// [`randgammaf`], with the same cost and the same rare infinite result. A NaN
/// `m` never returns.
///
/// # Errors
///
/// Returns an error when `m` is less than `0.5`, `omega` is zero or negative,
/// or `omega / m` is zero, as it is for an infinite `m`.
pub fn randnakmf(m: f32, omega: f32) -> Result<f32> {
    // validate input
    if m < 0.5 {
        return Err(Error::Config("m cannot be less than 0.5".to_string()));
    } else if omega <= 0.0 {
        return Err(Error::Config("omega must be greater than zero".to_string()));
    }

    // generate Gamma random variable
    let alpha = m;
    let beta = omega / m;
    let x = randgammaf(alpha, beta)?;

    // sqrt(x) ~ Nakagami(m,omega)
    Ok(x.sqrt())
}

/// Returns the probability density of a Nakagami-m distribution.
///
/// The parameters are as for [`randnakmf`]. The density is zero for `x` at or
/// below zero, including at zero itself where the true density is positive
/// for `m` of `0.5`.
///
/// # Errors
///
/// Returns an error when `m` is less than `0.5` or `omega` is zero or
/// negative.
pub fn randnakmf_pdf(x: f32, m: f32, omega: f32) -> Result<f32> {
    // validate input
    if m < 0.5 {
        return Err(Error::Config("m cannot be less than 0.5".to_string()));
    } else if omega <= 0.0 {
        return Err(Error::Config("omega must be greater than zero".to_string()));
    }

    if x <= 0.0 {
        return Ok(0.0);
    }

    let t0 = lngammaf(m);
    let t1 = m * (m / omega).ln();
    let t2 = (2.0 * m - 1.0) * x.ln();
    let t3 = -(m / omega) * x * x;

    Ok(2.0 * (-t0 + t1 + t2 + t3).exp())
}

/// Returns the cumulative distribution of a Nakagami-m distribution.
///
/// The parameters are as for [`randnakmf`]. The result is zero for `x` at or
/// below zero. It is computed from the lower incomplete gamma function, which
/// loses accuracy for large arguments, so a result within `[0, 1]` is not
/// proof of accuracy.
///
/// # Errors
///
/// Returns an error when `m` is less than `0.5`, `omega` is zero or negative,
/// or the calculation overflows or leaves `[0, 1]`.
pub fn randnakmf_cdf(x: f32, m: f32, omega: f32) -> Result<f32> {
    // validate input
    if m < 0.5 {
        return Err(Error::Config("m cannot be less than 0.5".to_string()));
    } else if omega <= 0.0 {
        return Err(Error::Config("omega must be greater than zero".to_string()));
    }

    if x <= 0.0 {
        return Ok(0.0);
    }

    let argument = x * x * m / omega;
    if !argument.is_finite() {
        return Err(Error::Range("x^2 * m / omega is not representable".to_string()));
    }

    let cdf = (lnlowergammaf(m, argument) - lngammaf(m)).exp();
    if !cdf.is_finite() || !(0.0..=1.0).contains(&cdf) {
        return Err(Error::Range("Nakagami CDF could not be evaluated for these parameters".to_string()));
    }
    Ok(cdf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_randnakmf_cdf_rayleigh() {
        assert_abs_diff_eq!(randnakmf_cdf(1.0, 1.0, 1.0).unwrap(), 1.0 - (-1.0f32).exp(), epsilon = 1e-5);
    }
}
