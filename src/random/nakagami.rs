use crate::error::{Error, Result};
use crate::math::gamma::{lngammaf, lnlowergammaf};
use crate::random::randgammaf;

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

/// Nakagami-m distribution probability distribution function
/// Nakagami-m
///  f(x) = (2/Gamma(m)) (m/omega)^m x^(2m-1) exp{-(m/omega)x^2}
/// where
///      m       : shape parameter, m >= 0.5
///      omega   : spread parameter, omega > 0
///      Gamma(z): regular complete gamma function
///      x >= 0
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

// Nakagami-m distribution cumulative distribution function
//  F(x) = gamma(m, x^2 m / omega) / Gamma(m)
//  where
//      gamma(z,a) = lower incomplete gamma function
//      Gamma(z)   = regular gamma function
//
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
