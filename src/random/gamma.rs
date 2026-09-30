use crate::error::{Error, Result};
use crate::math::{gammaf, lowergammaf};
use crate::random::randf;

/// Returns a gamma random sample.
///
/// `alpha` is the shape and `beta` is the scale. The distribution has mean
/// `alpha * beta`. Each sample takes `floor(alpha)` uniform draws plus a
/// rejection loop for the fractional part of `alpha`, so the cost grows with
/// `alpha`. An infinite or NaN `alpha` never returns.
///
/// # Errors
///
/// Returns an error when `alpha` or `beta` is zero or negative.
///
/// # Warning
///
/// Some uniform draws of exactly zero are not rejected, and the sample is
/// then infinite. Each draw is zero with probability `2^-24`.
pub fn randgammaf(alpha: f32, beta: f32) -> Result<f32> {
    // validate input
    if alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    let n = alpha.floor() as usize;
    let delta = alpha - n as f32;

    // generate x' ~ Gamma(n,1)
    let mut x_n = 0.0;
    for _ in 0..n {
        let u = randf();
        x_n -= u.ln();
    }

    // generate x'' ~ Gamma(delta,1) using rejection method
    let x_delta = randgammaf_delta(delta)?;

    Ok(beta * (x_delta + x_n))
}

/// Returns the probability density of a gamma distribution.
///
/// `alpha` is the shape and `beta` is the scale. The density is zero for `x`
/// at or below zero, including at zero itself where the true density is
/// positive or unbounded for `alpha` of one or less.
///
/// # Errors
///
/// Returns an error when `alpha` or `beta` is zero or negative.
pub fn randgammaf_pdf(x: f32, alpha: f32, beta: f32) -> Result<f32> {
    // validate input
    if alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    if x <= 0.0 {
        return Ok(0.0);
    }

    let t0 = x.powf(alpha - 1.0);
    let t1 = (-x / beta).exp();
    let t2 = gammaf(alpha);
    let t3 = beta.powf(alpha);

    Ok((t0 * t1) / (t2 * t3))
}

/// Returns the cumulative distribution of a gamma distribution.
///
/// `alpha` is the shape and `beta` is the scale. The result is zero for `x`
/// at or below zero. It is computed as the lower incomplete gamma function
/// over `Gamma(alpha)`, which loses accuracy as either grows large, so a
/// result within `[0, 1]` is not proof of accuracy.
///
/// # Errors
///
/// Returns an error when `alpha` or `beta` is zero, negative or not finite,
/// when `x` is NaN or positive infinity, or when the calculation overflows or
/// leaves `[0, 1]`.
pub fn randgammaf_cdf(x: f32, alpha: f32, beta: f32) -> Result<f32> {
    // validate input
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if !beta.is_finite() || beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    if x <= 0.0 {
        return Ok(0.0);
    }
    if !x.is_finite() {
        return Err(Error::Value("x must be finite".to_string()));
    }

    let argument = x / beta;
    if !argument.is_finite() {
        return Err(Error::Range("x / beta is not representable".to_string()));
    }

    let lower = lowergammaf(alpha, argument);
    let gamma = gammaf(alpha);
    if !lower.is_finite() || !gamma.is_finite() {
        return Err(Error::Range("Gamma CDF is not representable for these parameters".to_string()));
    }

    let cdf = lower / gamma;
    if !cdf.is_finite() || !(0.0..=1.0).contains(&cdf) {
        return Err(Error::Range("Gamma CDF could not be evaluated for these parameters".to_string()));
    }
    Ok(cdf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_randgammaf_cdf_bounds() {
        assert_abs_diff_eq!(randgammaf_cdf(1.0, 2.0, 1.0).unwrap(), 1.0 - 2.0 / std::f32::consts::E, epsilon = 1e-5);
        assert!(randgammaf_cdf(50.0, 50.0, 1.0).is_err());
    }
}

// generate x ~ Gamma(delta,1)
fn randgammaf_delta(delta: f32) -> Result<f32> {
    // validate input
    if delta < 0.0 || delta >= 1.0 {
        return Err(Error::Config("delta must be in [0,1)".to_string()));
    }

    // initialization
    let delta_inv = 1.0 / delta;
    let e = std::f32::consts::E;
    let v0_ = e / (e + delta);

    loop {
        // step 2
        let v0 = randf();
        let v1 = randf();
        let v2 = randf();

        let (xi, eta) = if v2 <= v0_ {
            // step 4
            let xi = v1.powf(delta_inv);
            let eta = v0 * xi.powf(delta - 1.0);
            (xi, eta)
        } else {
            // step 5
            let xi = 1.0 - v1.ln();
            let eta = v0 * (-xi).exp();
            (xi, eta)
        };

        // step 6
        if eta <= xi.powf(delta - 1.0) * (-xi).exp() {
            return Ok(xi);
        }
    }
}
