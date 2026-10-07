use crate::error::{Error, Result};
use crate::random::randf;

/// Returns a Weibull random sample.
///
/// `alpha` is the shape, `beta` is the scale and `gamma` is the location, the
/// lower edge of the support.
///
/// # Errors
///
/// Returns an error when a parameter is nonfinite, or `alpha` or `beta` is
/// zero or negative.
pub fn randweibf(alpha: f32, beta: f32, gamma: f32) -> Result<f32> {
    // validate input
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if !beta.is_finite() || beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }
    if !gamma.is_finite() {
        return Err(Error::Config("gamma must be finite".to_string()));
    }

    let u = loop {
        let u = randf();
        if u != 0.0 {
            break u;
        }
    };

    Ok(gamma + beta * (-u.ln()).powf(1.0 / alpha))
}

/// Returns the probability density of a Weibull distribution.
///
/// The parameters are as for [`randweibf`]. The density is zero for `x`
/// below `gamma`.
///
/// # Errors
///
/// Returns an error when a parameter is nonfinite, or `alpha` or `beta` is
/// zero or negative.
pub fn randweibf_pdf(x: f32, alpha: f32, beta: f32, gamma: f32) -> Result<f32> {
    // validate input
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if !beta.is_finite() || beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }
    if !gamma.is_finite() {
        return Err(Error::Config("gamma must be finite".to_string()));
    }

    if x < gamma {
        return Ok(0.0);
    }

    let t = x - gamma;
    Ok((alpha / beta) * (t / beta).powf(alpha - 1.0) * (-((t / beta).powf(alpha))).exp())
}

/// Returns the cumulative distribution of a Weibull distribution.
///
/// The parameters are as for [`randweibf`]. The result is zero for `x` at
/// or below `gamma`.
///
/// # Errors
///
/// Returns an error when a parameter is nonfinite, or `alpha` or `beta` is
/// zero or negative.
pub fn randweibf_cdf(x: f32, alpha: f32, beta: f32, gamma: f32) -> Result<f32> {
    // validate input
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if !beta.is_finite() || beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }
    if !gamma.is_finite() {
        return Err(Error::Config("gamma must be finite".to_string()));
    }

    if x <= gamma {
        return Ok(0.0);
    }

    Ok(1.0 - (-((x - gamma) / beta).powf(alpha)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_randweibf)]
    fn test_randweibf() {
        const NUM_TRIALS: usize = 250_000;
        const TOL: f32 = 0.15;

        let alpha: f32 = 1.0;
        let beta: f32 = 2.0;
        let gamma: f32 = 6.0;

        let mut m1: f32 = 0.0;
        let mut m2: f32 = 0.0;

        for _ in 0..NUM_TRIALS {
            let x = randweibf(alpha, beta, gamma).unwrap();
            m1 += x;
            m2 += x * x;
        }

        m1 /= NUM_TRIALS as f32;
        m2 = (m2 / NUM_TRIALS as f32) - m1 * m1;

        // compute expected moments (closed-form solution)
        let t0 = crate::math::gammaf(1.0 + 1.0 / alpha);
        let t1 = crate::math::gammaf(1.0 + 2.0 / alpha);
        let m1_exp = beta * t0 + gamma;
        let m2_exp = beta * beta * (t1 - t0 * t0);

        println!("m1 = {:.6} (expected {:.6})", m1, m1_exp);
        println!("m2 = {:.6} (expected {:.6})", m2, m2_exp);

        assert_abs_diff_eq!(m1, m1_exp, epsilon = TOL);
        assert_abs_diff_eq!(m2, m2_exp, epsilon = TOL);
    }
}
