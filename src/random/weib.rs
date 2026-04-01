use crate::error::{Error, Result};
use crate::random::randf;

/// Weibull
pub fn randweibf(alpha: f32, beta: f32, gamma: f32) -> Result<f32> {
    // validate input
    if alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    let u = loop {
        let u = randf();
        if u != 0.0 {
            break u;
        }
    };

    Ok(gamma + beta * (-u.ln()).powf(1.0 / alpha))
}

/// Weibull random number probability distribution function
pub fn randweibf_pdf(x: f32, alpha: f32, beta: f32, gamma: f32) -> Result<f32> {
    // validate input
    if alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    if x < gamma {
        return Ok(0.0);
    }

    let t = x - gamma;
    Ok((alpha / beta) * (t / beta).powf(alpha - 1.0) * (-((t / beta).powf(alpha))).exp())
}

pub fn randweibf_cdf(x: f32, alpha: f32, beta: f32, gamma: f32) -> Result<f32> {
    // validate input
    if alpha <= 0.0 {
        return Err(Error::Config("alpha must be greater than zero".to_string()));
    } else if beta <= 0.0 {
        return Err(Error::Config("beta must be greater than zero".to_string()));
    }

    if x <= gamma {
        return Ok(0.0);
    }

    Ok(1.0 - (-((x - gamma) / beta).powf(alpha)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;

    #[test]
    #[autotest_annotate(autotest_randweibf)]
    fn test_randweibf() {
        const NUM_TRIALS: usize = 100_000;
        const TOL: f32 = 0.1;

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
        let t0 = crate::math::gamma::gammaf(1.0 + 1.0 / alpha);
        let t1 = crate::math::gamma::gammaf(1.0 + 2.0 / alpha);
        let m1_exp = beta * t0 + gamma;
        let m2_exp = beta * beta * (t1 - t0 * t0);

        println!("m1 = {:.6} (expected {:.6})", m1, m1_exp);
        println!("m2 = {:.6} (expected {:.6})", m2, m2_exp);

        assert_relative_eq!(m1, m1_exp, epsilon = TOL);
        assert_relative_eq!(m2, m2_exp, epsilon = TOL);
    }
}