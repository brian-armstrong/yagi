use crate::error::{Error, Result};
use crate::random::randf;

/// Exponential
pub fn randexpf(lambda: f32) -> Result<f32> {
    // validate input
    if lambda <= 0.0 {
        return Err(Error::Range(format!("randexpf({}) has invalid range", lambda)));
    }

    // compute a non-zero uniform random variable in (0,1]
    let u = loop {
        let u = randf();
        if u != 0.0 {
            break u;
        }
    };

    // perform variable transformation
    Ok(-u.ln() / lambda)
}

/// Exponential random number probability distribution function
pub fn randexpf_pdf(x: f32, lambda: f32) -> Result<f32> {
    // validate input
    if lambda <= 0.0 {
        return Err(Error::Range(format!("randexpf_pdf({},{}) has invalid range", x, lambda)));
    }

    if x < 0.0 {
        Ok(0.0)
    } else {
        Ok(lambda * (-lambda * x).exp())
    }
}

/// Exponential random number cumulative distribution function
pub fn randexpf_cdf(x: f32, lambda: f32) -> Result<f32> {
    // validate input
    if lambda <= 0.0 {
        return Err(Error::Range(format!("randexpf_cdf({},{}) has invalid range", x, lambda)));
    }

    if x < 0.0 {
        Ok(0.0)
    } else {
        Ok(1.0 - (-lambda * x).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;

    #[test]
    #[autotest_annotate(autotest_randexpf)]
    fn test_randexpf() {
        const NUM_TRIALS: usize = 100_000;
        const TOL: f32 = 0.1;
        let lambda: f32 = 2.3f32;

        let mut m1: f32 = 0.0;
        let mut m2: f32 = 0.0;

        for _ in 0..NUM_TRIALS {
            let x = randexpf(lambda).unwrap();
            m1 += x;
            m2 += x * x;
        }

        m1 /= NUM_TRIALS as f32;
        m2 = (m2 / NUM_TRIALS as f32) - m1 * m1;

        // compute expected moments (closed-form solution)
        let m1_exp = 1.0 / lambda;
        let m2_exp = 1.0 / (lambda * lambda);

        println!("m1 = {:.6} (expected {:.6})", m1, m1_exp);
        println!("m2 = {:.6} (expected {:.6})", m2, m2_exp);

        assert_relative_eq!(m1, m1_exp, epsilon = TOL);
        assert_relative_eq!(m2, m2_exp, epsilon = TOL);
    }
}