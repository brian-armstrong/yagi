use crate::error::{Error, Result};
use rand::Rng;

/// Returns a uniform random sample in `[0, 1)`.
pub fn randf() -> f32 {
    let mut rng = rand::thread_rng();
    rng.gen::<f32>()
}

/// Returns the probability density of the uniform distribution on `[0, 1]`.
///
/// The density is one for `x` in `[0, 1]` and zero outside it.
pub fn randf_pdf(x: f32) -> f32 {
    if x < 0.0 || x > 1.0 {
        0.0
    } else {
        1.0
    }
}

/// Returns the cumulative distribution of the uniform distribution on `[0, 1]`.
///
/// The result is `x` clamped to `[0, 1]`.
pub fn randf_cdf(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Returns a uniform random sample in `[a, b)`.
///
/// Rounding can occasionally return `b` itself. Interval arithmetic uses
/// `f64` so finite `f32` endpoints can span the full range.
///
/// # Errors
///
/// Returns an error when either endpoint is nonfinite or `a` is not less than `b`.
pub fn randuf(a: f32, b: f32) -> Result<f32> {
    if !a.is_finite() || !b.is_finite() || a >= b {
        return Err(Error::Range(format!("randuf({},{}) has invalid range", a, b)));
    }

    Ok((a as f64 + (b as f64 - a as f64) * randf() as f64) as f32)
}

/// Returns the probability density of the uniform distribution on `[a, b]`.
///
/// The density is `1 / (b - a)` inside the interval and zero outside it. The
/// width is computed in `f64` to support the full finite `f32` range.
///
/// # Errors
///
/// Returns an error when either endpoint is nonfinite or `a` is not less than `b`.
pub fn randuf_pdf(x: f32, a: f32, b: f32) -> Result<f32> {
    if !a.is_finite() || !b.is_finite() || a >= b {
        return Err(Error::Range(format!("randuf_pdf({},{},{}) has invalid range", x, a, b)));
    }

    Ok(if x < a || x > b { 0.0 } else { (1.0 / (b as f64 - a as f64)) as f32 })
}

/// Returns the cumulative distribution of the uniform distribution on
/// `[a, b]`.
///
/// Interval arithmetic uses `f64` to support the full finite `f32` range.
///
/// # Errors
///
/// Returns an error when either endpoint is nonfinite or `a` is not less than `b`.
pub fn randuf_cdf(x: f32, a: f32, b: f32) -> Result<f32> {
    if !a.is_finite() || !b.is_finite() || a >= b {
        return Err(Error::Range(format!("randuf_cdf({},{},{}) has invalid range", x, a, b)));
    }

    Ok(if x < a {
        0.0
    } else if x > b {
        1.0
    } else {
        ((x as f64 - a as f64) / (b as f64 - a as f64)) as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_randf)]
    fn test_randf() {
        const NUM_TRIALS: usize = 100_000;
        const TOL: f32 = 0.1;

        let mut m1: f32 = 0.0;
        let mut m2: f32 = 0.0;

        for _ in 0..NUM_TRIALS {
            let x = randf();
            m1 += x;
            m2 += x * x;
        }

        m1 /= NUM_TRIALS as f32;
        m2 = (m2 / NUM_TRIALS as f32) - m1 * m1;

        println!("m1 = {:.6} (expected 0.5)", m1);
        println!("m2 = {:.6} (expected 1/12 = 0.0833333)", m2);

        assert_abs_diff_eq!(m1, 0.5f32, epsilon = TOL);
        assert_abs_diff_eq!(m2, 1.0f32 / 12.0f32, epsilon = TOL);
    }
}
