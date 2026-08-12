//
// compand.rs
//
// mu-law compander: compress and expand signal dynamic range
//

use crate::error::{Error, Result};
use num_complex::Complex32;

/// mu-law compression curve, defined on magnitudes. does not validate input.
fn compress_magnitude(x: f32, mu: f32) -> f32 {
    (1.0 + mu * x).ln() / (1.0 + mu).ln()
}

/// mu-law expansion curve, the inverse of [`compress_magnitude`]
fn expand_magnitude(y: f32, mu: f32) -> f32 {
    (1.0 / mu) * ((1.0 + mu).powf(y) - 1.0)
}

/// compress a real sample using mu-law encoding
///   x  : input sample
///   mu : compression factor, mu > 0
pub fn compress_mulaw(x: f32, mu: f32) -> Result<f32> {
    if mu <= 0.0 {
        return Err(Error::Range("compress_mulaw(), mu out of range".into()));
    }
    Ok(compress_magnitude(x.abs(), mu).copysign(x))
}

/// expand a real sample using mu-law decoding
///   y  : input sample
///   mu : compression factor, mu > 0
pub fn expand_mulaw(y: f32, mu: f32) -> Result<f32> {
    if mu <= 0.0 {
        return Err(Error::Range("expand_mulaw(), mu out of range".into()));
    }
    Ok(expand_magnitude(y.abs(), mu).copysign(y))
}

/// compress a complex sample using mu-law encoding, preserving phase
///   x  : input sample
///   mu : compression factor, mu > 0
pub fn compress_cf_mulaw(x: Complex32, mu: f32) -> Result<Complex32> {
    if mu <= 0.0 {
        return Err(Error::Range("compress_cf_mulaw(), mu out of range".into()));
    }
    // compress the magnitude, keep the angle
    Ok(Complex32::from_polar(compress_magnitude(x.norm(), mu), x.arg()))
}

/// expand a complex sample using mu-law decoding, preserving phase
///   y  : input sample
///   mu : compression factor, mu > 0
pub fn expand_cf_mulaw(y: Complex32, mu: f32) -> Result<Complex32> {
    if mu <= 0.0 {
        return Err(Error::Range("expand_cf_mulaw(), mu out of range".into()));
    }
    // expand the magnitude, keep the angle
    Ok(Complex32::from_polar(expand_magnitude(y.norm(), mu), y.arg()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_compand_float)]
    fn test_compand_float() {
        let mut x = -1.0f32;
        let mu = 255.0f32;
        let n = 30;

        let dx = 2.0 / n as f32;
        let tol = 1e-6f32;

        for _ in 0..n {
            let y = compress_mulaw(x, mu).unwrap();
            let x_hat = expand_mulaw(y, mu).unwrap();

            assert_abs_diff_eq!(x, x_hat, epsilon = tol);

            x += dx;
            x = if x > 1.0 { 1.0 } else { x };
        }
    }

    #[test]
    #[autotest_annotate(autotest_compand_cfloat)]
    fn test_compand_cfloat() {
        let mut x = Complex32::new(-0.707, -0.707);
        let mu = 255.0f32;
        let n = 30;

        let dx = Complex32::new(0.707, 0.707) * 2.0 / n as f32;
        let tol = 1e-6f32;

        for _ in 0..n {
            let y = compress_cf_mulaw(x, mu).unwrap();
            let z = expand_cf_mulaw(y, mu).unwrap();

            assert_abs_diff_eq!(x.re, z.re, epsilon = tol);
            assert_abs_diff_eq!(x.im, z.im, epsilon = tol);

            x += dx;
        }
    }

    #[test]
    fn test_compand_mu_range() {
        for mu in [0.0f32, -1.0, -255.0] {
            assert!(compress_mulaw(0.5, mu).is_err(), "compress mu={}", mu);
            assert!(expand_mulaw(0.5, mu).is_err(), "expand mu={}", mu);

            let x = Complex32::new(0.5, 0.5);
            assert!(compress_cf_mulaw(x, mu).is_err(), "compress_cf mu={}", mu);
            assert!(expand_cf_mulaw(x, mu).is_err(), "expand_cf mu={}", mu);
        }
    }

    #[test]
    fn test_compand_endpoints() {
        let mu = 255.0f32;

        // the compander maps [-1,1] onto itself, fixing -1, 0, and 1
        for x in [-1.0f32, 0.0, 1.0] {
            assert_abs_diff_eq!(compress_mulaw(x, mu).unwrap(), x, epsilon = 1e-6);
            assert_abs_diff_eq!(expand_mulaw(x, mu).unwrap(), x, epsilon = 1e-6);
        }

        // zero has no phase to preserve, but must not produce NaN
        let zero = compress_cf_mulaw(Complex32::new(0.0, 0.0), mu).unwrap();
        assert_eq!(zero, Complex32::new(0.0, 0.0));
        let zero = expand_cf_mulaw(Complex32::new(0.0, 0.0), mu).unwrap();
        assert_eq!(zero, Complex32::new(0.0, 0.0));
    }

    #[test]
    fn test_compand_expands_small_signals() {
        let mu = 255.0f32;

        // compression is the point: small inputs gain dynamic range, and the
        // gain grows the smaller the input gets (1.75x at 0.5, 41x at 0.001)
        let mut prev_gain = 1.0f32;
        for x in [0.5f32, 0.1, 0.01, 0.001] {
            let y = compress_mulaw(x, mu).unwrap();
            assert!(y > x, "compress({}) = {} should exceed input", x, y);

            let gain = y / x;
            assert!(gain > prev_gain, "gain at {} should exceed the last", x);
            prev_gain = gain;
        }
    }

    #[test]
    fn test_compand_cf_preserves_phase() {
        let mu = 255.0f32;

        for k in 0..8 {
            let theta = std::f32::consts::TAU * k as f32 / 8.0;
            let x = Complex32::from_polar(0.3, theta);

            let y = compress_cf_mulaw(x, mu).unwrap();
            // magnitude is compressed but the angle is untouched
            assert!(y.norm() > x.norm());
            assert_abs_diff_eq!(y.arg(), x.arg(), epsilon = 1e-6);

            let z = expand_cf_mulaw(y, mu).unwrap();
            assert_abs_diff_eq!(z.re, x.re, epsilon = 1e-6);
            assert_abs_diff_eq!(z.im, x.im, epsilon = 1e-6);
        }
    }
}
