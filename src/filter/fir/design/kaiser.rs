use crate::error::{Error, Result};
use crate::math::{kaiser, sincf};

/// Designs a low-pass FIR filter as a Kaiser-windowed sinc.
///
/// The filter has `n` taps and a cutoff of `fc` cycles per sample. `as_` is
/// the stopband attenuation in dB, which sets the window shape. `mu` shifts
/// the impulse response by up to one sample, and a positive value moves it
/// earlier.
///
/// The sinc peaks at one rather than `2 * fc`, so a long filter has a DC gain
/// of about `1 / (2 * fc)`. The exact gain depends on the length and window.
///
/// # Errors
///
/// Returns an error when `n` is zero, `fc` is not in `0.0 < fc <= 0.5`,
/// `as_` is nonfinite, zero or negative, or `mu` is outside `-1.0..=1.0`.
pub fn fir_design_kaiser(n: usize, fc: f32, as_: f32, mu: f32) -> Result<Vec<f32>> {
    // validate input
    if !(-1.0..=1.0).contains(&mu) {
        return Err(Error::Config(format!("fractional sample offset ({}) out of range [-1, 1]", mu)));
    }
    if !(fc > 0.0 && fc <= 0.5) {
        return Err(Error::Config(format!("cutoff frequency ({}) out of range (0, 0.5]", fc)));
    }
    if n == 0 {
        return Err(Error::Config("filter length must be greater than zero".into()));
    }
    if !as_.is_finite() || as_ <= 0.0 {
        return Err(Error::Config("stop-band attenuation must be finite and greater than zero".into()));
    }

    // compute Kaiser window paramter (beta)
    let beta = kaiser_beta_stopband_attenuation(as_);

    let mut h = vec![0.0; n];

    for (i, h_i) in h.iter_mut().enumerate() {
        // time vector
        let t = i as f32 - (n as f32 - 1.0) / 2.0 + mu;

        // sinc prototype
        let h1 = sincf(2.0 * fc * t);

        // kaiser window
        let h2 = kaiser(i, n, beta)?;

        // composite
        *h_i = h1 * h2;
    }

    Ok(h)
}

/// Returns the Kaiser window beta for a stopband attenuation in dB.
///
/// Attenuation of 21 dB or less gives zero, which is a rectangular window.
/// Negative attenuation is treated as its magnitude.
pub fn kaiser_beta_stopband_attenuation(as_: f32) -> f32 {
    // [Vaidyanathan:1993]
    let as_abs = as_.abs();
    if as_abs > 50.0 {
        0.1102 * (as_abs - 8.7)
    } else if as_abs > 21.0 {
        0.5842 * (as_abs - 21.0).powf(0.4) + 0.07886 * (as_abs - 21.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_kaiser_rejects_invalid_design_parameters() {
        let above_half = f32::from_bits(0.5f32.to_bits() + 1);
        for fc in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -f32::EPSILON, above_half] {
            assert!(matches!(fir_design_kaiser(5, fc, 60.0, 0.0), Err(Error::Config(_))));
        }
        for attenuation in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
            assert!(matches!(fir_design_kaiser(5, 0.25, attenuation, 0.0), Err(Error::Config(_))));
        }
        // Below 21 dB the window is rectangular, giving closed-form sinc taps.
        let h = fir_design_kaiser(5, 0.25, 10.0, 0.0).unwrap();
        let expected = [0.0, 2.0 / std::f32::consts::PI, 1.0, 2.0 / std::f32::consts::PI, 0.0];
        for (&actual, reference) in h.iter().zip(expected) {
            assert_abs_diff_eq!(actual, reference, epsilon = 1e-6);
        }
        let h = fir_design_kaiser(5, 0.5, 60.0, 0.0).unwrap();
        for (&actual, reference) in h.iter().zip([0.0, 0.0, 1.0, 0.0, 0.0]) {
            assert_abs_diff_eq!(actual, reference, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kaiser_single_tap_filter() {
        let h = fir_design_kaiser(1, 0.2, 60.0, 0.0).unwrap();
        assert_eq!(h, [1.0]);
    }

    #[test]
    fn test_kaiser_fractional_sample_offset_range() {
        assert!(fir_design_kaiser(41, 0.2, 60.0, -1.0).is_ok());
        assert!(fir_design_kaiser(41, 0.2, 60.0, 1.0).is_ok());
        assert!(fir_design_kaiser(41, 0.2, 60.0, -1.01).is_err());
        assert!(fir_design_kaiser(41, 0.2, 60.0, 1.01).is_err());
        assert!(fir_design_kaiser(41, 0.2, 60.0, f32::NAN).is_err());
    }

    #[test]
    fn test_kaiser_fractional_sample_offset_symmetry() {
        let hp = fir_design_kaiser(41, 0.2, 60.0, 0.75).unwrap();
        let hn = fir_design_kaiser(41, 0.2, 60.0, -0.75).unwrap();

        for (&a, &b) in hp.iter().zip(hn.iter().rev()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-6);
        }
    }
}
