use crate::error::{Error, Result};
use crate::math::sincd;
use std::f64::consts::PI;

/// Designs a raised-cosine Nyquist pulse.
///
/// The pulse has `k` samples per symbol and spans `m` symbols on each side of
/// its center tap, for `2 * k * m + 1` taps. `beta` is the rolloff factor.
/// `dt` shifts the pulse in samples, and a positive value moves it earlier.
/// The DC gain is about `k`, and the center tap is one when `dt` is zero.
///
/// `dt` is not range-checked.
///
/// # Errors
///
/// Returns an error when `k` or `m` is zero, or `beta` is nonfinite or outside
/// `0.0..=1.0`.
///
/// # Panics
///
/// Panics when `2 * k * m + 1` overflows `usize` and overflow checks are
/// enabled. Without overflow checks the count wraps.
pub fn fir_design_rcos(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    if k < 1 {
        return Err(Error::Config("k must be greater than 0".into()));
    }
    if m < 1 {
        return Err(Error::Config("m must be greater than 0".into()));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(Error::Config("beta must be in [0,1]".into()));
    }

    let h_len = 2 * k * m + 1;
    let mut h = vec![0.0; h_len];

    // Calculate filter coefficients
    for (n, hn) in h.iter_mut().enumerate() {
        let nf = n as f32;
        let kf = k as f32;
        let mf = m as f32;

        let z = (nf + dt) / kf - mf;

        // promote to f64 for extra precision here (sinc and cos are sensitive)
        let z = z as f64;
        let beta = beta as f64;

        let t1 = (beta * PI * z).cos();
        let t2 = sincd(z);
        let t3 = 1.0 - 4.0 * beta * beta * z * z;

        // check for special condition where 4*beta^2*z^2 equals 1
        *hn = if t3.abs() < 1e-3 {
            ((PI / (2.0 * beta)).sin() * beta * 0.5) as f32
        } else {
            (t1 * t2 / t3) as f32
        };
    }

    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    fn test_rcos_rolloff_range() {
        let above_one = f32::from_bits(1.0f32.to_bits() + 1);
        for beta in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -f32::EPSILON, above_one] {
            assert!(matches!(fir_design_rcos(4, 3, beta, 0.0), Err(Error::Config(_))));
        }
        let pi = std::f32::consts::PI;
        // Samples at t = 0, 1/4, 1/2 and 1, including the unit-rolloff limit.
        for (beta, expected) in
            [(0.0, [1.0, 2.0 * 2.0f32.sqrt() / pi, 2.0 / pi, 0.0]), (1.0, [1.0, 8.0 / (3.0 * pi), 0.5, 0.0])]
        {
            let h = fir_design_rcos(4, 3, beta, 0.0).unwrap();
            for (index, value) in [12, 13, 14, 16].into_iter().zip(expected) {
                assert_abs_diff_eq!(h[index], value, epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_liquid_firdes_rcos)]
    fn test_liquid_firdes_rcos() {
        // Initialize variables
        let k = 2;
        let m = 3;
        let beta = 0.3;
        let offset = 0.0;

        // Initialize pre-determined coefficient array
        // nb the last initializer was missing from the autotest
        #[rustfmt::skip]
        let h0: [f32; 13] = [
             1.65502646542134e-17,
             7.20253052925685e-02,
            -1.26653717080575e-16,
            -1.74718023726940e-01,
             2.95450626814946e-16,
             6.23332275392119e-01,
             1.00000000000000e+00,
             6.23332275392119e-01,
            -2.23850244261176e-16,
            -1.74718023726940e-01,
            -2.73763990895627e-17,
             7.20253052925685e-02,
             0.00000000000000e+00,
        ];

        // Create filter
        let h = fir_design_rcos(k, m, beta, offset).unwrap();

        assert_eq!(h.len(), h0.len());

        // Ensure data are equal
        for i in 0..13 {
            assert_abs_diff_eq!(h[i], h0[i], epsilon = 0.00001);
        }
    }
}
