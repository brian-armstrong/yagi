//! Phase-unwrapping helpers.

use std::f32::consts::PI;

/// Unwraps a phase sequence in place.
///
/// Each sample, from the second on, gains or loses whole turns until it lies
/// within `pi` radians of the already unwrapped sample before it. The first
/// sample is unchanged, as are empty and one-sample slices.
///
/// Values must be finite and small enough that adding `2*pi` changes their
/// `f32` value. An infinite value, or one too large for a turn to register,
/// makes the call loop forever.
pub fn unwrap_phase(theta: &mut [f32]) {
    for i in 1..theta.len() {
        while theta[i] - theta[i - 1] > PI {
            theta[i] -= 2.0 * PI;
        }
        while theta[i] - theta[i - 1] < -PI {
            theta[i] += 2.0 * PI;
        }
    }
}

/// Unwraps a phase sequence in place, allowing for a mean phase increment.
///
/// This works like [`unwrap_phase`], but centers the allowed step on an
/// estimate of the mean increment instead of on zero. The estimate is taken
/// from the input as given, before unwrapping, so it equals
/// `(theta[n - 1] - theta[0]) / (n - 1)`. For input wrapped to one turn, that
/// is at most `2*pi / (n - 1)` in magnitude and says little about the true
/// increment. The result then differs from [`unwrap_phase`] only for steps
/// close to half a turn. Prefer [`unwrap_phase`].
///
/// The same input limits apply as for [`unwrap_phase`]. An infinite value, or
/// one too large for a turn to register, makes the call loop forever.
pub fn unwrap_phase2(theta: &mut [f32]) {
    // TODO: verify this method
    let n = theta.len();
    if n < 2 {
        return;
    }

    // make an initial estimate of phase difference
    let mut dphi = 0.0f32;
    for i in 1..n {
        dphi += theta[i] - theta[i - 1];
    }

    dphi /= (n - 1) as f32;

    for i in 1..n {
        while theta[i] - theta[i - 1] > PI + dphi {
            theta[i] -= 2.0 * PI;
        }
        while theta[i] - theta[i - 1] < -PI + dphi {
            theta[i] += 2.0 * PI;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_nco_unwrap_phase)]
    fn test_nco_unwrap_phase() {
        let n = 32; // number of steps
        let tol = 1e-6f32; // error tolerance

        let phi0 = 3.0f32; // initial phase
        let dphi = 0.1f32; // phase step

        let mut phi = vec![0.0f32; n];
        let mut phi_hat = vec![0.0f32; n];

        for i in 0..n {
            // phase input
            phi[i] = phi0 + i as f32 * dphi;

            // wrapped array
            let mut theta = phi[i];
            while theta > PI {
                theta -= 2.0 * PI;
            }
            while theta < -PI {
                theta += 2.0 * PI;
            }

            // initialize output
            phi_hat[i] = theta;
        }

        // unwrap phase
        unwrap_phase(&mut phi_hat);

        // compare input to output
        for i in 0..n {
            assert_abs_diff_eq!(phi[i], phi_hat[i], epsilon = tol);
        }
    }

    #[test]
    fn test_unwrap_phase_edge_cases() {
        // empty and single-element arrays are left alone
        let mut empty: [f32; 0] = [];
        unwrap_phase(&mut empty);
        unwrap_phase2(&mut empty);

        let mut one = [1.5f32];
        unwrap_phase(&mut one);
        assert_eq!(one, [1.5]);
        unwrap_phase2(&mut one);
        assert_eq!(one, [1.5]);

        // an already-continuous sequence is unchanged
        let mut smooth = [0.0f32, 0.1, 0.2, 0.3];
        let want = smooth;
        unwrap_phase(&mut smooth);
        assert_eq!(smooth, want);
    }

    #[test]
    fn test_unwrap_phase_negative_direction() {
        // a decreasing ramp wraps the other way
        let n = 24;
        let dphi = -0.2f32;
        let mut phi = vec![0.0f32; n];
        let mut wrapped = vec![0.0f32; n];

        for i in 0..n {
            phi[i] = 1.0 + i as f32 * dphi;
            let mut t = phi[i];
            while t > PI {
                t -= 2.0 * PI;
            }
            while t < -PI {
                t += 2.0 * PI;
            }
            wrapped[i] = t;
        }

        unwrap_phase(&mut wrapped);
        for i in 0..n {
            assert_abs_diff_eq!(phi[i], wrapped[i], epsilon = 1e-5);
        }
    }
}
