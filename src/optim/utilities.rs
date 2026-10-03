//! Sample objective functions for numerical optimization.
//!
//! Each accepts a parameter slice and returns `0.0` for an empty slice.

use std::f32::consts::PI;

/// Returns the Rosenbrock function of `parameters`, whose minimum of zero is
/// at all ones.
///
/// The value is the sum of `(1 - parameters[i])^2 +
/// 100 * (parameters[i+1] - parameters[i]^2)^2` over neighboring pairs.
/// One element gives `(1 - parameters[0])^2`, and an empty slice gives zero.
pub fn rosenbrock(parameters: &[f32]) -> f32 {
    if parameters.is_empty() {
        return 0.0;
    } else if parameters.len() == 1 {
        return (1.0 - parameters[0]) * (1.0 - parameters[0]);
    }

    let mut u = 0.0f32;
    for i in 0..parameters.len() - 1 {
        u += (1.0 - parameters[i]).powi(2) + 100.0 * (parameters[i + 1] - parameters[i].powi(2)).powi(2);
    }

    u
}

/// Returns an inverted Gaussian bowl, whose minimum of zero is at all ones.
///
/// The value is `1 - exp(-t)`, where `t` sums
/// `(parameters[i] - 1)^2 / sigma_i^2` and `sigma_i = 1.5^i`, so it lies in
/// `[0, 1)` and later coordinates matter less. An empty slice gives zero.
pub fn invgauss(parameters: &[f32]) -> f32 {
    if parameters.is_empty() {
        return 0.0;
    }

    let mut t = 0.0f32;
    let mut sigma = 1.0f32;
    for &vi in parameters {
        t += (vi - 1.0) * (vi - 1.0) / (sigma * sigma);

        // increase variance along this dimension
        sigma *= 1.5;
    }

    1.0 - (-t).exp()
}

/// Returns a multimodal function, whose global minimum of zero is at the
/// origin.
///
/// A product of cosine terms puts a local minimum near every point with
/// integer coordinates, inside a wide Gaussian envelope centered on the
/// origin. An empty slice gives zero.
pub fn multimodal(parameters: &[f32]) -> f32 {
    if parameters.is_empty() {
        return 0.0;
    }

    let mut t0 = 1.0f32;
    let mut t1 = 0.0f32;
    let sigma = 4.0f32;

    for &vi in parameters {
        t0 *= 0.5 + 0.5 * (2.0 * PI * vi).cos();
        t1 += vi * vi / (sigma * sigma);
    }

    1.0 - t0 * (-t1).exp()
}

/// Returns a function of the first two coordinates with a spiral-shaped
/// valley.
///
/// Further coordinates add their squares. One coordinate gives its square,
/// and an empty slice gives zero.
///
/// With two or more coordinates, the value approaches zero near the origin
/// along the negative first axis but is one at the origin itself. The
/// function is discontinuous there.
pub fn spiral(parameters: &[f32]) -> f32 {
    if parameters.is_empty() {
        return 0.0;
    } else if parameters.len() == 1 {
        return parameters[0] * parameters[0];
    }

    // n is at least 2
    let r_hat = (parameters[0] * parameters[0] + parameters[1] * parameters[1]).sqrt();
    let theta_hat = parameters[1].atan2(parameters[0]);

    // keep the wrap in f64, then narrow back to f32.
    let mut delta = (theta_hat - r_hat * 10.0) as f64;
    while delta > std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    while delta < -std::f64::consts::PI {
        delta += 2.0 * std::f64::consts::PI;
    }

    let delta = (delta / std::f64::consts::PI) as f32;

    let mut u = 1.0 - delta * delta * (-r_hat * r_hat / 10.0).exp();

    // additional error...
    for &vi in &parameters[2..] {
        u += vi * vi;
    }

    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_optim_rosenbrock)]
    fn optim_rosenbrock() {
        // optimum
        let v_ones = [1.0f32; 8];
        assert_abs_diff_eq!(rosenbrock(&v_ones), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(rosenbrock(&v_ones[..1]), 0.0, epsilon = 1e-6);

        // very far from optimum
        let v_misc = [0.3f32, 1.0, 4.5, -2.2, 6.7, -0.2, 1.1, -0.9];
        assert!(rosenbrock(&v_misc) > 1000.0);

        // invalid configuration (empty input)
        assert_eq!(rosenbrock(&[]), 0.0);
    }

    #[test]
    fn optim_invgauss() {
        // minimum at {1,1,...}, value 0
        let v_ones = [1.0f32; 6];
        assert_abs_diff_eq!(invgauss(&v_ones), 0.0, epsilon = 1e-6);
        // away from the optimum the value is strictly greater and bounded by 1
        let v_misc = [0.3f32, -1.0, 2.5, 0.0];
        let u = invgauss(&v_misc);
        assert!(u > 0.0 && u < 1.0);
        assert_eq!(invgauss(&[]), 0.0);
    }

    #[test]
    fn optim_multimodal() {
        // minimum at {0,0,...}: t0 = prod(1) = 1, t1 = 0, so u = 0
        let v_zeros = [0.0f32; 5];
        assert_abs_diff_eq!(multimodal(&v_zeros), 0.0, epsilon = 1e-6);
        // integer coordinates sit on the same cosine peaks (cos(2*pi*k) = 1)
        let v_int = [1.0f32, -2.0, 3.0];
        assert!(multimodal(&v_int) < multimodal(&[0.5f32, 0.5, 0.5]));
        assert_eq!(multimodal(&[]), 0.0);
    }

    #[test]
    fn optim_spiral() {
        // single-dimension case reduces to v^2
        assert_abs_diff_eq!(spiral(&[3.0]), 9.0, epsilon = 1e-6);
        // extra dimensions beyond the first two add their squares
        let u2 = spiral(&[0.5f32, -0.5]);
        let u4 = spiral(&[0.5f32, -0.5, 2.0, 1.0]);
        assert_abs_diff_eq!(u4 - u2, 5.0, epsilon = 1e-5);
        assert_eq!(spiral(&[]), 0.0);
    }
}
