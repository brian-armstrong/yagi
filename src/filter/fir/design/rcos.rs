use crate::error::{Error, Result};
use crate::math::sincd;
use std::f64::consts::PI;


/// Design Nyquist raised-cosine filter
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// A vec of filter coefficients
pub fn fir_design_rcos(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    if k < 1 {
        return Err(Error::Config("k must be greater than 0".into()));
    }
    if m < 1 {
        return Err(Error::Config("m must be greater than 0".into()));
    }
    if beta < 0.0 || beta > 1.0 {
        return Err(Error::Config("beta must be in [0,1]".into()));
    }

    let h_len = 2 * k * m + 1;
    let mut h = vec![0.0; h_len];

    // Calculate filter coefficients
    for n in 0..h_len {
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
        h[n] = if t3.abs() < 1e-3 {
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
    #[autotest_annotate(autotest_liquid_firdes_rcos)]
    fn test_liquid_firdes_rcos() {
        // Initialize variables
        let k = 2;
        let m = 3;
        let beta = 0.3;
        let offset = 0.0;

        // Initialize pre-determined coefficient array
        // nb the last initializer was missing from the autotest
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