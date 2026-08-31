use crate::error::{Error, Result};
use crate::math::sincd;
use std::f64::consts::PI;

/// Design root-Nyquist raised-cosine filter
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 <= beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// A vec of filter coefficients
pub fn fir_design_rrcos(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
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

        // promote to f64 for extra precision here (sin and cos are sensitive)
        let z = z as f64;
        let beta = beta as f64;

        // Check for special condition where z equals zero
        h[n] = if beta == 0.0 {
            sincd(z) as f32
        } else if z.abs() < 1e-5 {
            (1.0 - beta + 4.0 * beta / PI) as f32
        } else {
            let mut g = 1.0 - 16.0 * beta * beta * z * z;
            g *= g;

            // Check for special condition where 16*beta^2*z^2 equals 1
            if g.abs() < 1e-5 {
                let g1 = 1.0 + 2.0 / PI;
                let g2 = (0.25 * PI / beta).sin();
                let g3 = 1.0 - 2.0 / PI;
                let g4 = (0.25 * PI / beta).cos();
                (beta / 2.0_f64.sqrt() * (g1 * g2 + g3 * g4)) as f32
            } else {
                // TODO find out why original uses sqrt(T) for T = 1.0
                let t1 = ((1.0 + beta) * PI * z).cos();
                let t2 = ((1.0 - beta) * PI * z).sin();
                let t3 = 1.0 / (4.0 * beta * z);
                let t4 = 4.0 * beta / (PI * (1.0 - (16.0 * beta * beta * z * z)));
                (t4 * (t1 + (t2 * t3))) as f32
            }
        }
    }

    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_liquid_firdes_rrcos)]
    fn test_liquid_firdes_rrcos() {
        // Initialize variables
        let k = 2;
        let m = 3;
        let beta = 0.3;
        let offset = 0.0;

        // Initialize pre-determined coefficient array
        let h0: [f32; 13] = [
            -3.311577E-02, 
             4.501582E-02, 
             5.659688E-02, 
            -1.536039E-01, 
            -7.500154E-02, 
             6.153450E-01, 
             1.081972E+00, 
             6.153450E-01, 
            -7.500154E-02, 
            -1.536039E-01, 
             5.659688E-02, 
             4.501582E-02,
            -3.311577E-02
        ];

        // Create filter
        let h = fir_design_rrcos(k, m, beta, offset).unwrap();

        // Ensure data are equal
        for i in 0..13 {
            assert_abs_diff_eq!(h[i], h0[i], epsilon = 0.00001);
        }
    }

    #[test]
    fn test_fir_design_rrcos_zero_beta() {
        let k = 4;
        let m = 3;
        let h = fir_design_rrcos(k, m, 0.0, 0.0).unwrap();

        for (n, &h_n) in h.iter().enumerate() {
            let z = n as f64 / k as f64 - m as f64;
            assert_abs_diff_eq!(h_n, sincd(z) as f32, epsilon = f32::EPSILON);
        }
    }
}
