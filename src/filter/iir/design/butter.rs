use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f32::consts::PI;

/// Compute analog zeros, poles, gain of low-pass Butterworth filter, grouping 
/// complex conjugates together. If filter order is odd, the single real pole 
/// (-1) is at the end of the array. There are no zeros for the analog Butterworth 
/// filter. The gain is unity.
/// 
/// # Arguments
/// 
/// * `n` - filter order
/// * `za` - output analog zeros [length: 0]
/// * `pa` - output analog poles [length: _n]
/// * `ka` - output analog gain
pub fn iir_design_butter_analog(
    n: usize,
    za: &mut Vec<Complex32>,
    pa: &mut Vec<Complex32>,
    ka: &mut Complex32
) -> Result<()> {
    if n == 0 {
        return Err(Error::Config("filter order must be greater than zero".to_string()));
    }

    let r = n % 2;
    let l = (n - r) / 2;

    za.clear();
    pa.clear();
    
    for i in 0..l {
        let theta = (2.0 * (i as f32 + 1.0) + n as f32 - 1.0) * PI / (2.0 * n as f32);
        pa.push(Complex32::from_polar(1.0, theta));
        pa.push(Complex32::from_polar(1.0, -theta));
    }

    if r == 1 {
        pa.push(Complex32::new(-1.0, 0.0));
    }

    if pa.len() != n {
        return Err(Error::Internal("filter order mismatch".to_string()));
    }

    *ka = Complex32::new(1.0, 0.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_butter_azpkf() {
        // test odd/even filter orders
        for n in 1..=10 {
            let mut za = vec![Complex32::new(0.0, 0.0); 0];
            let mut pa = vec![Complex32::new(0.0, 0.0); n];
            let mut ka = Complex32::new(0.0, 0.0);
            iir_design_butter_analog(n, &mut za, &mut pa, &mut ka).unwrap();

            // check lengths
            assert_eq!(za.len(), 0);
            assert_eq!(pa.len(), n);

            // check gain
            assert_abs_diff_eq!(ka.re, 1.0);
            assert_abs_diff_eq!(ka.im, 0.0);

            // check poles
            for i in 0..n {
                // poles should be on unit circle
                assert_abs_diff_eq!(pa[i].norm(), 1.0, epsilon = 1e-6);

                // poles should be in left half of s-plane
                assert!(pa[i].re < 0.0);

                // check values for even orders
                if n % 2 == 0 {
                    // poles should be in conjugate pairs
                    if i % 2 == 0 {
                        assert_abs_diff_eq!(pa[i].re, pa[i+1].re, epsilon = 1e-6);
                        assert_abs_diff_eq!(pa[i].im, -pa[i+1].im, epsilon = 1e-6);
                    }
                } else {
                    // odd orders should have one real pole at -1
                    if i == n-1 {
                        assert_abs_diff_eq!(pa[i].re, -1.0, epsilon = 1e-6);
                        assert_abs_diff_eq!(pa[i].im, 0.0, epsilon = 1e-6);
                    } else if i % 2 == 0 {
                        // remaining poles should be in conjugate pairs
                        assert_abs_diff_eq!(pa[i].re, pa[i+1].re, epsilon = 1e-6);
                        assert_abs_diff_eq!(pa[i].im, -pa[i+1].im, epsilon = 1e-6);
                    }
                }
            }
        }
    }
}