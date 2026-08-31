use crate::error::{Error, Result};
use crate::fft::{fft_run, Direction};

use num_complex::Complex32;

use super::FirFilterShape;

// Flipped Nyquist/root-Nyquist filter designs
//
// References:
//   [Beaulieu:2001]
//   [Assalini:2004]


/// Design flipped Nyquist/root-Nyquist filter
///
/// # Arguments
/// * `ftype`  : filter type (e.g. FirdesFilterType::Fexp)
/// * `root`   : square-root Nyquist filter?
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_fnyquist(
    ftype: FirFilterShape,
    root: bool,
    k: usize,
    m: usize,
    beta: f32,
    _dt: f32,
) -> Result<Vec<f32>> {
    // validate input
    if k < 1 {
        return Err(Error::Config("k must be greater than 0".into()));
    }
    if m < 1 {
        return Err(Error::Config("m must be greater than 0".into()));
    }
    if beta < 0.0 || beta > 1.0 {
        return Err(Error::Config("beta must be in [0,1]".into()));
    }

    // derived values
    let h_len = 2 * k * m + 1; // filter length

    let mut h_prime = vec![0.0; h_len]; // frequency response of Nyquist filter (real)
    let mut h = vec![Complex32::new(0.0, 0.0); h_len]; // frequency response of Nyquist filter
    let mut h_time = vec![Complex32::new(0.0, 0.0); h_len]; // impulse response of Nyquist filter

    // compute Nyquist filter frequency response
    match ftype {
        FirFilterShape::Fexp => fir_design_fexp_freqresponse(k, m, beta, &mut h_prime)?,
        FirFilterShape::Fsech => fir_design_fsech_freqresponse(k, m, beta, &mut h_prime)?,
        FirFilterShape::Farcsech => fir_design_farcsech_freqresponse(k, m, beta, &mut h_prime)?,
        _ => return Err(Error::Config("unsupported filter type".into())),
    }

    // copy result to fft input buffer, computing square root
    // if required
    for i in 0..h_len {
        h[i] = Complex32::new(if root { h_prime[i].sqrt() } else { h_prime[i] }, 0.0);
    }

    // compute ifft
    fft_run(&h, &mut h_time, Direction::Backward);

    // copy shifted, scaled response
    let mut h_out = vec![0.0; h_len];
    for i in 0..h_len {
        h_out[i] = h_time[(i + k * m + 1) % h_len].re * k as f32 / h_len as f32;
    }

    Ok(h_out)
}

/// Design fexp Nyquist filter
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_fexp(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // compute response using generic function
    fir_design_fnyquist(FirFilterShape::Fexp, false, k, m, beta, dt)
}

/// Design fexp square-root Nyquist filter
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_rfexp(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // compute response using generic function
    fir_design_fnyquist(FirFilterShape::Fexp, true, k, m, beta, dt)
}

/// Flipped exponential frequency response
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `H`      : output frequency response (length: 2*k*m+1)
pub fn fir_design_fexp_freqresponse(k: usize, _m: usize, beta: f32, h: &mut [f32]) -> Result<()> {
    let h_len = h.len();

    let f0 = 0.5 * (1.0 - beta) / k as f32;
    let f1 = 0.5 / k as f32;
    let f2 = 0.5 * (1.0 + beta) / k as f32;

    let b = 0.5 / k as f32;
    let gamma = (2.0f32).ln() / (beta * b);

    // compute frequency response of Nyquist filter
    for i in 0..h_len {
        let mut f = i as f32 / h_len as f32;
        if f > 0.5 {
            f = f - 1.0;
        }

        // enforce even symmetry
        f = f.abs();

        h[i] = if f <= f0 {
            // pass band
            1.0
        } else if f > f0 && f < f2 {
            // transition band
            if f < f1 {
                (gamma * (b * (1.0 - beta) - f)).exp()
            } else {
                1.0 - (gamma * (f - (1.0 + beta) * b)).exp()
            }
        } else {
            // stop band
            0.0
        };
    }

    Ok(())
}

/// Design fsech Nyquist filter
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_fsech(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // compute response using generic function
    fir_design_fnyquist(FirFilterShape::Fsech, false, k, m, beta, dt)
}

/// Design fsech square-root Nyquist filter
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_rfsech(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // compute response using generic function
    fir_design_fnyquist(FirFilterShape::Fsech, true, k, m, beta, dt)
}

/// Flipped sech frequency response
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `H`      : output frequency response (length: 2*k*m+1)
pub fn fir_design_fsech_freqresponse(k: usize, _m: usize, beta: f32, h: &mut [f32]) -> Result<()> {
    let h_len = h.len();

    let f0 = 0.5 * (1.0 - beta) / k as f32;
    let f1 = 0.5 / k as f32;
    let f2 = 0.5 * (1.0 + beta) / k as f32;

    let b = 0.5 / k as f32;
    let gamma = (3.0f32.sqrt() + 2.0f32).ln() / (beta * b);

    // compute frequency response of Nyquist filter
    for i in 0..h_len {
        let mut f = i as f32 / h_len as f32;
        if f > 0.5 {
            f = f - 1.0;
        }

        // enforce even symmetry
        f = f.abs();

        h[i] = if f <= f0 {
            // pass band
            1.0
        } else if f > f0 && f < f2 {
            // transition band
            if f < f1 {
                (gamma * (f - b * (1.0 - beta))).cosh().recip()
            } else {
                1.0 - (gamma * (b * (1.0 + beta) - f)).cosh().recip()
            }
        } else {
            // stop band
            0.0
        };  
    }

    Ok(())
}

/// Design farcsech Nyquist filter
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_farcsech(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // compute response using generic function
    fir_design_fnyquist(FirFilterShape::Farcsech, false, k, m, beta, dt)
}

/// Design farcsech square-root Nyquist filter
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_rfarcsech(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // compute response using generic function
    fir_design_fnyquist(FirFilterShape::Farcsech, true, k, m, beta, dt)
}

/// hyperbolic arc-secant
///
/// # Arguments
/// * `z`      : input value
///
/// # Returns
/// 
/// hyperbolic arc-secant
fn asechf(z: f32) -> f32 {
    if z <= 0.0 || z > 1.0 {
        return 0.0;
    }

    let z_inv = 1.0 / z;

    return ((z_inv - 1.0).sqrt() * (z_inv + 1.0).sqrt() + z_inv).ln()
}

/// Flipped arccosh frequency response
///
/// # Arguments
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : rolloff factor (0 < beta <= 1)
/// * `H`      : output frequency response (length: 2*k*m+1)
pub fn fir_design_farcsech_freqresponse(k: usize, _m: usize, beta: f32, h: &mut [f32]) -> Result<()> {
    let h_len = h.len();

    let f0 = 0.5 * (1.0 - beta) / k as f32;
    let f1 = 0.5 / k as f32;
    let f2 = 0.5 * (1.0 + beta) / k as f32;

    let b = 0.5 / k as f32;
    let gamma = (3.0f32.sqrt() + 2.0f32).ln() / (beta * b);
    let zeta = 1.0f32 / (2.0f32 * beta * b);

    // compute frequency response of Nyquist filter
    for i in 0..h_len {
        let mut f = i as f32 / h_len as f32;
        if f > 0.5 {
            f = f - 1.0;
        }

        // enforce even symmetry
        f = f.abs();

        h[i] = if f <= f0 {
            // pass band
            1.0
        } else if f > f0 && f < f2 {
            // transition band
            if f < f1 {
                1.0 - (zeta / gamma) * asechf(zeta * (b * (1.0 + beta) - f))
            } else {
                (zeta / gamma) * asechf(zeta * (f - b * (1.0 - beta)))
            }
        } else {
            // stop band
            0.0
        }   
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fnyquist_lower_transition_endpoint() {
        let k = 4;
        let m = 3;

        // at beta=1, the lower transition endpoint f0 is DC. The response at
        // this endpoint is unity, so every design must preserve a DC gain of k.
        for ftype in [FirFilterShape::Fexp, FirFilterShape::Fsech, FirFilterShape::Farcsech] {
            for root in [false, true] {
                let h = fir_design_fnyquist(ftype, root, k, m, 1.0, 0.0).unwrap();
                assert_abs_diff_eq!(h.iter().sum::<f32>(), k as f32, epsilon = 1e-4);
            }
        }
    }
}
