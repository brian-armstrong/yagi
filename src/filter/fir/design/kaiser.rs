use crate::error::{Error, Result};
use crate::math::sincf;
use crate::math::windows;

/// Design FIR using kaiser window
///
/// # Arguments
/// * `n`      : filter length, n > 0
/// * `fc`     : cutoff frequency, 0 < fc < 0.5
/// * `as_`    : stop-band attenuation \[dB\], as_ > 0
/// * `mu`     : fractional sample offset, -0.5 < mu < 0.5
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_kaiser(n: usize, fc: f32, as_: f32, mu: f32) -> Result<Vec<f32>> {
    // validate input
    if mu <= -0.5 || mu > 0.5 {
        return Err(Error::Config(format!("fractional sample offset ({}) out of range (-0.5, 0.5)", mu)));
    }
    if fc <= 0.0 || fc > 0.5 {
        return Err(Error::Config(format!("cutoff frequency ({}) out of range (0, 0.5)", fc)));
    }
    if n == 0 {
        return Err(Error::Config("filter length must be greater than zero".into()));
    }
    if as_ <= 0.0 {
        return Err(Error::Config("stop-band attenuation must be greater than zero".into()));
    }

    // compute Kaiser window paramter (beta)
    let beta = kaiser_beta_stopband_attenuation(as_);

    let mut h = vec![0.0; n];

    for i in 0..n {
        // time vector
        let t = i as f32 - (n as f32 - 1.0) / 2.0 + mu;

        // sinc prototype
        let h1 = sincf(2.0 * fc * t);

        // kaiser window
        let h2 = windows::kaiser(i, n, beta)?;

        // composite
        h[i] = h1 * h2;
    }

    Ok(h)
}


/// Compute Kaiser window beta factor from stop-band attenuation
///
/// # Arguments
/// * `as_`    : target filter's stop-band attenuation \[dB\], as_ > 0
///
/// # Returns
/// 
/// Kaiser window beta factor
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