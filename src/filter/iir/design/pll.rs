use crate::error::{Error, Result};

//           1 + t2 * s
//  F(s) = ------------
//          1 + t1 * s
//

/// Design 2nd-order IIR filter (active lag)
///
/// Arguments:
/// * `natural_frequency`: analog-prototype natural angular frequency, normalized
///   to the sample interval (approximately radians per sample for small values)
/// * `damping_factor`: damping factor (1/sqrt(2) suggested)
/// * `loop_gain`: loop gain (1000 suggested)
/// * `b`: output feed-forward coefficients [size: 3 x 1]
/// * `a`: output feed-back coefficients [size: 3 x 1]
pub fn iir_design_pll_active_lag(
    natural_frequency: f32,
    damping_factor: f32,
    loop_gain: f32,
    b: &mut [f32; 3],
    a: &mut [f32; 3],
) -> Result<()> {
    // validate input
    if natural_frequency <= 0.0 {
        return Err(Error::Config("natural frequency must be greater than 0".into()));
    } else if damping_factor <= 0.0 {
        return Err(Error::Config("damping factor must be greater than 0".into()));
    } else if loop_gain <= 0.0 {
        return Err(Error::Config("loop gain must be greater than 0".into()));
    }

    let t1 = loop_gain / (natural_frequency * natural_frequency);
    let t2 = 2.0 * damping_factor / natural_frequency - 1.0 / loop_gain;

    b[0] = 2.0 * loop_gain * (1.0 + t2 / 2.0);
    b[1] = 2.0 * loop_gain * 2.0;
    b[2] = 2.0 * loop_gain * (1.0 - t2 / 2.0);

    a[0] = 1.0 + t1 / 2.0;
    a[1] = -t1;
    a[2] = -1.0 + t1 / 2.0;

    Ok(())
}

//           1 + t2 * s
//  F(s) = ------------
//            t1 * s
//

/// Design 2nd-order IIR filter (active PI)
///
/// Arguments:
/// * `natural_frequency`: analog-prototype natural angular frequency, normalized
///   to the sample interval (approximately radians per sample for small values)
/// * `damping_factor`: damping factor (1/sqrt(2) suggested)
/// * `loop_gain`: loop gain (1000 suggested)
/// * `b`: output feed-forward coefficients [size: 3 x 1]
/// * `a`: output feed-back coefficients [size: 3 x 1]
pub fn iir_design_pll_active_pi(
    natural_frequency: f32,
    damping_factor: f32,
    loop_gain: f32,
    b: &mut [f32; 3],
    a: &mut [f32; 3],
) -> Result<()> {
    // validate input
    if natural_frequency <= 0.0 {
        return Err(Error::Config("natural frequency must be greater than 0".into()));
    } else if damping_factor <= 0.0 {
        return Err(Error::Config("damping factor must be greater than 0".into()));
    } else if loop_gain <= 0.0 {
        return Err(Error::Config("loop gain must be greater than 0".into()));
    }

    // loop filter (active lag)
    let t1 = loop_gain / (natural_frequency * natural_frequency);
    let t2 = 2.0 * damping_factor / natural_frequency;

    b[0] = 2.0 * loop_gain * (1.0 + t2 / 2.0);
    b[1] = 2.0 * loop_gain * 2.0;
    b[2] = 2.0 * loop_gain * (1.0 - t2 / 2.0);

    a[0] = t1 / 2.0;
    a[1] = -t1;
    a[2] = t1 / 2.0;

    Ok(())
}
