use crate::error::{Error, Result};

//           1 + t2 * s
//  F(s) = ------------
//          1 + t1 * s
//

/// Designs a second-order active-lag loop filter for a phase-locked loop.
///
/// The analog loop filter is `F(s) = (1 + t2 * s) / (1 + t1 * s)`, with
/// `t1 = loop_gain / natural_frequency^2` and
/// `t2 = 2 * damping_factor / natural_frequency - 1 / loop_gain`.
/// `natural_frequency` is in radians per sample, and `damping_factor` is
/// dimensionless. `b` and `a` each receive three coefficients in ascending
/// powers of `z^-1`.
///
/// Very small positive values can overflow `t1`.
///
/// # Errors
///
/// Returns an error when `natural_frequency`, `damping_factor`, or
/// `loop_gain` is zero or negative. `b` and `a` are unchanged on error.
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

/// Designs a second-order active proportional-integral loop filter for a
/// phase-locked loop.
///
/// The analog loop filter is `F(s) = (1 + t2 * s) / (t1 * s)`, with
/// `t1 = loop_gain / natural_frequency^2` and
/// `t2 = 2 * damping_factor / natural_frequency`. `natural_frequency` is in
/// radians per sample, and `damping_factor` is dimensionless. `b` and `a`
/// each receive three coefficients in ascending powers of `z^-1`.
///
/// Very small positive values can overflow `t1`.
///
/// # Errors
///
/// Returns an error when `natural_frequency`, `damping_factor`, or
/// `loop_gain` is zero or negative. `b` and `a` are unchanged on error.
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
