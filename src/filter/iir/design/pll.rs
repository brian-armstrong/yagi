use crate::error::{Error, Result};

//           1 + t2 * s
//  F(s) = ------------
//          1 + t1 * s
//

/// Design 2nd-order IIR filter (active lag)
///
/// Arguments:
/// * `w`: filter bandwidth
/// * `zeta`: damping factor (1/sqrt(2) suggested)
/// * `k`: loop gain (1000 suggested)
/// * `b`: output feed-forward coefficients [size: 3 x 1]
/// * `a`: output feed-back coefficients [size: 3 x 1]
pub fn iir_design_pll_active_lag(w: f32, zeta: f32, k: f32, b: &mut [f32; 3], a: &mut [f32; 3]) -> Result<()> {
    // validate input
    if w <= 0.0 {
        return Err(Error::Config("bandwidth must be greater than 0".into()));
    } else if zeta <= 0.0 {
        return Err(Error::Config("damping factor must be greater than 0".into()));
    } else if k <= 0.0 {
        return Err(Error::Config("gain must be greater than 0".into()));
    }

    let wn = w; // natural frequency
    let t1 = k / (wn * wn); //
    let t2 = 2.0 * zeta / wn - 1.0 / k; //

    b[0] = 2.0 * k * (1.0 + t2 / 2.0);
    b[1] = 2.0 * k * 2.0;
    b[2] = 2.0 * k * (1.0 - t2 / 2.0);

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
/// * `w`: filter bandwidth
/// * `zeta`: damping factor (1/sqrt(2) suggested)
/// * `k`: loop gain (1000 suggested)
/// * `b`: output feed-forward coefficients [size: 3 x 1]
/// * `a`: output feed-back coefficients [size: 3 x 1]
pub fn iir_design_pll_active_pi(w: f32, zeta: f32, k: f32, b: &mut [f32; 3], a: &mut [f32; 3]) -> Result<()> {
    // validate input
    if w <= 0.0 {
        return Err(Error::Config("bandwidth must be greater than 0".into()));
    } else if zeta <= 0.0 {
        return Err(Error::Config("damping factor must be greater than 0".into()));
    } else if k <= 0.0 {
        return Err(Error::Config("gain must be greater than 0".into()));
    }

    // loop filter (active lag)
    let wn = w; // natural frequency
    let t1 = k / (wn * wn); //
    let t2 = 2.0 * zeta / wn; //

    b[0] = 2.0 * k * (1.0 + t2 / 2.0);
    b[1] = 2.0 * k * 2.0;
    b[2] = 2.0 * k * (1.0 - t2 / 2.0);

    a[0] = t1 / 2.0;
    a[1] = -t1;
    a[2] = t1 / 2.0;

    Ok(())
}
