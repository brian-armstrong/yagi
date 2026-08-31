// Root-Nyquist Kaiser filter design.

use crate::error::{Error, Result};
use crate::filter::fir::design::{estimate_req_filter_stopband_attenuation, filter_isi};
use crate::filter::fir::design::kaiser::fir_design_kaiser;

/// Design frequency-shifted root-Nyquist filter based on the Kaiser-windowed sinc.
///
/// # Arguments
/// * `k`      : filter over-sampling rate (samples/symbol)
/// * `m`      : filter delay (symbols)
/// * `beta`   : filter excess bandwidth factor (0,1)
/// * `dt`     : filter fractional sample delay
///
/// # Returns
/// 
/// A vec of filter coefficients
pub fn fir_design_rkaiser(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // validate input
    if k < 2 {
        return Err(Error::Config("k must be at least 2".into()));
    }
    if m < 1 {
        return Err(Error::Config("m must be at least 1".into()));
    }
    if beta <= 0.0 || beta >= 1.0 {
        return Err(Error::Config("beta must be in (0,1)".into()));
    }
    if dt < -1.0 || dt > 1.0 {
        return Err(Error::Config("dt must be in [-1,1]".into()));
    }

    // simply call internal method and ignore output rho value
    let mut h = vec![0.0; 2 * k * m + 1];
    let _rho = fir_design_rkaiser_quadratic(k, m, beta, dt, &mut h)?;
    Ok(h)
}

/// Design frequency-shifted root-Nyquist filter based on
/// the Kaiser-windowed sinc using approximation for rho.
///
/// # Arguments
/// * `k`      : filter over-sampling rate (samples/symbol)
/// * `m`      : filter delay (symbols)
/// * `beta`   : filter excess bandwidth factor (0,1)
/// * `dt`     : filter fractional sample delay
///
/// # Returns
/// 
/// A vec of filter coefficients
pub fn fir_design_arkaiser(k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // validate input
    if k < 2 {
        return Err(Error::Config("k must be at least 2".into()));
    }
    if m < 1 {
        return Err(Error::Config("m must be at least 1".into()));
    }
    if beta <= 0.0 || beta >= 1.0 {
        return Err(Error::Config("beta must be in (0,1)".into()));
    }
    if dt < -1.0 || dt > 1.0 {
        return Err(Error::Config("dt must be in [-1,1]".into()));
    }

    // compute bandwidth correction factor, rho ~ c0 + c1*log(beta) + c2*log^2(beta)
    let c0 = 0.762886 + 0.067663 * (m as f32).ln();
    let c1 = 0.065515;
    let c2 = (1.0 - 0.088 * (m as f32).powf(-1.6)).ln();
    let log_beta = beta.ln();
    let mut rho_hat = c0 + c1 * log_beta + c2 * log_beta * log_beta;

    // ensure range is valid and override if approximation is out of range
    if rho_hat <= 0.0 || rho_hat >= 1.0 {
        rho_hat = rkaiser_approximate_rho(m, beta);
    }

    // compute filter design parameters
    let n = 2 * k * m + 1;                                        // filter length
    let kf = k as f32;                                            // samples/symbol (float)
    let del = beta * rho_hat / kf;                                // transition bandwidth
    let as_ = estimate_req_filter_stopband_attenuation(del, n)?;  // stop-band suppression
    let fc = 0.5 * (1.0 + beta * (1.0 - rho_hat)) / kf;           // filter cutoff

    // compute filter coefficients
    let mut h = fir_design_kaiser(n, fc, as_, dt)?;

    // normalize coefficients
    let e2: f32 = h.iter().map(|&x| x * x).sum();
    for h_i in h.iter_mut() {
        *h_i *= (k as f32 / e2).sqrt();
    }

    Ok(h)
}

/// Find approximate bandwidth adjustment factor rho based on
/// filter delay and desired excess bandwidth factor.
///
/// # Arguments
/// * `m`      : filter delay (symbols)
/// * `beta`   : filter excess bandwidth factor (0,1)
///
/// # Returns
/// 
/// Bandwidth adjustment factor
fn rkaiser_approximate_rho(m: usize, beta: f32) -> f32 {
    if m < 1 {
        panic!("rkaiser_approximate_rho(): m must be greater than 0");
    } else if beta < 0.0 || beta > 1.0 {
        panic!("rkaiser_approximate_rho(): beta must be in [0,1]");
    }

    // compute bandwidth adjustment estimate
    let (c0, c1, c2) = match m {
        1 =>  (0.75749731, 0.06134303, -0.08729663),
        2 =>  (0.81151861, 0.07437658, -0.01427088),
        3 =>  (0.84249538, 0.07684185, -0.00536879),
        4 =>  (0.86140782, 0.07144126, -0.00558652),
        5 =>  (0.87457740, 0.06578694, -0.00650447),
        6 =>  (0.88438797, 0.06074265, -0.00736405),
        7 =>  (0.89216620, 0.05669236, -0.00791222),
        8 =>  (0.89874983, 0.05361696, -0.00815301),
        9 =>  (0.90460032, 0.05167952, -0.00807893),
        10 => (0.91034430, 0.05130753, -0.00746192),
        11 => (0.91587675, 0.05180436, -0.00670711),
        12 => (0.92121875, 0.05273801, -0.00588351),
        13 => (0.92638195, 0.05400764, -0.00508452),
        14 => (0.93123555, 0.05516163, -0.00437306),
        15 => (0.93564993, 0.05596561, -0.00388152),
        16 => (0.93976742, 0.05662274, -0.00348280),
        17 => (0.94351703, 0.05694120, -0.00318821),
        18 => (0.94557273, 0.05227591, -0.00400676),
        19 => (0.95001614, 0.05681641, -0.00300628),
        20 => (0.95281708, 0.05637607, -0.00304790),
        21 => (0.95536256, 0.05575880, -0.00312988),
        22 => (0.95754206, 0.05426060, -0.00385945),
        _ => (0.056873 * (m as f32 + 1e-3).ln() + 0.781388, 0.05426, -0.00386),
    };

    let b = beta.ln();
    let rho_hat = c0 + c1 * b + c2 * b * b;

    // ensure estimate is in [0,1]
    rho_hat.clamp(0.0, 1.0)
}

#[cfg(not(feature = "liquid-quirks"))]
fn fir_design_rkaiser_bounded(
    k: usize,
    m: usize,
    beta: f32,
    dt: f32,
    rho_hat: f32,
    h: &mut [f32],
) -> Result<f32> {
    // algorithm:
    //  1. choose three initial points [x0, x1, x2] where x0 < x1 < x2
    //  2. bisect [x0, x1] and [x1, x2] to obtain xa and xb
    //  3. evaluate the three interior points [xa, x1, xb]
    //  4. retain the interval surrounding the smallest interior point
    //  5. go to step 2

    // initial bandwidth adjustment bounds
    let mut x0 = 0.5 * rho_hat; // lower bound
    let mut x1 = rho_hat;       // midpoint: use initial estimate
    let mut x2 = 1.0;           // upper bound

    // evaluate performance (ISI) at each bandwidth adjustment
    let y0 = fir_design_rkaiser_internal_isi(k, m, beta, dt, x0, h)?;
    let mut y1 = fir_design_rkaiser_internal_isi(k, m, beta, dt, x1, h)?;
    let y2 = fir_design_rkaiser_internal_isi(k, m, beta, dt, x2, h)?;

    // keep the best sampled point so the fallback cannot regress from rho_hat
    let mut rho_opt = x1;
    let mut y_opt = y1;
    if y0 < y_opt {
        rho_opt = x0;
        y_opt = y0;
    }
    if y2 < y_opt {
        rho_opt = x2;
        y_opt = y2;
    }

    // refine the bandwidth adjustment to minimize inter-symbol interference
    for _ in 0..14 {
        // choose midway points xa and xb, then compute their ISI
        let xa = 0.5 * (x0 + x1);
        let xb = 0.5 * (x1 + x2);
        let ya = fir_design_rkaiser_internal_isi(k, m, beta, dt, xa, h)?;
        let yb = fir_design_rkaiser_internal_isi(k, m, beta, dt, xb, h)?;

        if ya < y_opt {
            rho_opt = xa;
            y_opt = ya;
        }
        if yb < y_opt {
            rho_opt = xb;
            y_opt = yb;
        }

        // find the minimum of [ya, y1, yb] and update the bounds
        if y1 < ya && y1 < yb {
            x0 = xa;
            x2 = xb;
        } else if ya < yb {
            x2 = x1;
            x1 = xa;
            y1 = ya;
        } else {
            x0 = x1;
            x1 = xb;
            y1 = yb;
        }
    }

    // return the optimal bandwidth adjustment
    Ok(rho_opt)
}

/// Design frequency-shifted root-Nyquist filter based on
/// the Kaiser-windowed sinc using the quadratic search method.
///
/// # Arguments
/// * `k`      : filter over-sampling rate (samples/symbol)
/// * `m`      : filter delay (symbols)
/// * `beta`   : filter excess bandwidth factor (0,1)
/// * `dt`     : filter fractional sample delay
/// * `h`      : resulting filter [size: 2*k*m+1]
/// 
/// # Returns
/// 
/// Bandwidth adjustment factor
fn fir_design_rkaiser_quadratic(k: usize, m: usize, beta: f32, dt: f32, h: &mut [f32]) -> Result<f32> {
    // algorithm:
    //  1. choose initial bounding points [x0,x2] where x0 < x2
    //  2. choose x1 as bisection of [x0,x2]: x1 = 0.5*(x0+x2)
    //  3. choose x_hat as solution to quadratic equation (x0,y0), (x1,y1), (x2,y2)
    //  4. re-select boundary: (x0,y0) <- (x1,y1)   if x_hat > x1
    //                         (x2,y2) <- (x1,y1)   otherwise
    //  5. go to step 2

    // compute bandwidth adjustment estimate
    let rho_hat = rkaiser_approximate_rho(m, beta);
    let mut rho_opt = rho_hat;

    // bandwidth adjustment
    let mut x1 = rho_hat; // initial estimate

    // evaluate performance (ISI) of each bandwidth adjustment
    let mut y_opt = 0.0;

    // run parabolic search to find bandwidth adjustment x_hat which
    // minimizes the inter-symbol interference of the filter
    let pmax = 14;
    let mut dx = 0.2;    // bounding size
    let tol = 1e-6;      // tolerance

    for p in 0..pmax {
        // choose boundary points
        let x0 = (x1 - dx).max(0.01);
        let x2 = (x1 + dx).min(0.99);

        // evaluate all points
        let y0 = fir_design_rkaiser_internal_isi(k, m, beta, dt, x0, h)?;
        let y1 = fir_design_rkaiser_internal_isi(k, m, beta, dt, x1, h)?;
        let y2 = fir_design_rkaiser_internal_isi(k, m, beta, dt, x2, h)?;

        // save optimum
        if p == 0 || y1 < y_opt {
            rho_opt = x1;
            y_opt = y1;
        }

        // compute minimum of quadratic function
        let ta = y0 * (x1 * x1 - x2 * x2) +
                 y1 * (x2 * x2 - x0 * x0) +
                 y2 * (x0 * x0 - x1 * x1);

        let tb = y0 * (x1 - x2) +
                 y1 * (x2 - x0) +
                 y2 * (x0 - x1);

        // update estimate
        let x_hat = 0.5 * ta / tb;

        // ensure x_hat is within boundary (this will fail if y1 > y0 || y1 > y2)
        if x_hat < x0 || x_hat > x2 {
            // when this fires at p == 0, rho_opt is still rho_hat and the search has
            // contributed nothing 
            if p == 0 {
                rho_opt = fir_design_rkaiser_bounded(k, m, beta, dt, rho_hat, h)?;
            }
            break;
        }

        // break if step size is sufficiently small
        let del = x_hat - x1;
        if p > 3 && del.abs() < tol {
            break;
        }

        // update estimate, reduce bound
        x1 = x_hat;
        dx *= 0.5;
    }

    // re-design filter with optimal value for rho
    fir_design_rkaiser_internal_isi(k, m, beta, dt, rho_opt, h)?;

    // normalize filter magnitude
    let e2: f32 = h.iter().map(|&x| x * x).sum();
    for h_i in h.iter_mut() {
        *h_i *= (k as f32 / e2).sqrt();
    }

    // save transition bandwidth adjustment
    Ok(rho_opt)
}

/// Compute filter coefficients and determine resulting ISI
///
/// # Arguments
/// * `k`      : filter over-sampling rate (samples/symbol)
/// * `m`      : filter delay (symbols)
/// * `beta`   : filter excess bandwidth factor (0,1)
/// * `dt`     : filter fractional sample delay
/// * `rho`    : transition bandwidth adjustment, 0 < rho < 1
/// * `h`      : filter buffer [size: 2*k*m+1]
///
/// # Returns
/// 
/// RMS of ISI
fn fir_design_rkaiser_internal_isi(k: usize, m: usize, beta: f32, dt: f32, rho: f32, h: &mut [f32]) -> Result<f32> {
    // validate input
    if rho < 0.0 || rho > 1.0 {
        return Err(Error::Config(format!("rho must be in [0,1], got {}", rho)));
    }

    let n = 2 * k * m + 1;                   // filter length
    let kf = k as f32;                       // samples/symbol (float)
    let del = beta * rho / kf;               // transition bandwidth
    let as_ = estimate_req_filter_stopband_attenuation(del, n)?;  // stop-band suppression
    let fc = 0.5 * (1.0 + beta * (1.0 - rho)) / kf; // filter cutoff

    // compute filter
    let h_kaiser = fir_design_kaiser(n, fc, as_, dt)?;
    h.copy_from_slice(&h_kaiser);

    // compute filter ISI
    let (isi_rms, _isi_max) = filter_isi(&h, k, m);

    // return RMS of ISI
    Ok(isi_rms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_liquid_rkaiser_config)]
    fn test_liquid_rkaiser_config() {
        // Testing liquid_firdes_rkaiser
        assert!(fir_design_rkaiser(0, 12, 0.2, 0.0).is_err());  // k too small
        assert!(fir_design_rkaiser(2, 0, 0.2, 0.0).is_err());   // m too small
        assert!(fir_design_rkaiser(2, 12, -0.7, 0.0).is_err()); // beta too small
        assert!(fir_design_rkaiser(2, 12, 2.7, 0.0).is_err());  // beta too large
        assert!(fir_design_rkaiser(2, 12, 0.2, -2.0).is_err()); // dt too small
        assert!(fir_design_rkaiser(2, 12, 0.2, 3.0).is_err());  // dt too large

        // Testing liquid_firdes_arkaiser
        assert!(fir_design_arkaiser(0, 12, 0.2, 0.0).is_err());  // k too small
        assert!(fir_design_arkaiser(2, 0, 0.2, 0.0).is_err());   // m too small
        assert!(fir_design_arkaiser(2, 12, -0.7, 0.0).is_err()); // beta too small
        assert!(fir_design_arkaiser(2, 12, 2.7, 0.0).is_err());  // beta too large
        assert!(fir_design_arkaiser(2, 12, 0.2, -2.0).is_err()); // dt too small
        assert!(fir_design_arkaiser(2, 12, 0.2, 3.0).is_err());  // dt too large
    }

    #[test]
    fn test_rkaiser_bounded_fallback() {
        let k = 2;
        let m = 2;
        let beta = 0.15;
        let dt = 0.0;

        let h = fir_design_rkaiser(k, m, beta, dt).unwrap();
        let (isi, _) = filter_isi(&h, k, m);

        let rho_hat = rkaiser_approximate_rho(m, beta);
        let mut h_approx = vec![0.0; 2 * k * m + 1];
        let isi_approx = fir_design_rkaiser_internal_isi(
            k,
            m,
            beta,
            dt,
            rho_hat,
            &mut h_approx,
        ).unwrap();

        assert!(isi_approx > 0.03);
        if cfg!(feature = "liquid-quirks") {
            assert!((isi - isi_approx).abs() < 1e-6);
        } else {
            assert!(isi < 0.02);
            assert!(isi < 0.6 * isi_approx);
        }
    }

}
