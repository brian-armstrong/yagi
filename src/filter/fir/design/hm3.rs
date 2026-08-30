use crate::error::{Error, Result};
use crate::filter::fir::design;

// References
//  [harris:2005] f. harris, C. Dick, S. Seshagiri, K. Moerder, "An
//      Improved Square-Root Nyquist Shaping Filter," Proceedings of
//      the Software-Defined Radio Forum, 2005
//

/// Design root-Nyquist harris-Moerder filter using Parks-McClellan algorithm
///
/// # Arguments
/// * `k`      : filter over-sampling rate (samples/symbol)
/// * `m`      : filter delay (symbols)
/// * `beta`   : filter excess bandwidth factor (0,1)
/// * `dt`     : filter fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_hm3(k: usize, m: usize, beta: f32, _dt: f32) -> Result<Vec<f32>> {
    if k < 2 {
        return Err(Error::Config("k must be greater than 1".into()));
    }
    if m < 1 {
        return Err(Error::Config("m must be greater than 0".into()));
    }
    if beta < 0.0 || beta > 1.0 {
        return Err(Error::Config("beta must be in [0,1]".into()));
    }

    let n = 2 * k * m + 1;       // filter length

    let fc = 1.0 / (2.0 * k as f32); // filter cutoff
    let mut fp = fc * (1.0 - beta);    // pass-band
    let fs = fc * (1.0 + beta);    // stop-band

    // root nyquist
    let num_bands = 3;
    let mut bands = [0.0, fp, fc, fc, fs, 0.5];
    let des = [1.0, 1.0 / 2.0_f32.sqrt(), 0.0];
    let weights = [1.0, 1.0, 1.0];

    let btype = design::pm::FirPmBandType::Bandpass;
    let wtype = [
        design::pm::FirPmWeightType::Flat,
        design::pm::FirPmWeightType::Flat,
        design::pm::FirPmWeightType::Exp,
    ];

    let mut h = vec![0.0; n];
    let h_pm = design::pm::fir_design_pm(n, num_bands, &bands, &des, Some(&weights), Some(&wtype), btype)?;
    h.copy_from_slice(&h_pm);

    let (isi_rms, _isi_max) = design::filter_isi(&h, k, m);

    // iterate...
    let mut isi_rms_min = isi_rms;
    let pmax = 100;
    for p in 0..pmax {
        // increase pass-band edge
        fp = fc * (1.0 - beta * p as f32 / pmax as f32);
        bands[1] = fp;

        // execute filter design
        let h_pm = match design::pm::fir_design_pm(n, num_bands, &bands, &des, Some(&weights), Some(&wtype), btype) {
            Ok(h) => h,
            Err(Error::NoConvergence(_)) => break, // retain the best prior candidate
            Err(error) => return Err(error),
        };

        // compute inter-symbol interference (MSE, max)
        let (isi_rms, _isi_max) = design::filter_isi(&h_pm, k, m);

        if isi_rms > isi_rms_min {
            // search complete
            break;
        } else {
            isi_rms_min = isi_rms;
            h.copy_from_slice(&h_pm);
        }
    }

    // normalize
    let e2: f32 = h.iter().map(|&x| x * x).sum();
    for h_i in h.iter_mut() {
        *h_i *= (k as f32 / e2).sqrt();
    }
    
    Ok(h)
}
