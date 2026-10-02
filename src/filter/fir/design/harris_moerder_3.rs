use super::{filter_isi, fir_design_pm, FirPmWeightType};
use crate::error::{Error, Result};

// References
//  [harris:2005] f. harris, C. Dick, S. Seshagiri, K. Moerder, "An
//      Improved Square-Root Nyquist Shaping Filter," Proceedings of
//      the Software-Defined Radio Forum, 2005
//

/// Designs a Harris-Moerder root-Nyquist pulse.
///
/// The pulse has `k` samples per symbol and spans `m` symbols on each side of
/// its center tap, for `2 * k * m + 1` taps. `beta` is the excess bandwidth.
/// The pulse has zero fractional delay.
///
/// The design starts from a Parks-McClellan root-Nyquist filter whose
/// passband ends at `(1 - beta) / (2 * k)`. It then steps the passband edge
/// down from `1 / (2 * k)` toward that value and keeps the design with the
/// lowest RMS inter-symbol interference, as measured by [`filter_isi`]. The
/// search stops once the interference stops improving. The taps are scaled to
/// a total energy of `k`.
///
/// # Errors
///
/// Returns an error when `k` is less than two, `m` is zero, `beta` is outside
/// `0.0..=1.0`, or a Parks-McClellan design fails. After the first design,
/// non-convergence ends the search instead of returning an error.
///
/// # Panics
///
/// Panics when `2 * k * m + 1` overflows `usize` and overflow checks are
/// enabled. Without overflow checks the count wraps.
pub fn fir_design_hm3(k: usize, m: usize, beta: f32) -> Result<Vec<f32>> {
    if k < 2 {
        return Err(Error::Config("k must be greater than 1".into()));
    }
    if m < 1 {
        return Err(Error::Config("m must be greater than 0".into()));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(Error::Config("beta must be in [0,1]".into()));
    }

    let n = 2 * k * m + 1; // filter length

    let fc = 1.0 / (2.0 * k as f32); // filter cutoff
    let mut fp = fc * (1.0 - beta); // pass-band
    let fs = fc * (1.0 + beta); // stop-band

    // root nyquist
    let mut bands = [0.0, fp, fc, fc, fs, 0.5];
    let des = [1.0, 1.0 / 2.0_f32.sqrt(), 0.0];
    let weights = [1.0, 1.0, 1.0];

    let wtype = [FirPmWeightType::Flat, FirPmWeightType::Flat, FirPmWeightType::Exp];

    let mut h = vec![0.0; n];
    let h_pm = fir_design_pm(n, &bands, &des, Some(&weights), Some(&wtype))?;
    h.copy_from_slice(&h_pm);

    let (isi_rms, _isi_max) = filter_isi(&h, k, m);

    // iterate...
    let mut isi_rms_min = isi_rms;
    let pmax = 100;
    for p in 0..pmax {
        // increase pass-band edge
        fp = fc * (1.0 - beta * p as f32 / pmax as f32);
        bands[1] = fp;

        // execute filter design
        let h_pm = match fir_design_pm(n, &bands, &des, Some(&weights), Some(&wtype)) {
            Ok(h) => h,
            Err(Error::NoConvergence(_)) => break, // retain the best prior candidate
            Err(error) => return Err(error),
        };

        // compute inter-symbol interference (MSE, max)
        let (isi_rms, _isi_max) = filter_isi(&h_pm, k, m);

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
