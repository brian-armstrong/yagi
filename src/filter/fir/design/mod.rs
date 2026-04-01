mod fnyquist;
mod gmsk;
mod hm3;
mod kaiser;
mod pm_halfband;
mod pm;
mod rcos;
mod rkaiser;
mod rrcos;

pub use fnyquist::*;
pub use gmsk::*;
pub use hm3::*;
pub use kaiser::*;
pub use pm_halfband::*;
pub use pm::*;
pub use rcos::*;
pub use rkaiser::*;
pub use rrcos::*;

use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::math::sincf;
use crate::math::windows;

use num_complex::{Complex32, Complex64, ComplexFloat};

//
// Finite impulse response filter design
//
// References:
//  [Herrmann:1973] O. Herrmann, L. R. Rabiner, and D. S. K. Chan,
//      "Practical design rules for optimum finite impulse response
//      lowpass digital filters," Bell Syst. Tech. Journal, vol. 52,
//      pp. 769--99, July-Aug. 1973
//  [Vaidyanathan:1993] Vaidyanathan, P. P., "Multirate Systems and
//      Filter Banks," 1993, Prentice Hall, Section 3.2.1

/// FIR filter design shape
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FirFilterShape {
    // Nyquist filter prototypes

    /// Nyquist Kaiser filter
    Kaiser,
    /// Parks-McClellan filter
    Pm,
    /// Raised-cosine filter
    Rcos,
    /// Flipped exponential filter
    Fexp,
    /// Flipped hyperbolic secant filter
    Fsech,
    /// Flipped arc-hyperbolic secant filter
    Farcsech,

    // Root Nyquist filter prototypes

    /// Root-Nyquist Kaiser (approximate optimum)
    Arkaiser,
    /// Root-Nyquist Kaiser (true optimum)
    Rkaiser,
    /// Root raised-cosine filter
    Rrcos,
    /// Harris-Moerder-3 filter
    Hm3,
    /// GMSK transmit filter
    Gmsktx,
    /// GMSK receive filter
    Gmskrx,
    /// Flipped exponential filter
    Rfexp,
    /// Flipped hyperbolic secant filter
    Rfsech,
    /// Flipped arc-hyperbolic secant filter
    Rfarcsech,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterInfo {
    short_name: &'static str,
    long_name: &'static str,
}

// Define window information
const FILTER_INFO: [FilterInfo; 15] = [
    // FilterInfo { short_name: "unknown", long_name: "unknown" },
    FilterInfo { short_name: "kaiser", long_name: "Nyquist Kaiser filter" },
    FilterInfo { short_name: "pm", long_name: "Parks-McClellan filter" },
    FilterInfo { short_name: "rcos", long_name: "raised-cosine filter" },
    FilterInfo { short_name: "fexp", long_name: "flipped exponential" },
    FilterInfo { short_name: "fsech", long_name: "flipped hyperbolic secant" },
    FilterInfo { short_name: "farcsech", long_name: "flipped arc-hyperbolic secant" },
    FilterInfo { short_name: "arkaiser", long_name: "root-Nyquist Kaiser (approximate optimum)" },
    FilterInfo { short_name: "rkaiser", long_name: "root-Nyquist Kaiser (true optimum)" },
    FilterInfo { short_name: "rrcos", long_name: "root raised-cosine filter" },
    FilterInfo { short_name: "hm3", long_name: "Harris-Moerder-3 filter" },
    FilterInfo { short_name: "gmsktx", long_name: "GMSK transmit filter" },
    FilterInfo { short_name: "gmskrx", long_name: "GMSK receive filter" },
    FilterInfo { short_name: "rfexp", long_name: "root flipped exponential" },
    FilterInfo { short_name: "rfsech", long_name: "root flipped hyperbolic secant" },
    FilterInfo { short_name: "rfarcsech", long_name: "root flipped arc-hyperbolic secant" },
];

/// Convert filter name to filter type
///
/// # Arguments
/// * `s`      : filter name
///
/// # Returns
/// 
/// A `FirFilterType` matching the filter name
impl FirFilterShape {
    pub fn from_str(s: &str) -> Result<FirFilterShape> {
        for (i, info) in FILTER_INFO.iter().enumerate() {
            if info.short_name == s {
                return Ok(unsafe { std::mem::transmute(i as u8) });
            }
        }
        Err(Error::Config("Unknown filter type".into()))
    }
}


const USE_KAISER_REQ_FILTER_LEN_ESTIMATE: bool = true;


/// estimate required filter length given transition bandwidth and
/// stop-band attenuation
///
/// # Arguments
/// * `df`     : transition bandwidth (0 < df < 0.5)
/// * `as_`    : stopband suppression level \[dB\] (as_ > 0)
///
/// # Returns
/// 
/// Filter length
pub fn estimate_req_filter_len(df: f32, as_: f32) -> Result<usize> {
    if df <= 0.0 || df > 0.5 {
        return Err(Error::Config(format!("cutoff frequency ({}) out of range (0, 0.5)", df)));
    }
    if as_ <= 0.0 {
        return Err(Error::Config("stopband attenuation must be greater than zero".into()));
    }

    let n = if USE_KAISER_REQ_FILTER_LEN_ESTIMATE {
        estimate_req_filter_len_kaiser(df, as_)?
    } else {
        estimate_req_filter_len_herrmann(df, as_)?
    };
    Ok(n as usize)
}

/// estimate filter stop-band attenuation given
/// * `df`     : transition bandwidth (0 < df < 0.5)
/// * `n`      : filter length
///
/// # Returns
/// 
/// Stop-band attenuation \[dB\]
pub fn estimate_req_filter_stopband_attenuation(df: f32, n: usize) -> Result<f32> {
    // run search for stop-band attenuation which gives these results
    let mut as0 = 0.01;    // lower bound
    let mut as1 = 200.0;   // upper bound
    let mut as_hat = 0.0;  // stop-band attenuation estimate
    let mut n_hat;         // filter length estimate

    for _ in 0..20 {
        // bisect limits
        as_hat = 0.5 * (as1 + as0);
        n_hat = if USE_KAISER_REQ_FILTER_LEN_ESTIMATE {
            estimate_req_filter_len_kaiser(df, as_hat)?
        } else {
            estimate_req_filter_len_herrmann(df, as_hat)?
        };
        // update limits
        if n_hat < n as f32 {
            as0 = as_hat;
        } else {
            as1 = as_hat;
        }
    }
    Ok(as_hat)
}

/// estimate filter transition bandwidth given
/// * `as_`    : stop-band attenuation \[dB\], as_ > 0
/// * `n`      : filter length
///
/// # Returns
/// 
/// Transition bandwidth
pub fn estimate_req_filter_transition_bandwidth(as_: f32, n: usize) -> Result<f32> {
    // run search for transition bandwidth which gives these results
    let mut df0 = 1e-3;    // lower bound
    let mut df1 = 0.499;   // upper bound
    let mut df_hat = 0.0;  // transition bandwidth estimate
    let mut n_hat;         // filter length estimate

    for _ in 0..20 {
        // bisect limits
        df_hat = 0.5 * (df1 + df0);
        n_hat = if USE_KAISER_REQ_FILTER_LEN_ESTIMATE {
            estimate_req_filter_len_kaiser(df_hat, as_)?
        } else {
            estimate_req_filter_len_herrmann(df_hat, as_)?
        };
        // update limits
        if n_hat < n as f32 {
            df1 = df_hat;
        } else {
            df0 = df_hat;
        }
    }
    Ok(df_hat)
}   

/// estimate required filter length given transition bandwidth and
/// stop-band attenuation
///
/// # Arguments
/// * `df`     : transition bandwidth (0 < df < 0.5)
/// * `as_`    : stop-band attenuation \[dB\] (as_ > 0)
///
/// # Returns
/// 
/// Filter length
pub fn estimate_req_filter_len_kaiser(df: f32, as_: f32) -> Result<f32> {
    // [Vaidyanathan:1993]
    if df > 0.5 || df <= 0.0 {
        return Err(Error::Config(format!("cutoff frequency ({}) out of range (0, 0.5)", df)));
    }
    if as_ <= 0.0 {
        return Err(Error::Config("stopband attenuation must be greater than zero".into()));
    }   
    let h_len = (as_ - 7.95) / (14.26 * df);
    Ok(h_len)
}

/// estimate required filter length given transition bandwidth and
/// stop-band attenuation
///
/// # Arguments
/// * `df`     : transition bandwidth (0 < df < 0.5)
/// * `as_`    : stop-band attenuation \[dB\] (as_ > 0)
///
/// # Returns
/// 
/// Filter length
pub fn estimate_req_filter_len_herrmann(df: f32, as_: f32) -> Result<f32> {
    // [Herrmann:1973]
    if df > 0.5 || df <= 0.0 {
        return Err(Error::Config(format!("cutoff frequency ({}) out of range (0, 0.5)", df)));
    }
    if as_ <= 0.0 {
        return Err(Error::Config("stopband attenuation must be greater than zero".into()));
    }

    // Gaeddert's revisions:
    if as_ > 105.0 {
        return estimate_req_filter_len_kaiser(df, as_);
    }

    let as_ = as_ + 7.4;

    // compute delta_1, delta_2
    let d1 = 10.0f32.powf(-as_/20.0);
    let d2 = d1;

    // compute log of delta_1, delta_2
    let t1 = d1.log10();
    let t2 = d2.log10();

    // compute D_infinity(delta_1, delta_2)
    let dinf = (0.005309 * t1 * t1 + 0.07114 * t1 - 0.4761) * t2 -
               (0.002660 * t1 * t1 + 0.59410 * t1 + 0.4278);

    // compute f(delta_1, delta_2)
    let f = 11.012 + 0.51244 * (t1 - t2);

    // compute filter length estimate
    let h_len = (dinf - f * df * df) / df + 1.0;
    Ok(h_len)
}


/// Design FIR filter using generic window/taper method
///
/// # Arguments
/// * `wtype`  : window type, e.g. windows::WindowType::Hamming
/// * `n`      : filter length, n > 0
/// * `fc`     : cutoff frequency, 0 < fc < 0.5
/// * `arg`    : window-specific argument, if required
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_windowf(wtype: windows::WindowType, n: usize, fc: f32, arg: f32) -> Result<Vec<f32>> {
    // validate input
    if fc <= 0.0 || fc > 0.5 {
        return Err(Error::Config(format!("cutoff frequency ({}) out of range (0, 0.5)", fc)));
    }
    if n == 0 {
        return Err(Error::Config("filter length must be greater than zero".into()));
    }

    let mut h = vec![0.0; n];

    for i in 0..n {
        // time vector
        let t = i as f32 - (n as f32 - 1.0) / 2.0;

        // sinc prototype
        let h1 = sincf(2.0 * fc * t);

        // window
        let h2 = windows::window(wtype, i, n, arg)?;

        // composite
        h[i] = h1 * h2;
    }

    Ok(h)
}

/// Design finite impulse response notch filter
///
/// # Arguments
/// * `m`      : filter semi-length, m in \[1,1000\]
/// * `f0`     : filter notch frequency (normalized), -0.5 <= f0 <= 0.5
/// * `as_`    : stop-band attenuation \[dB\], as_ > 0
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_notch(m: usize, f0: f32, as_: f32) -> Result<Vec<f32>> {

    // validate inputs
    if m < 1 || m > 1000 {
        return Err(Error::Config(format!("filter semi-length ({}) out of range [1,1000]", m)));
    }
    if f0 < -0.5 || f0 > 0.5 {
        return Err(Error::Config(format!("notch frequency ({}) out of range [-0.5,0.5]", f0)));
    }
    if as_ <= 0.0 {
        return Err(Error::Config("stop-band attenuation must be greater than zero".into()));
    }

    let mut h = vec![0.0; 2*m+1];

    // choose kaiser beta parameter (approximate)
    let beta = kaiser::kaiser_beta_stopband_attenuation(as_);

    // design filter
    let mut scale = 0.0;
    for i in 0..h.len() {
        // tone at carrier frequency
        let p = -(2.0 * std::f32::consts::PI * f0 * ((i as f32) - (m as f32))).cos();

        // window
        let w = windows::kaiser(i, h.len(), beta)?;

        // save un-normalized filter
        h[i] = p * w;

        // accumulate scale
        scale += h[i] * p;
    }

    // normalize
    for i in 0..h.len() {
        h[i] /= scale;
    }

    // add impulse and return
    h[m] += 1.0;
    Ok(h)
}

/// Design (root-)Nyquist filter from prototype
///
/// # Arguments
/// * `ftype`  : filter type (e.g. FirdesFilterType::Rrcos)
/// * `k`      : samples/symbol
/// * `m`      : symbol delay
/// * `beta`   : excess bandwidth factor, beta in \[0,1\]
/// * `dt`     : fractional sample delay
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_prototype(ftype: FirFilterShape, k: usize, m: usize, beta: f32, dt: f32) -> Result<Vec<f32>> {
    // compute filter parameters
    let h_len = 2 * k * m + 1;
    let fc = 0.5 / k as f32;
    let df = beta / k as f32;
    let as_ = estimate_req_filter_stopband_attenuation(df, h_len)?;

    match ftype {
        FirFilterShape::Kaiser => {
            kaiser::fir_design_kaiser(h_len, fc, as_, dt)
        }
        FirFilterShape::Pm => {
            // Parks-McClellan algorithm parameters
            let bands = [0.0, fc-0.5*df, fc, fc, fc+0.5*df, 0.5];
            let des = [k as f32, 0.5 * k as f32, 0.0];
            let weights = [1.0, 1.0, 1.0];
            let wtype = [pm::FirPmWeightType::Flat, pm::FirPmWeightType::Flat, pm::FirPmWeightType::Flat];
            pm::fir_design_pm(h_len, 3, &bands, &des, Some(&weights), Some(&wtype), pm::FirPmBandType::Bandpass)
        }
        FirFilterShape::Rcos => {
            rcos::fir_design_rcos(k, m, beta, dt)
        }
        FirFilterShape::Fexp => {
            fnyquist::fir_design_fexp(k, m, beta, dt)
        }
        FirFilterShape::Fsech => {
            fnyquist::fir_design_fsech(k, m, beta, dt)
        }
        FirFilterShape::Farcsech => {
            fnyquist::fir_design_farcsech(k, m, beta, dt)
        }
        FirFilterShape::Arkaiser => {
            rkaiser::fir_design_arkaiser(k, m, beta, dt)
        }
        FirFilterShape::Rkaiser => {
            rkaiser::fir_design_rkaiser(k, m, beta, dt)
        }
        FirFilterShape::Rrcos => {
            rrcos::fir_design_rrcos(k, m, beta, dt)
        }
        FirFilterShape::Hm3 => {
            hm3::fir_design_hm3(k, m, beta, dt)
        }
        FirFilterShape::Gmsktx => {
            gmsk::fir_design_gmsktx(k, m, beta, dt)
        }
        FirFilterShape::Gmskrx => {
            gmsk::fir_design_gmskrx(k, m, beta, dt)
        }
        FirFilterShape::Rfexp => {
            fnyquist::fir_design_rfexp(k, m, beta, dt)
        }
        FirFilterShape::Rfsech => {
            fnyquist::fir_design_rfsech(k, m, beta, dt)
        }
        FirFilterShape::Rfarcsech => {
            fnyquist::fir_design_rfarcsech(k, m, beta, dt)
        }
    }
}

/// Design doppler filter
///
/// # Arguments
/// * `n`      : filter length
/// * `fd`     : normalized doppler frequency (0 < fd < 0.5)
/// * `k`      : Rice fading factor (k >= 0)
/// * `theta`  : LoS component angle of arrival
///
/// # Returns
/// 
/// Vec of filter coefficients
pub fn fir_design_doppler(n: usize, fd: f32, k: f32, theta: f32) -> Result<Vec<f32>> {
    let beta = 4.0;
    let mut h = vec![0.0; n];
    for i in 0..n {
        // time sample
        let t = i as f32 - (n as f32 - 1.0) / 2.0;

        // Bessel
        let j = 1.5 * crate::math::besselj0f((2.0 * std::f32::consts::PI * fd * t).abs());

        // Rice-K component
        let r = 1.5 * k / (k + 1.0) * (2.0 * std::f32::consts::PI * fd * t * theta.cos()).cos();

        // window
        let w = windows::kaiser(i, n, beta)?;

        // composite
        h[i] = (j + r) * w;
    }
    Ok(h)
}

/// Compute auto-correlation of filter at a specific lag
///
/// # Arguments
/// * `h`      : filter coefficients
/// * `lag`    : auto-correlation lag (samples)
///
/// # Returns
/// 
/// Auto-correlation value
pub fn filter_autocorr(h: &[f32], lag: isize) -> f32 {
    // auto-correlation is even symmetric
    let lag = lag.abs() as usize;

    // lag outside of filter length is zero
    if lag >= h.len() {
        return 0.0;
    }

    // compute auto-correlation
    let mut rxx = 0.0;
    for i in lag..h.len() {
        rxx += h[i] * h[i - lag];
    }
    rxx
}

/// Compute cross-correlation of two filters at a specific lag
///
/// # Arguments
/// * `h`      : filter coefficients
/// * `g`      : filter coefficients
/// * `lag`    : cross-correlation lag (samples)
///
/// # Returns
/// 
/// Cross-correlation value
pub fn filter_crosscorr(h: &[f32], g: &[f32], lag: isize) -> f32 {
    // cross-correlation is odd symmetric
    if h.len() < g.len() {
        return filter_crosscorr(g, h, -lag);
    }

    if lag <= -(g.len() as isize) {
        return 0.0;
    }
    if lag >= h.len() as isize {
        return 0.0;
    }

    let ig = if lag < 0 { -lag } else { 0 };
    let ih = if lag > 0 { lag } else { 0 };


    // compute length of overlap
    //     condition 1:             condition 2:          condition 3:
    //    [------ h ------]     [------ h ------]     [------ h ------]
    //  [-- g --]                    [-- g --]                  [-- g --]
    //   >|  n  |<                  >|   n   |<                >|  n  |<
    //
    let n = if lag < 0 {
        g.len() as isize + lag
    } else if lag < (h.len() as isize - g.len() as isize) {
        g.len() as isize
    } else {
        h.len() as isize - lag
    };

    let mut rxy = 0.0;
    for i in 0..n {
        rxy += h[(ih + i) as usize] * g[(ig + i) as usize];
    }
    rxy
}

/// Compute inter-symbol interference (ISI)--both RMS and
/// maximum--for the filter h.
///
/// # Arguments
/// * `h`      : filter coefficients [size: 2*k*m+1 x 1]
/// * `k`      : filter over-sampling rate (samples/symbol)
/// * `m`      : filter delay (symbols)
///
/// # Returns
/// 
/// A tuple of ISI RMS and maximum
pub fn filter_isi(h: &[f32], k: usize, m: usize) -> (f32, f32) {
    let rxx0 = filter_autocorr(h, 0);
    let mut isi_rms = 0.0;
    let mut isi_max = 0.0;
    for i in 1..2*m {
        let e = filter_autocorr(h, (i*k) as isize) / rxx0;
        let e = e.abs();
        isi_rms += e*e;
        if i == 1 || e > isi_max {
            isi_max = e;
        }
    }
    ((isi_rms / (2*m) as f32).sqrt(), isi_max)
}

/// Compute relative out-of-band energy
///
/// # Arguments
/// * `h`      : filter coefficients
/// * `fc`     : analysis cut-off frequency
/// * `nfft`   : fft size
///
/// # Returns
/// 
/// Relative out-of-band energy
pub fn filter_energy(h: &[f32], fc: f32, nfft: usize) -> Result<f32> {
    if fc < 0.0 || fc > 0.5 {
        return Err(Error::Config(format!("cutoff frequency ({}) out of range [0, 0.5]", fc)));
    }
    if h.is_empty() {
        return Err(Error::Config("filter coefficients must be non-empty".into()));
    }
    if nfft == 0 {
        return Err(Error::Config("fft size must be greater than zero".into()));
    }

    let mut expjwt = vec![Complex32::new(0.0, 0.0); h.len()];

    let mut e_total = 0.0;
    let mut e_stopband = 0.0;

    for i in 0..nfft {
        let f = 0.5 * (i as f32) / (nfft as f32);
        for k in 0..h.len() {
            expjwt[k] = Complex32::new(0.0, 2.0 * std::f32::consts::PI * f * k as f32).exp();
        }
        let v = expjwt.dotprod(h);
        let e2 = (v * v.conj()).re;
        e_total += e2;
        if f >= fc {
            e_stopband += e2;
        }
    }

    Ok(e_stopband / e_total)
}

/// Get static frequency response from filter coefficients at particular
/// frequency with real-valued coefficients
///
/// # Arguments
/// * `h`      : coefficients
/// * `fc`     : center frequency for analysis, -0.5 <= fc <= 0.5
///
/// # Returns
/// 
/// A frequency response value
pub fn freqrespf(h: &[f32], fc: f32) -> Result<Complex32> {
    freqresponse(h, fc)
}

/// Get static frequency response from filter coefficients at particular
/// frequency with complex coefficients
///
/// # Arguments
/// * `h`      : coefficients
/// * `fc`     : center frequency for analysis, -0.5 <= fc <= 0.5
///
/// # Returns
/// 
/// A frequency response value
pub fn freqrespcf(h: &[Complex32], fc: f32) -> Result<Complex32> {
    freqresponse(h, fc)
}

/// Get static frequency response from filter coefficients at particular
/// frequency with real or complex coefficients
///
/// # Arguments
/// * `h`      : coefficients
/// * `fc`     : center frequency for analysis, -0.5 <= fc <= 0.5
///
/// # Returns
/// 
/// A frequency response value
pub fn freqresponse<T: ComplexFloat>(h: &[T], fc: f32) -> Result<Complex32> where Complex32: From<T> {
    let mut h_res = Complex32::new(0.0, 0.0);
    let fc = fc as f64;
    for i in 0..h.len() {
        let expjwt = Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * fc * i as f64);
        let product = Complex32::from(h[i]) * Complex32::new(expjwt.re as f32, expjwt.im as f32);
        h_res += product;
    }
    Ok(h_res)
}

/// Compute group delay for a FIR filter
///
/// # Arguments
/// * `h`      : filter coefficients
/// * `n`      : filter length
/// * `fc`     : frequency at which delay is evaluated (-0.5 < fc < 0.5)
///
/// # Returns
/// 
/// Group delay value
pub fn fir_group_delay(h: &[f32], fc: f32) -> Result<f32> {
    // validate input
    if h.is_empty() {
        return Err(Error::Config("fir_group_delay(), length must be greater than zero".to_string()));
    } else if fc < -0.5 || fc > 0.5 {
        return Err(Error::Config("fir_group_delay(), _fc must be in [-0.5,0.5]".to_string()));
    }

    let mut t0 = Complex32::new(0.0, 0.0);
    let mut t1 = Complex32::new(0.0, 0.0);
    for (i, &h_i) in h.iter().enumerate() {
        let expjwt = Complex32::new(0.0, 2.0 * std::f32::consts::PI * fc * i as f32).exp();
        t0 += h_i * expjwt * i as f32;
        t1 += h_i * expjwt;
    }

    Ok((t0 / t1).re)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use test_macro::autotest_annotate;

    use crate::math::windows::WindowType;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_signalf};
    use crate::fft::{fft_run, Direction};

    fn test_harness_matched_filter(
        filter_type: FirFilterShape,
        k: usize,
        m: usize,
        beta: f32,
        tol_isi: f32,
        tol_as: f32,
    ) {
        // Create filter
        let mut h = fir_design_prototype(filter_type, k, m, beta, 0.0).unwrap();

        // scale by samples per symbol
        // TODO replace by liquid_vectorf_mulscalar when that's a thing
        for val in h.iter_mut() {
            *val *= 1.0 / k as f32;
        }

        // compute filter ISI
        let (isi_rms, isi_max) = filter_isi(&h, k, m);

        // ensure ISI is sufficiently small (log scale)
        assert!(20.0 * isi_max.log10() < tol_isi);
        assert!(20.0 * isi_rms.log10() < tol_isi);

        // verify spectrum response
        let regions = [
            PsdRegion { fmin: -0.50, fmax: -0.35, pmin:  0.0,  pmax: tol_as, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.20, fmax:  0.20, pmin: -1.0,  pmax: 1.0,    test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.35, fmax:  0.50, pmin:  0.0,  pmax: tol_as, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_signalf(&h, &regions).unwrap());
    }

    // test matched filter responses for square-root nyquist filter prototypes
    #[test]
    #[autotest_annotate(autotest_firdes_rrcos)]
    fn test_firdes_rrcos() {
        test_harness_matched_filter(FirFilterShape::Rrcos, 2, 10, 0.3, -60.0, -40.0);
    }

    #[test]
    #[autotest_annotate(autotest_firdes_rkaiser)]
    fn test_firdes_rkaiser() {
        test_harness_matched_filter(FirFilterShape::Rkaiser, 2, 10, 0.3, -60.0, -70.0);
    }

    #[test]
    #[autotest_annotate(autotest_firdes_arkaiser)]
    fn test_firdes_arkaiser() {
        test_harness_matched_filter(FirFilterShape::Arkaiser, 2, 10, 0.3, -60.0, -70.0);
    }

    #[test]
    #[autotest_annotate(autotest_liquid_firdes_dcblock)]
    fn test_liquid_firdes_dcblock() {
        // options
        let m: usize = 20;      // filter semi-length
        let as_: f32 = 60.0;    // stop-band suppression/pass-band ripple

        // Create filter
        let h = fir_design_notch(m, 0.0, as_).unwrap();

        // compute filter response and evaluate at several frequencies
        let nfft: usize = 1200;
        let mut buf_time = vec![Complex32::new(0.0, 0.0); nfft];
        let mut buf_freq = vec![Complex32::new(0.0, 0.0); nfft];

        for i in 0..nfft {
            buf_time[i] = Complex32::new(if i < h.len() { h[i] } else { 0.0 }, 0.0);
        }

        fft_run(&buf_time, &mut buf_freq, Direction::Forward);

        // evaluate at several points
        let tol = 2.0 * 10f32.powf(-as_ / 20.0); // generous
        assert_relative_eq!(buf_freq[0].norm(),            0.0, epsilon = tol);   // notch at DC
        assert_relative_eq!(buf_freq[nfft / 4].norm(),     1.0, epsilon = tol);   // pass at  Fs/4
        assert_relative_eq!(buf_freq[2 * nfft / 4].norm(), 1.0, epsilon = tol);   // pass at  Fs/2
        assert_relative_eq!(buf_freq[3 * nfft / 4].norm(), 1.0, epsilon = tol);   // pass at -Fs/4
    }

    #[test]
    #[autotest_annotate(autotest_liquid_firdes_notch)]
    fn test_liquid_firdes_notch() {
        // options
        let m: usize = 20;      // filter semi-length
        let as_: f32 = 60.0;    // stop-band suppression/pass-band ripple
        let f0: f32 = 0.2;      // notch frequency (must be greater than zero here)

        // Create filter
        let h = fir_design_notch(m, f0, as_).unwrap();

        // compute filter response and evaluate at several frequencies
        let nfft: usize = 1200;
        let mut buf_time = vec![Complex32::new(0.0, 0.0); nfft];
        let mut buf_freq = vec![Complex32::new(0.0, 0.0); nfft];

        for i in 0..nfft {
            buf_time[i] = Complex32::new(if i < h.len() { h[i] } else { 0.0 }, 0.0);
        }

        fft_run(&buf_time, &mut buf_freq, Direction::Forward);

        // indices to evaluate
        let i0 = (f0 * nfft as f32).round() as usize; // positive
        let i1 = nfft - i0;                           // negative

        // evaluate at several points
        let tol = 2.0 * 10f32.powf(-as_ / 20.0); // generous
        assert_relative_eq!(buf_freq[i0].norm(), 0.0, epsilon = tol);   // notch at +f0
        assert_relative_eq!(buf_freq[i1].norm(), 0.0, epsilon = tol);   // notch at -f0
        assert_relative_eq!(buf_freq[0].norm(), 1.0, epsilon = tol);    // pass at  0
        assert_relative_eq!(buf_freq[nfft/2].norm(), 1.0, epsilon = tol); // pass at  Fs/2
    }

    #[test]
    #[autotest_annotate(autotest_liquid_getopt_str2firfilt)]
    fn test_liquid_getopt_str2firfilt() {
        // TODO maybe allow an unknown filter type/enum
        // assert_eq!(liquid_getopt_str2firfilt("unknown").unwrap(), FirdesFilterType::Unknown);
        assert_eq!(FirFilterShape::from_str("kaiser").unwrap(), FirFilterShape::Kaiser);
        assert_eq!(FirFilterShape::from_str("pm").unwrap(), FirFilterShape::Pm);
        assert_eq!(FirFilterShape::from_str("rcos").unwrap(), FirFilterShape::Rcos);
        assert_eq!(FirFilterShape::from_str("fexp").unwrap(), FirFilterShape::Fexp);
        assert_eq!(FirFilterShape::from_str("fsech").unwrap(), FirFilterShape::Fsech);
        assert_eq!(FirFilterShape::from_str("farcsech").unwrap(), FirFilterShape::Farcsech);
        assert_eq!(FirFilterShape::from_str("arkaiser").unwrap(), FirFilterShape::Arkaiser);
        assert_eq!(FirFilterShape::from_str("rkaiser").unwrap(), FirFilterShape::Rkaiser);
        assert_eq!(FirFilterShape::from_str("rrcos").unwrap(), FirFilterShape::Rrcos);
        assert_eq!(FirFilterShape::from_str("hm3").unwrap(), FirFilterShape::Hm3);
        assert_eq!(FirFilterShape::from_str("gmsktx").unwrap(), FirFilterShape::Gmsktx);
        assert_eq!(FirFilterShape::from_str("gmskrx").unwrap(), FirFilterShape::Gmskrx);
        assert_eq!(FirFilterShape::from_str("rfexp").unwrap(), FirFilterShape::Rfexp);
        assert_eq!(FirFilterShape::from_str("rfsech").unwrap(), FirFilterShape::Rfsech);
        assert_eq!(FirFilterShape::from_str("rfarcsech").unwrap(), FirFilterShape::Rfarcsech);
    }

    #[test]
    #[autotest_annotate(autotest_liquid_firdes_config)]
    fn test_liquid_firdes_config() {
        // Check that estimate methods return zero for invalid configs
        assert!(estimate_req_filter_len(-0.1, 60.0).is_err()); // invalid transition band
        assert!(estimate_req_filter_len(0.0, 60.0).is_err());  // invalid transition band
        assert!(estimate_req_filter_len(0.6, 60.0).is_err());  // invalid transition band
        assert!(estimate_req_filter_len(0.2, -1.0).is_err());  // invalid stop-band suppression
        assert!(estimate_req_filter_len(0.2, 0.0).is_err());   // invalid stop-band suppression

        assert!(estimate_req_filter_len_kaiser(-0.1, 60.0).is_err()); // invalid transition band
        assert!(estimate_req_filter_len_kaiser(0.0, 60.0).is_err());  // invalid transition band
        assert!(estimate_req_filter_len_kaiser(0.6, 60.0).is_err());  // invalid transition band
        assert!(estimate_req_filter_len_kaiser(0.2, -1.0).is_err());  // invalid stop-band suppression
        assert!(estimate_req_filter_len_kaiser(0.2, 0.0).is_err());   // invalid stop-band suppression

        assert!(estimate_req_filter_len_herrmann(-0.1, 60.0).is_err()); // invalid transition band
        assert!(estimate_req_filter_len_herrmann(0.0, 60.0).is_err());  // invalid transition band
        assert!(estimate_req_filter_len_herrmann(0.6, 60.0).is_err());  // invalid transition band
        assert!(estimate_req_filter_len_herrmann(0.2, -1.0).is_err());  // invalid stop-band suppression
        assert!(estimate_req_filter_len_herrmann(0.2, 0.0).is_err());   // invalid stop-band suppression

        let m = 4;
        let h_len = 2 * m + 1;
        let wtype = WindowType::Hamming;

        assert!(fir_design_windowf(wtype, h_len, 0.2, 0.0).is_ok());
        assert!(fir_design_windowf(wtype, 0, 0.2, 0.0).is_err());
        assert!(fir_design_windowf(wtype, h_len, -0.1, 0.0).is_err());
        assert!(fir_design_windowf(wtype, h_len, 0.0, 0.0).is_err());
        assert!(fir_design_windowf(wtype, h_len, 0.6, 0.0).is_err());

        assert!(kaiser::fir_design_kaiser(h_len, 0.2, 60.0, 0.0).is_ok());
        assert!(kaiser::fir_design_kaiser(0, 0.2, 60.0, 0.0).is_err());
        assert!(kaiser::fir_design_kaiser(h_len, -0.1, 60.0, 0.0).is_err());
        assert!(kaiser::fir_design_kaiser(h_len, 0.0, 60.0, 0.0).is_err());
        assert!(kaiser::fir_design_kaiser(h_len, 0.6, 60.0, 0.0).is_err());
        assert!(kaiser::fir_design_kaiser(h_len, 0.2, 60.0, -0.7).is_err());
        assert!(kaiser::fir_design_kaiser(h_len, 0.2, 60.0, 0.7).is_err());

        assert!(fir_design_notch(m, 0.2, 60.0).is_ok());
        assert!(fir_design_notch(0, 0.2, 60.0).is_err());
        assert!(fir_design_notch(m, -0.7, 60.0).is_err());
        assert!(fir_design_notch(m, 0.7, 60.0).is_err());
        assert!(fir_design_notch(m, 0.2, -8.0).is_err());

        // no unknown filter type
        // assert!(liquid_firdes_prototype(FirdesFilterType::Unknown, 2, 2, 0.3, 0.0).is_err());

        // Test energy calculation configuration; design proper filter
        let h = fir_design_windowf(wtype, h_len, 0.2, 0.0).unwrap();
        assert!(filter_energy(&h, -0.1, 1200).is_err());
        assert!(filter_energy(&h, 0.7, 1200).is_err());
        assert!(filter_energy(&h, 0.3, 0).is_err());

        assert!(FirFilterShape::from_str("unknown-filter-type").is_err());
    }

    #[test]
    #[autotest_annotate(autotest_liquid_firdes_estimate)]
    fn test_liquid_firdes_estimate() {
        let tol = 0.05; // dB

        // Kaiser's method
        assert_relative_eq!(estimate_req_filter_len_kaiser(0.05, 60.0).unwrap(), 73.00140381, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_kaiser(0.10, 60.0).unwrap(), 36.50070190, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_kaiser(0.20, 60.0).unwrap(), 18.25035095, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_kaiser(0.30, 60.0).unwrap(), 12.16689968, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_kaiser(0.40, 60.0).unwrap(), 9.12517548, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_kaiser(0.05, 80.0).unwrap(), 101.05189514, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_kaiser(0.05, 100.0).unwrap(), 129.10238647, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_kaiser(0.05, 120.0).unwrap(), 157.15287781, max_relative = tol);

        // Herrmann's method
        assert_relative_eq!(estimate_req_filter_len_herrmann(0.05, 60.0).unwrap(), 75.51549530, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_herrmann(0.10, 60.0).unwrap(), 37.43184662, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_herrmann(0.20, 60.0).unwrap(), 17.56412315, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_herrmann(0.30, 60.0).unwrap(), 10.20741558, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_herrmann(0.40, 60.0).unwrap(), 5.97846174, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_herrmann(0.05, 80.0).unwrap(), 102.72290039, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_herrmann(0.05, 100.0).unwrap(), 129.88548279, max_relative = tol);
        assert_relative_eq!(estimate_req_filter_len_herrmann(0.05, 120.0).unwrap(), 157.15287781, max_relative = tol);
    }

    fn testbench_firdes_prototype(filter_type: &str, k: usize, m: usize, beta: f32, as_: f32) {
        // design filter
        let ftype = FirFilterShape::from_str(filter_type).unwrap();
        let mut h = fir_design_prototype(ftype, k, m, beta, 0.0).unwrap();

        // scale by samples per symbol
        // TODO use vectorf_mulscalar when it exists
        for v in h.iter_mut() {
            *v *= 1.0 / k as f32;
        }

        // verify spectrum
        let bw = 1.0 / k as f32;
        let f0 = 0.45 * bw * (1.0 - beta);
        let f1 = 0.55 * bw * (1.0 + beta);
        let regions = [
            PsdRegion { fmin: -0.5, fmax: -f1, pmin: 0.0,  pmax: -as_, test_lo: false, test_hi: true },
            PsdRegion { fmin: -f0,  fmax: f0,  pmin: -1.0, pmax: 1.0,  test_lo: true,  test_hi: true },
            PsdRegion { fmin: f1,   fmax: 0.5, pmin: 0.0,  pmax: -as_, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signalf(&h, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_kaiser)]
    fn test_firdes_prototype_kaiser() { testbench_firdes_prototype("kaiser", 4, 12, 0.3, 60.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_pm)]
    fn test_firdes_prototype_pm() { testbench_firdes_prototype("pm", 4, 12, 0.3, 80.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_rcos)]
    fn test_firdes_prototype_rcos() { testbench_firdes_prototype("rcos", 4, 12, 0.3, 60.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_fexp)]
    fn test_firdes_prototype_fexp() { testbench_firdes_prototype("fexp", 4, 12, 0.3, 40.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_fsech)]
    fn test_firdes_prototype_fsech() { testbench_firdes_prototype("fsech", 4, 12, 0.3, 60.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_farcsech)]
    fn test_firdes_prototype_farcsech() { testbench_firdes_prototype("farcsech", 4, 12, 0.3, 40.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_arkaiser)]
    fn test_firdes_prototype_arkaiser() { testbench_firdes_prototype("arkaiser", 4, 12, 0.3, 90.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_rkaiser)]
    fn test_firdes_prototype_rkaiser() { testbench_firdes_prototype("rkaiser", 4, 12, 0.3, 90.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_rrcos)]
    fn test_firdes_prototype_rrcos() { testbench_firdes_prototype("rrcos", 4, 12, 0.3, 45.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_hm3)]
    fn test_firdes_prototype_hm3() { testbench_firdes_prototype("hm3", 4, 12, 0.3, 100.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_rfexp)]
    fn test_firdes_prototype_rfexp() { testbench_firdes_prototype("rfexp", 4, 12, 0.3, 30.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_rfsech)]
    fn test_firdes_prototype_rfsech() { testbench_firdes_prototype("rfsech", 4, 12, 0.3, 40.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_prototype_rfarcsech)]
    fn test_firdes_prototype_rfarcsech() { testbench_firdes_prototype("rfarcsech", 4, 12, 0.3, 30.0); }

    // ignore gmsk filters as these weren't designed for flat pass-band responses
    // #[test]
    // #[autotest_annotate(xautotest_firdes_prototype_gmsktx)]
    // fn test_firdes_prototype_gmsktx() { testbench_firdes_prototype("gmsktx", 4, 12, 0.3, 60.0); }

    // #[test]
    // #[autotest_annotate(xautotest_firdes_prototype_gmskrx)]
    // fn test_firdes_prototype_gmskrx() { testbench_firdes_prototype("gmskrx", 4, 12, 0.3, 60.0); }

    #[test]
    #[autotest_annotate(autotest_firdes_doppler)]
    fn test_firdes_doppler() {
        // design filter
        let fd: f32 = 0.2;  // Normalized Doppler frequency
        let k: f32 = 10.0;  // Rice fading factor
        let theta: f32 = 0.0;  // LoS component angle of arrival
        let h_len: usize = 161;  // filter length
        let h = fir_design_doppler(h_len, fd, k, theta).unwrap();

        // verify resulting spectrum
        let regions = [
            PsdRegion { fmin: -0.5,   fmax: -0.25,  pmin:  0.0, pmax:  0.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.205, fmax: -0.195, pmin: 30.0, pmax: 40.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin: -0.14,  fmax:  0.14,  pmin:  6.0, pmax: 12.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.195, fmax:  0.205, pmin: 30.0, pmax: 40.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.25,  fmax:  0.5,   pmin:  0.0, pmax:  0.0, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signalf(&h, &regions).unwrap());
    }

    // check frequency response (real-valued coefficients)
    #[test]
    #[autotest_annotate(autotest_liquid_freqrespf)]
    fn test_liquid_freqrespf() {
        // design filter
        let h_len = 41;
        let h = kaiser::fir_design_kaiser(h_len, 0.27, 80.0, 0.3).unwrap();

        // compute frequency response with FFT
        let nfft = 400;
        let mut buf_time = vec![Complex32::new(0.0, 0.0); nfft];
        for i in 0..nfft {
            buf_time[i] = Complex32::new(if i < h_len { h[i] } else { 0.0 }, 0.0);
        }
        let mut buf_freq = vec![Complex32::new(0.0, 0.0); nfft];
        fft_run(&buf_time, &mut buf_freq, Direction::Forward);

        // compare to manual calculation
        let tol = 1e-5;
        for i in 0..nfft {
            let fc = (i as f32) / (nfft as f32) + if i >= nfft / 2 { -1.0 } else { 0.0 };
            let h_freq = freqrespf(&h, fc).unwrap();

            assert_relative_eq!(buf_freq[i].re, h_freq.re, epsilon = tol);
            assert_relative_eq!(buf_freq[i].im, h_freq.im, epsilon = tol);
        }
    }

    // check frequency response (complex-valued coefficients)
    #[test]
    #[autotest_annotate(autotest_liquid_freqrespcf)]
    fn test_liquid_freqrespcf() {
        // design filter and apply complex phasor
        let h_len = 41;
        let hf = kaiser::fir_design_kaiser(h_len, 0.27, 80.0, 0.3).unwrap();
        let mut h = vec![Complex32::new(0.0, 0.0); h_len];
        for i in 0..h_len {
            h[i] = hf[i] * Complex32::from_polar(1.0, 0.1 * (i * i) as f32);
        }

        // compute frequency response with FFT
        let nfft = 400;
        let mut buf_time = vec![Complex32::new(0.0, 0.0); nfft];
        for i in 0..nfft {
            buf_time[i] = if i < h_len { h[i] } else { Complex32::new(0.0, 0.0) };
        }
        let mut buf_freq = vec![Complex32::new(0.0, 0.0); nfft];
        fft_run(&buf_time, &mut buf_freq, Direction::Forward);

        // compare to manual calculation
        let tol = 1e-5;
        for i in 0..nfft {
            let fc = (i as f32) / (nfft as f32) + if i >= nfft / 2 { -1.0 } else { 0.0 };
            let h_freq = freqrespcf(&h, fc).unwrap();

            println!("i: {}, buf_freq[i]: {:?} + {:?}j, H: {:?} + {:?}j", i, buf_freq[i].re, buf_freq[i].im, h_freq.re, h_freq.im);
            assert_relative_eq!(buf_freq[i].re, h_freq.re, epsilon = tol);
            assert_relative_eq!(buf_freq[i].im, h_freq.im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_filter_crosscorr_rrrf)]
    fn test_filter_crosscorr_rrrf() {
        // options
        let tol = 1e-3;

        // input vectors
        let x_len = 16;
        let x: [f32; 16] = [
            0.25887000,   0.11752000,   0.67812000,  -1.02480000, 
            1.46750000,  -0.67462000,   0.93029000,   0.98751000, 
            0.00969890,   1.05300000,   1.38100000,   1.47540000, 
            1.14110000,  -0.39480000,  -0.30426000,   1.58190000
        ];

        let y_len = 8;
        let y: [f32; 8] = [
            -1.15920000,  -1.57390000,   0.65239000,  -0.54542000, 
            -0.97277000,   0.99115000,  -0.76247000,  -1.08210000
        ];

        // derived values
        let rxy_len = x_len + y_len - 1;
        let mut rxy = vec![0.0; rxy_len];
        let rxy_test: [f32; 23] = [
            -0.28013000,  -0.32455000,  -0.56685000,   0.45660000, 
            -0.39008000,  -1.95950000,   1.25850000,  -3.35780000, 
            -1.85760000,   1.07920000,  -5.31760000,  -2.18630000, 
            -2.05850000,  -3.52450000,  -0.90010000,  -4.55350000, 
            -4.17770000,  -1.09920000,  -5.13670000,  -1.76270000, 
            1.96850000,  -2.13700000,  -1.83370000
        ];

        // corr(x,y)
        for i in 0..rxy_len {
            let lag = i as isize - y_len as isize + 1;
            rxy[i] = filter_crosscorr(&x, &y, lag);
        }
        for i in 0..rxy_len {
            assert_relative_eq!(rxy[i], rxy_test[i], epsilon = tol);
        }

        // derived values
        let ryx_len = x_len + y_len - 1;
        let mut ryx = vec![0.0; ryx_len];
        let ryx_test: [f32; 23] = [
            -1.83370000,  -2.13700000,   1.96850000,  -1.76270000, 
            -5.13670000,  -1.09920000,  -4.17770000,  -4.55350000, 
            -0.90010000,  -3.52450000,  -2.05850000,  -2.18630000, 
            -5.31760000,   1.07920000,  -1.85760000,  -3.35780000, 
            1.25850000,  -1.95950000,  -0.39008000,   0.45660000, 
            -0.56685000,  -0.32455000,  -0.28013000
        ];
            
        // corr(y,x)
        for i in 0..ryx_len {
            let lag = i as isize - x_len as isize + 1;
            ryx[i] = filter_crosscorr(&y, &x, lag);
        }
        for i in 0..ryx_len {
            assert_relative_eq!(ryx[i], ryx_test[i], epsilon = tol);
        }
    }

}