use num_complex::Complex;

use crate::error::{Error, Result};
use crate::fft::spgram::Spgram;
use crate::fft::{fft_run, Direction};
use crate::filter::{FirFilter, IirFilter};
use crate::math::nextpow2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PsdRegion {
    pub fmin: f32,
    pub fmax: f32,
    pub pmin: f32,
    pub pmax: f32,
    pub test_lo: bool,
    pub test_hi: bool,
}

pub fn validate_psd_spectrum(psd: &[f32], nfft: usize, regions: &[PsdRegion]) -> Result<bool> {
    let mut fail = vec![false; nfft];

    for region in regions.iter() {
        if region.fmin < -0.5 || region.fmax > 0.5 || region.fmin > region.fmax {
            return Err(Error::Config("invalid frequency range".into()));
        }

        for j in 0..nfft {
            // compute frequency value and check region
            let f = (j as f32) / (nfft as f32) - 0.5;
            if f < region.fmin || f > region.fmax {
                continue;
            }

            // test lower bound
            if region.test_lo && psd[j] < region.pmin {
                fail[j] = true;
            }

            // test upper bound
            if region.test_hi && psd[j] > region.pmax {
                fail[j] = true;
            }
        }
    }

    // Check if any failures occurred
    Ok(!fail.iter().any(|&x| x))
}

pub fn validate_psd_signal(buf: &[Complex<f32>], regions: &[PsdRegion]) -> Result<bool> {
    let buf_len = buf.len() as u32;
    let nfft = 4 << nextpow2(buf_len.max(64))?;
    let mut buf_time = vec![Complex::new(0.0, 0.0); nfft];
    let mut buf_freq = vec![Complex::new(0.0, 0.0); nfft];
    let mut buf_psd = vec![0.0; nfft];

    for i in 0..nfft {
        buf_time[i] = if i < buf.len() { buf[i] } else { Complex::new(0.0, 0.0) };
    }

    fft_run(&buf_time, &mut buf_freq, Direction::Forward);

    for i in 0..nfft {
        buf_psd[i] = 20.0 * buf_freq[(i + nfft / 2) % nfft].norm().log10();
    }

    validate_psd_spectrum(&buf_psd, nfft, regions)
}

pub fn validate_psd_signalf(buf: &[f32], regions: &[PsdRegion]) -> Result<bool> {
    let buf_cplx: Vec<Complex<f32>> = buf.iter().map(|&x| Complex::new(x, 0.0)).collect();
    validate_psd_signal(&buf_cplx, regions)
}

pub fn validate_psd_firfilt(
    firfilt: &FirFilter<Complex<f32>, f32>,
    nfft: usize,
    regions: &[PsdRegion],
) -> Result<bool> {
    let mut psd = vec![0.0; nfft];
    for i in 0..nfft {
        let f = (i as f32) / (nfft as f32) - 0.5;
        let h = firfilt.freqresponse(f);
        psd[i] = 20.0 * h.norm().log10();
    }

    validate_psd_spectrum(&psd, nfft, regions)
}

pub fn validate_psd_firfiltc(
    firfilt: &FirFilter<Complex<f32>, Complex<f32>>,
    nfft: usize,
    regions: &[PsdRegion],
) -> Result<bool> {
    let mut psd = vec![0.0; nfft];
    for i in 0..nfft {
        let f = (i as f32) / (nfft as f32) - 0.5;
        let h = firfilt.freqresponse(f);
        psd[i] = 20.0 * h.norm().log10();
    }

    validate_psd_spectrum(&psd, nfft, regions)
}

pub fn validate_psd_iirfilt(iirfilt: &IirFilter<f32, f32>, nfft: usize, regions: &[PsdRegion]) -> Result<bool> {
    let mut psd = vec![0.0; nfft];
    for i in 0..nfft {
        let f = (i as f32) / (nfft as f32) - 0.5;
        let h = iirfilt.freqresponse(f);
        psd[i] = 20.0 * h.norm().log10();
    }

    validate_psd_spectrum(&psd, nfft, regions)
}

pub fn validate_psd_spgramcf(spgram: &Spgram<Complex<f32>>, regions: &[PsdRegion]) -> Result<bool> {
    let nfft = spgram.get_nfft();
    let psd = spgram.get_psd();
    validate_psd_spectrum(&psd, nfft, regions)
}
