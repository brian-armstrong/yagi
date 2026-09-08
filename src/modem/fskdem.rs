use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use num_complex::Complex32;

#[derive(Clone, Debug)]
pub struct Fskdem {
    k: usize,                 // samples per symbol
    m_size: usize,            // constellation size (M)
    k_size: usize,            // FFT size (K)
    buf_time: Vec<Complex32>, // FFT input buffer
    buf_freq: Vec<Complex32>, // FFT output buffer
    fft: Fft<f32>,            // FFT object
    demod_map: Vec<usize>,    // demodulation map
    s_demod: usize,           // demodulated symbol (used for frequency error)
}

impl Fskdem {
    pub fn new(m: usize, k: usize, bandwidth: f32) -> Result<Self> {
        // Validate input
        if m == 0 {
            return Err(Error::Config("bits/symbol must be greater than 0".into()));
        }
        if !(2..=2048).contains(&k) {
            return Err(Error::Config("samples/symbol must be in [2^m, 2048]".into()));
        }
        if !(0.0..0.5).contains(&bandwidth) {
            return Err(Error::Config("bandwidth must be in (0,0.5)".into()));
        }

        let m_size = 1 << m;
        let m2 = 0.5 * (m_size - 1) as f32;

        // Compute demodulation FFT size
        let df = bandwidth / m2;
        let mut err_min = 1e9f32;
        let k_min = k;
        let k_max = (k * 4).min(16);
        let mut k_size = k_min;

        for k_hat in k_min..=k_max {
            let v = 0.5 * df * k_hat as f32;
            let err = (v.round() - v).abs();

            if k_hat == k_min || err < err_min {
                k_size = k_hat;
                err_min = err;
            }

            // Perfect match; no need to continue searching
            if err < 1e-6 {
                break;
            }
        }

        // Determine demodulation mapping between tones and frequency bins
        let mut demod_map = vec![0; m_size];
        for (i, dm) in demod_map.iter_mut().enumerate() {
            let freq = ((i as f32) - m2) * bandwidth / m2;
            let idx = freq * k_size as f32;
            let index = if idx < 0.0 { (idx + k_size as f32).round() as usize } else { idx.round() as usize };
            *dm = index;
        }

        // Check for uniqueness
        for i in 1..m_size {
            if demod_map[i] == demod_map[i - 1] {
                return Err(Error::Config("demod map is not unique; consider increasing bandwidth".into()));
            }
        }

        let buf_time = vec![Complex32::new(0.0, 0.0); k_size];
        let buf_freq = vec![Complex32::new(0.0, 0.0); k_size];
        let fft = Fft::new(k_size, Direction::Forward);

        let mut q = Self { k, m_size, k_size, buf_time, buf_freq, fft, demod_map, s_demod: 0 };

        q.reset()?;
        Ok(q)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.buf_time.fill(Complex32::new(0.0, 0.0));
        self.buf_freq.fill(Complex32::new(0.0, 0.0));
        self.s_demod = 0;
        Ok(())
    }

    pub fn demodulate(&mut self, y: &[Complex32]) -> Result<usize> {
        if y.len() != self.k {
            return Err(Error::Range("input length must match samples/symbol".into()));
        }

        // Copy input to internal time buffer
        self.buf_time[..self.k].copy_from_slice(y);

        // Compute transform
        self.fft.run(&self.buf_time, &mut self.buf_freq);

        // Find maximum by looking at particular bins
        let mut vmax = 0.0;
        self.s_demod = 0;

        // Run search
        for s in 0..self.m_size {
            let v = self.buf_freq[self.demod_map[s]].norm();
            if s == 0 || v > vmax {
                self.s_demod = s;
                vmax = v;
            }
        }

        Ok(self.s_demod)
    }

    pub fn get_frequency_error(&self) -> f32 {
        // Get index of peak bin
        let vm = self.buf_freq[(self.s_demod + self.k_size - 1) % self.k_size].norm(); // previous
        let v0 = self.buf_freq[self.s_demod].norm(); // peak
        let vp = self.buf_freq[(self.s_demod + 1) % self.k_size].norm(); // post

        // Compute derivative
        (vp - vm) / v0
    }

    pub fn get_symbol_energy(&self, s: usize, range: usize) -> Result<f32> {
        if s >= self.m_size {
            return Err(Error::Range(format!("input symbol ({}) exceeds maximum ({})", s, self.m_size)));
        }

        let range = range.min(self.k_size);
        let index = self.demod_map[s];

        // Compute energy around FFT bin
        let mut energy = self.buf_freq[index].norm_sqr();

        for i in 1..=range {
            let i0 = (index + i) % self.k_size;
            let i1 = (index + self.k_size - i) % self.k_size;

            energy += self.buf_freq[i0].norm_sqr();
            energy += self.buf_freq[i1].norm_sqr();
        }

        Ok(energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modem::fskmod::Fskmod;
    use crate::random::randnf;
    use test_macro::autotest_annotate;

    #[test]
    fn test_fskdem_create() {
        let result = Fskdem::new(2, 8, 0.25);
        assert!(result.is_ok());

        let result = Fskdem::new(0, 8, 0.25);
        assert!(result.is_err());

        let result = Fskdem::new(2, 1, 0.25);
        assert!(result.is_err());

        let result = Fskdem::new(2, 8, 0.6);
        assert!(result.is_err());
    }

    #[test]
    fn test_fskdem_demodulate() -> Result<()> {
        let mut dem = Fskdem::new(2, 8, 0.25)?;
        let y = vec![Complex32::new(0.0, 0.0); 8];

        // Test valid demodulation
        assert!(dem.demodulate(&y).is_ok());

        // Test invalid input length
        let y_short = vec![Complex32::new(0.0, 0.0); 4];
        assert!(dem.demodulate(&y_short).is_err());

        Ok(())
    }

    // Helper function to keep code base small
    fn fskmodem_test_mod_demod(m: usize, k: usize, bandwidth: f32) -> Result<()> {
        // create modulator/demodulator pair
        let mut modulator = Fskmod::new(m, k, bandwidth)?;
        let mut demodulator = Fskdem::new(m, k, bandwidth)?;

        let m_size = 1 << m; // constellation size
        let mut buf = vec![Complex32::new(0.0, 0.0); k];

        // modulate, demodulate, count errors
        for sym_in in 0..m_size {
            // modulate
            modulator.modulate(sym_in, &mut buf)?;

            // demodulate
            let sym_out = demodulator.demodulate(&buf)?;

            // count errors
            assert_eq!(sym_in, sym_out);
        }

        Ok(())
    }

    // AUTOTESTS: basic properties: M=2^m, k = 2*M, bandwidth = 0.25
    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M2)]
    fn test_fskmodem_norm_m2() -> Result<()> {
        fskmodem_test_mod_demod(1, 4, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M4)]
    fn test_fskmodem_norm_m4() -> Result<()> {
        fskmodem_test_mod_demod(2, 8, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M8)]
    fn test_fskmodem_norm_m8() -> Result<()> {
        fskmodem_test_mod_demod(3, 16, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M16)]
    fn test_fskmodem_norm_m16() -> Result<()> {
        fskmodem_test_mod_demod(4, 32, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M32)]
    fn test_fskmodem_norm_m32() -> Result<()> {
        fskmodem_test_mod_demod(5, 64, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M64)]
    fn test_fskmodem_norm_m64() -> Result<()> {
        fskmodem_test_mod_demod(6, 128, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M128)]
    fn test_fskmodem_norm_m128() -> Result<()> {
        fskmodem_test_mod_demod(7, 256, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M256)]
    fn test_fskmodem_norm_m256() -> Result<()> {
        fskmodem_test_mod_demod(8, 512, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M512)]
    fn test_fskmodem_norm_m512() -> Result<()> {
        fskmodem_test_mod_demod(9, 1024, 0.25)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_norm_M1024)]
    fn test_fskmodem_norm_m1024() -> Result<()> {
        fskmodem_test_mod_demod(10, 2048, 0.25)
    }

    // AUTOTESTS: obscure properties: M=2^m, k not relative to M, bandwidth basically irrational
    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M2)]
    fn test_fskmodem_misc_m2() -> Result<()> {
        fskmodem_test_mod_demod(1, 5, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M4)]
    fn test_fskmodem_misc_m4() -> Result<()> {
        fskmodem_test_mod_demod(2, 10, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M8)]
    fn test_fskmodem_misc_m8() -> Result<()> {
        fskmodem_test_mod_demod(3, 20, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M16)]
    fn test_fskmodem_misc_m16() -> Result<()> {
        fskmodem_test_mod_demod(4, 30, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M32)]
    fn test_fskmodem_misc_m32() -> Result<()> {
        fskmodem_test_mod_demod(5, 60, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M64)]
    fn test_fskmodem_misc_m64() -> Result<()> {
        fskmodem_test_mod_demod(6, 100, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M128)]
    fn test_fskmodem_misc_m128() -> Result<()> {
        fskmodem_test_mod_demod(7, 200, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M256)]
    fn test_fskmodem_misc_m256() -> Result<()> {
        fskmodem_test_mod_demod(8, 500, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M512)]
    fn test_fskmodem_misc_m512() -> Result<()> {
        fskmodem_test_mod_demod(9, 1000, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskmodem_misc_M1024)]
    fn test_fskmodem_misc_m1024() -> Result<()> {
        fskmodem_test_mod_demod(10, 2000, 0.3721451)
    }

    #[test]
    #[autotest_annotate(autotest_fskdem_copy)]
    fn test_fskdem_copy() -> Result<()> {
        // options
        let m = 3; // bits per symbol
        let k = 200; // samples per symbol
        let bw = 0.2345; // occupied bandwidth

        // create modulator/demodulator pair
        let mut dem_orig = Fskdem::new(m, k, bw)?;

        let num_symbols = 96;
        let mut buf = vec![Complex32::new(0.0, 0.0); k];

        // run original object
        for _ in 0..num_symbols {
            // generate random signal and demodulate
            for j in 0..k {
                buf[j] = Complex32::new(randnf(), randnf());
            }
            dem_orig.demodulate(&buf)?;
        }

        // copy object
        let mut dem_copy = dem_orig.clone();

        // run through both objects and compare
        for _ in 0..num_symbols {
            // generate random signal and demodulate
            for j in 0..k {
                buf[j] = Complex32::new(randnf(), randnf());
            }
            let sym_orig = dem_orig.demodulate(&buf)?;
            let sym_copy = dem_copy.demodulate(&buf)?;
            // check result
            assert_eq!(sym_orig, sym_copy);
        }

        Ok(())
    }
}
