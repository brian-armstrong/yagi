use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f32::consts::PI;

#[derive(Debug, Clone)]
#[doc(alias = "Freqdem")]
pub struct FrequencyDemodulator {
    ref_: f32,          // 1/(2*pi*kf)
    r_prime: Complex32, // previous received sample
}

impl FrequencyDemodulator {
    pub fn new(modulation_sensitivity: f32) -> Result<Self> {
        // Validate input
        if modulation_sensitivity <= 0.0 {
            return Err(Error::Config(format!(
                "modulation factor {:.4e} must be greater than 0",
                modulation_sensitivity
            )));
        }

        let mut q = Self { ref_: 1.0 / (2.0 * PI * modulation_sensitivity), r_prime: Complex32::new(0.0, 0.0) };

        q.reset()?;
        Ok(q)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.r_prime = Complex32::new(0.0, 0.0);
        Ok(())
    }

    pub fn demodulate(&mut self, input: Complex32) -> Result<f32> {
        // Compute phase difference and normalize by modulation index
        let output = (self.r_prime.conj() * input).arg() * self.ref_;

        // Save previous input sample
        self.r_prime = input;

        Ok(output)
    }

    pub fn demodulate_block(&mut self, input: &[Complex32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::Range("input and output arrays must be same length".into()));
        }

        for (input_sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = self.demodulate(*input_sample)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modem::FrequencyModulator;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    fn test_freqdem_create() {
        let result = FrequencyDemodulator::new(0.5);
        assert!(result.is_ok());

        let result = FrequencyDemodulator::new(0.0);
        assert!(result.is_err());

        let result = FrequencyDemodulator::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_freqdem_demodulate() -> Result<()> {
        let mut dem = FrequencyDemodulator::new(0.5)?;

        // Test single sample demodulation
        let r = Complex32::new(1.0, 0.0);
        let result = dem.demodulate(r);
        assert!(result.is_ok());

        // Test block demodulation
        let r = vec![Complex32::new(1.0, 0.0); 10];
        let mut m = vec![0.0f32; 10];
        assert!(dem.demodulate_block(&r, &mut m).is_ok());

        // Test block demodulation with mismatched lengths
        let mut m_short = vec![0.0f32; 5];
        assert!(dem.demodulate_block(&r, &mut m_short).is_err());

        Ok(())
    }

    // Help function to keep code base small
    //  kf     :   modulation factor
    fn freqmodem_test(kf: f32) -> Result<()> {
        // options
        let num_samples = 1024;
        let tol = 5e-2f32;

        // create mod/demod objects
        let mut mod_ = FrequencyModulator::new(kf)?; // modulator
        let mut dem = FrequencyDemodulator::new(kf)?; // demodulator

        // allocate arrays
        let mut m = vec![0.0f32; num_samples]; // message signal
        let mut r = vec![Complex32::new(0.0, 0.0); num_samples]; // received signal (complex baseband)
        let mut y = vec![0.0f32; num_samples]; // demodulator output

        // generate message signal (sum of sines)
        for i in 0..num_samples {
            let i = i as f32;
            m[i as usize] = 0.3 * (2.0 * PI * 0.013 * i + 0.0).cos()
                + 0.2 * (2.0 * PI * 0.021 * i + 0.4).cos()
                + 0.4 * (2.0 * PI * 0.037 * i + 1.7).cos();
        }

        // modulate signal
        mod_.modulate_block(&m, &mut r)?;

        // demodulate signal
        dem.demodulate_block(&r, &mut y)?;

        // compare demodulated signal to original, skipping first sample
        for i in 1..num_samples {
            assert_abs_diff_eq!(y[i], m[i], epsilon = tol);
        }

        Ok(())
    }

    // AUTOTESTS: generic PSK
    #[test]
    #[autotest_annotate(autotest_freqmodem_kf_0_02)]
    fn test_freqmodem_kf_0_02() -> Result<()> {
        freqmodem_test(0.02)
    }

    #[test]
    #[autotest_annotate(autotest_freqmodem_kf_0_04)]
    fn test_freqmodem_kf_0_04() -> Result<()> {
        freqmodem_test(0.04)
    }

    #[test]
    #[autotest_annotate(autotest_freqmodem_kf_0_08)]
    fn test_freqmodem_kf_0_08() -> Result<()> {
        freqmodem_test(0.08)
    }
}
