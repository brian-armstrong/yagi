use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f32::consts::PI;

#[derive(Debug, Clone)]
#[doc(alias = "Freqdem")]
/// Frequency demodulator that recovers a real message.
///
/// Each output is the phase change from the previous input sample, divided
/// by `2*pi * modulation_sensitivity`. This undoes a
/// [`FrequencyModulator`](super::FrequencyModulator) with the same
/// sensitivity, with no delay. A phase change of more than half a turn per
/// sample is read as a smaller change of the opposite sign.
///
/// The input amplitude does not matter, but a zero sample has no phase. It
/// gives an output of zero, as does the sample after it.
pub struct FrequencyDemodulator {
    ref_: f32,          // 1/(2*pi*kf)
    r_prime: Complex32, // previous received sample
}

impl FrequencyDemodulator {
    /// Creates a frequency demodulator.
    ///
    /// `modulation_sensitivity` is the frequency deviation, in cycles per
    /// sample, that gives an output of one.
    ///
    /// # Errors
    ///
    /// Returns an error when `modulation_sensitivity` is zero or negative.
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

    /// Forgets the previous input sample.
    ///
    /// The next output is then zero. This never returns an error.
    pub fn reset(&mut self) -> Result<()> {
        self.r_prime = Complex32::new(0.0, 0.0);
        Ok(())
    }

    /// Demodulates one complex sample into one message sample.
    ///
    /// The first output after a reset is zero, because there is no previous
    /// sample to compare with. This never returns an error.
    pub fn demodulate(&mut self, input: Complex32) -> Result<f32> {
        // Compute phase difference and normalize by modulation index
        let output = (self.r_prime.conj() * input).arg() * self.ref_;

        // Save previous input sample
        self.r_prime = input;

        Ok(output)
    }

    /// Demodulates each sample of `input` into `output`.
    ///
    /// The results go to the first `input.len()` samples of `output`, and the
    /// rest of `output` is left unchanged. Returns `input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`. The
    /// demodulator and `output` are unchanged on error.
    pub fn demodulate_block(&mut self, input: &[Complex32], output: &mut [f32]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Range("output length must be at least the input length".into()));
        }

        for (input_sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = self.demodulate(*input_sample)?;
        }
        Ok(input.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modem::FrequencyModulator;
    use crate::utility::test_helpers::validate_block_execute;
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

    #[test]
    fn test_freqdem_block_contract() {
        let q = FrequencyDemodulator::new(0.5).unwrap();
        let input = [Complex32::new(1.0, -0.5), Complex32::new(0.25, 0.75), Complex32::new(-1.0, 0.0)];
        validate_block_execute(&q, &input, -9.0, |q, x, y| q.demodulate_block(x, y));
    }
}
