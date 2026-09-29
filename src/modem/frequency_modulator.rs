use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f32::consts::PI;

#[derive(Debug, Clone)]
#[doc(alias = "Freqmod")]
/// Frequency modulator for a real message.
///
/// Each input sample `x` advances the phase by
/// `2*pi * modulation_sensitivity * x` radians, so the output frequency is
/// `modulation_sensitivity * x` cycles per sample. The output is a
/// unit-magnitude complex sample. Each phase advance is rounded to a multiple
/// of 1/65536 of a turn, and the output is then read from a 1024-entry table
/// at the nearest entry to the phase.
///
/// [`FrequencyDemodulator`](super::FrequencyDemodulator) with the same
/// sensitivity recovers the message, as long as the frequency stays within
/// half a cycle per sample.
pub struct FrequencyModulator {
    ref_: f32,                    // phase reference: kf*2^16
    sincos_table_len: usize,      // table length: 10 bits
    sincos_table_phase: u16,      // accumulated phase: 16 bits
    sincos_table: Vec<Complex32>, // sin|cos look-up table: 2^10 entries
}

impl FrequencyModulator {
    /// Creates a frequency modulator with its phase at zero.
    ///
    /// `modulation_sensitivity` is the frequency deviation, in cycles per
    /// sample, for an input of one.
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

        let mut q = Self {
            ref_: modulation_sensitivity * (1 << 16) as f32,
            sincos_table_len: 1024,
            sincos_table_phase: 0,
            sincos_table: Vec::with_capacity(1024),
        };

        // Initialize look-up table
        for i in 0..q.sincos_table_len {
            let phase = 2.0 * PI * i as f32 / q.sincos_table_len as f32;
            q.sincos_table.push(Complex32::from_polar(1.0, phase));
        }

        q.reset()?;
        Ok(q)
    }

    /// Sets the phase to zero.
    ///
    /// This never returns an error.
    pub fn reset(&mut self) -> Result<()> {
        self.sincos_table_phase = 0;
        Ok(())
    }

    /// Advances the phase by one message sample and returns the output
    /// sample at the new phase.
    ///
    /// This never returns an error.
    ///
    /// # Panics
    ///
    /// Panics when `modulation_sensitivity * input` is about 32766 or more and
    /// overflow checks are enabled. Without overflow checks, the phase
    /// advance is still correct up to 32768 and is clamped above that.
    pub fn modulate(&mut self, input: f32) -> Result<Complex32> {
        // Accumulate phase; this wraps around a 16-bit boundary and ensures
        // that negative numbers are mapped to positive numbers
        self.sincos_table_phase =
            (self.sincos_table_phase as i32 + (1 << 16) + (self.ref_ * input).round() as i32) as u16;

        // Compute table index: mask out 10 most significant bits with rounding
        // (adding 0x0020 effectively rounds to nearest value with 10 bits of precision)
        let index = ((self.sincos_table_phase as u32 + 0x0020) >> 6) & 0x03ff;

        // Return table value at index
        Ok(self.sincos_table[index as usize])
    }

    /// Modulates each sample of `input` into `output`.
    ///
    /// The results go to the first `input.len()` samples of `output`, and the
    /// rest of `output` is left unchanged. Returns `input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`. The modulator
    /// and `output` are unchanged on error.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::modulate`].
    pub fn modulate_block(&mut self, input: &[f32], output: &mut [Complex32]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Range("output length must be at least the input length".into()));
        }

        for (input_sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = self.modulate(*input_sample)?;
        }
        Ok(input.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::test_helpers::validate_block_execute;

    #[test]
    fn test_freqmod_create() {
        let result = FrequencyModulator::new(0.5);
        assert!(result.is_ok());

        let result = FrequencyModulator::new(0.0);
        assert!(result.is_err());

        let result = FrequencyModulator::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_freqmod_modulate() -> Result<()> {
        let mut mod_ = FrequencyModulator::new(0.5)?;

        // Test single sample modulation
        let result = mod_.modulate(1.0);
        assert!(result.is_ok());

        // Test block modulation
        let m = vec![0.0f32; 10];
        let mut s = vec![Complex32::new(0.0, 0.0); 10];
        assert!(mod_.modulate_block(&m, &mut s).is_ok());

        // Test block modulation with mismatched lengths
        let mut s_short = vec![Complex32::new(0.0, 0.0); 5];
        assert!(mod_.modulate_block(&m, &mut s_short).is_err());

        Ok(())
    }

    #[test]
    fn test_freqmod_block_contract() {
        let q = FrequencyModulator::new(0.5).unwrap();
        validate_block_execute(&q, &[0.25, -0.5, 1.0], Complex32::new(-9.0, -9.0), |q, x, y| q.modulate_block(x, y));
    }
}
