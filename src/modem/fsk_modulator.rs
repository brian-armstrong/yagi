use crate::error::{Error, Result};
use crate::nco::{Nco, NcoBackend};
use num_complex::Complex32;
use std::f32::consts::PI;

#[derive(Clone, Debug)]
#[doc(alias = "Fskmod")]
/// M-ary frequency-shift keying modulator.
///
/// Each symbol is sent as a tone lasting `samples_per_symbol` samples. The
/// `M = 2^bits_per_symbol` tones are evenly spaced from `-bandwidth` to
/// `+bandwidth` cycles per sample, with symbol 0 at the lowest. The phase is
/// continuous from one symbol to the next. Output samples have magnitude
/// close to one.
///
/// Use an [`FskDemodulator`](super::FskDemodulator) with the same settings to
/// recover the symbols.
pub struct FskModulator {
    k: usize,        // samples per symbol
    bandwidth: f32,  // filter bandwidth parameter
    m_size: usize,   // constellation size (M)
    m2: f32,         // (M-1)/2
    oscillator: Nco, // nco
}

impl FskModulator {
    /// Creates an FSK modulator with its phase at zero.
    ///
    /// `bandwidth` is the frequency of the outermost tones, in cycles per
    /// sample, so the tones span `2 * bandwidth`.
    ///
    /// # Errors
    ///
    /// Returns an error when `bits_per_symbol` is zero, `samples_per_symbol`
    /// is outside `2..=2048`, or `bandwidth` is outside `(0, 0.5)`.
    ///
    /// # Panics
    ///
    /// Panics when `bits_per_symbol` is at least the width of `usize` and
    /// overflow checks are enabled. Without overflow checks, the shift wraps
    /// and the constellation size is wrong.
    pub fn new(bits_per_symbol: usize, samples_per_symbol: usize, bandwidth: f32) -> Result<Self> {
        // Validate input
        if bits_per_symbol == 0 {
            return Err(Error::Config("bits/symbol must be greater than 0".into()));
        }
        if !(2..=2048).contains(&samples_per_symbol) {
            return Err(Error::Config("samples/symbol must be in [2^m, 2048]".into()));
        }
        if !(bandwidth > 0.0 && bandwidth < 0.5) {
            return Err(Error::Config("bandwidth must be in (0,0.5)".into()));
        }

        let m = bits_per_symbol;
        let k = samples_per_symbol;

        let m_size = 1 << m;
        let m2 = 0.5 * (m_size - 1) as f32;

        let mut q = Self { k, bandwidth, m_size, m2, oscillator: Nco::new(NcoBackend::InterpolatedLookupTable) };

        q.reset()?;
        Ok(q)
    }

    /// Sets the phase to zero.
    ///
    /// This never returns an error.
    pub fn reset(&mut self) -> Result<()> {
        self.oscillator.reset();
        Ok(())
    }

    /// Writes the tone for `symbol` to `output`.
    ///
    /// `output` must hold exactly `samples_per_symbol` samples. The phase
    /// carries over from the previous call.
    ///
    /// # Errors
    ///
    /// Returns an error when `symbol` is `2^bits_per_symbol` or more, or the
    /// length of `output` is not `samples_per_symbol`. The modulator and
    /// `output` are unchanged on error.
    pub fn modulate(&mut self, symbol: usize, output: &mut [Complex32]) -> Result<()> {
        // Validate input
        if symbol >= self.m_size {
            return Err(Error::Range(format!("input symbol ({}) exceeds maximum ({})", symbol, self.m_size)));
        }
        if output.len() != self.k {
            return Err(Error::Range(format!(
                "output buffer length ({}) must match samples/symbol ({})",
                output.len(),
                self.k
            )));
        }

        // Compute appropriate frequency
        let dphi = ((symbol as f32) - self.m2) * 2.0 * PI * self.bandwidth / self.m2;

        // Set frequency appropriately
        self.oscillator.set_frequency(dphi);

        // Generate output tone
        for yi in &mut output[..self.k] {
            // Compute complex output
            *yi = self.oscillator.cexp();

            // Step oscillator
            self.oscillator.step();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    fn test_fskmod_create() {
        let result = FskModulator::new(2, 8, 0.25);
        assert!(result.is_ok());

        let result = FskModulator::new(0, 8, 0.25);
        assert!(result.is_err());

        let result = FskModulator::new(2, 1, 0.25);
        assert!(result.is_err());

        let result = FskModulator::new(2, 8, 0.6);
        assert!(result.is_err());

        assert!(FskModulator::new(2, 8, 0.0).is_err());
        assert!(FskModulator::new(2, 8, -0.0).is_err());
        assert!(FskModulator::new(2, 8, f32::NAN).is_err());
        assert!(FskModulator::new(2, 8, 0.001).is_ok());
    }

    #[test]
    fn test_fskmod_modulate() -> Result<()> {
        let mut mod_ = FskModulator::new(2, 8, 0.25)?;
        let mut y = vec![Complex32::new(0.0, 0.0); 8];

        // Test valid symbol
        assert!(mod_.modulate(0, &mut y).is_ok());

        // Test invalid symbol
        assert!(mod_.modulate(4, &mut y).is_err());

        // Test invalid output buffer length
        let mut y_short = vec![Complex32::new(0.0, 0.0); 4];
        assert!(mod_.modulate(0, &mut y_short).is_err());

        Ok(())
    }

    #[test]
    #[autotest_annotate(autotest_fskmod_copy)]
    fn test_fskmod_copy() -> Result<()> {
        // options
        let m = 3; // bits per symbol
        let k = 200; // samples per symbol
        let bw = 0.2345; // occupied bandwidth

        // create modulator/demodulator pair
        let mut mod_orig = FskModulator::new(m, k, bw)?;

        let num_symbols = 96;
        let mut buf_orig = vec![Complex32::new(0.0, 0.0); k];
        let mut buf_copy = vec![Complex32::new(0.0, 0.0); k];
        let mut ms = crate::sequence::MaximalLengthSequence::from_degree(7)?;

        // run original object
        for _ in 0..num_symbols {
            // generate random symbol and modulate
            let s = ms.generate_symbol(m as u32) as usize;
            mod_orig.modulate(s, &mut buf_orig)?;
        }

        // copy object
        let mut mod_copy = mod_orig.clone();

        // run through both objects and compare
        for _ in 0..num_symbols {
            // generate random symbol and modulate
            let s = ms.generate_symbol(m as u32) as usize;
            mod_orig.modulate(s, &mut buf_orig)?;
            mod_copy.modulate(s, &mut buf_copy)?;
            // check result
            assert_eq!(buf_orig, buf_copy);
        }

        Ok(())
    }
}
