use crate::error::{Error, Result};
use crate::nco::{Osc, OscScheme};
use num_complex::Complex32;
use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct Fskmod {
    k: usize,        // samples per symbol
    bandwidth: f32,  // filter bandwidth parameter
    m_size: usize,   // constellation size (M)
    m2: f32,         // (M-1)/2
    oscillator: Osc, // nco
}

impl Fskmod {
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

        let mut q = Self { k, bandwidth, m_size, m2, oscillator: Osc::new(OscScheme::Vco) };

        q.reset()?;
        Ok(q)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.oscillator.reset();
        Ok(())
    }

    pub fn modulate(&mut self, s: usize, y: &mut [Complex32]) -> Result<()> {
        // Validate input
        if s >= self.m_size {
            return Err(Error::Range(format!("input symbol ({}) exceeds maximum ({})", s, self.m_size)));
        }
        if y.len() != self.k {
            return Err(Error::Range(format!(
                "output buffer length ({}) must match samples/symbol ({})",
                y.len(),
                self.k
            )));
        }

        // Compute appropriate frequency
        let dphi = ((s as f32) - self.m2) * 2.0 * PI * self.bandwidth / self.m2;

        // Set frequency appropriately
        self.oscillator.set_frequency(dphi);

        // Generate output tone
        for yi in &mut y[..self.k] {
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
        let result = Fskmod::new(2, 8, 0.25);
        assert!(result.is_ok());

        let result = Fskmod::new(0, 8, 0.25);
        assert!(result.is_err());

        let result = Fskmod::new(2, 1, 0.25);
        assert!(result.is_err());

        let result = Fskmod::new(2, 8, 0.6);
        assert!(result.is_err());
    }

    #[test]
    fn test_fskmod_modulate() -> Result<()> {
        let mut mod_ = Fskmod::new(2, 8, 0.25)?;
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
        let mut mod_orig = Fskmod::new(m, k, bw)?;

        let num_symbols = 96;
        let mut buf_orig = vec![Complex32::new(0.0, 0.0); k];
        let mut buf_copy = vec![Complex32::new(0.0, 0.0); k];
        let mut ms = crate::sequence::MSequence::from_degree(7)?;

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
