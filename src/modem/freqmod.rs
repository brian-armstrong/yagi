use crate::error::{Error, Result};
use num_complex::Complex32;
use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct Freqmod {
    ref_: f32,                    // phase reference: kf*2^16
    sincos_table_len: usize,      // table length: 10 bits
    sincos_table_phase: u16,      // accumulated phase: 16 bits
    sincos_table: Vec<Complex32>, // sin|cos look-up table: 2^10 entries
}

impl Freqmod {
    pub fn new(kf: f32) -> Result<Self> {
        // Validate input
        if kf <= 0.0 {
            return Err(Error::Config(format!("modulation factor {:.4e} must be greater than 0", kf)));
        }

        let mut q = Self {
            ref_: kf * (1 << 16) as f32,
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

    pub fn reset(&mut self) -> Result<()> {
        self.sincos_table_phase = 0;
        Ok(())
    }

    pub fn modulate(&mut self, m: f32) -> Result<Complex32> {
        // Accumulate phase; this wraps around a 16-bit boundary and ensures
        // that negative numbers are mapped to positive numbers
        self.sincos_table_phase = (self.sincos_table_phase as i32 + (1 << 16) + (self.ref_ * m).round() as i32) as u16;

        // Compute table index: mask out 10 most significant bits with rounding
        // (adding 0x0020 effectively rounds to nearest value with 10 bits of precision)
        let index = ((self.sincos_table_phase as u32 + 0x0020) >> 6) & 0x03ff;

        // Return table value at index
        Ok(self.sincos_table[index as usize])
    }

    pub fn modulate_block(&mut self, m: &[f32], s: &mut [Complex32]) -> Result<()> {
        if m.len() != s.len() {
            return Err(Error::Range("input and output arrays must be same length".into()));
        }

        for (x, y) in m.iter().zip(s.iter_mut()) {
            *y = self.modulate(*x)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freqmod_create() {
        let result = Freqmod::new(0.5);
        assert!(result.is_ok());

        let result = Freqmod::new(0.0);
        assert!(result.is_err());

        let result = Freqmod::new(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_freqmod_modulate() -> Result<()> {
        let mut mod_ = Freqmod::new(0.5)?;

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
}
