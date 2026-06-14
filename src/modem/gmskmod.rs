// Gauss minimum-shift keying modulator

use crate::error::{Error, Result};
use crate::filter::{FirFilterShape, FirInterpolationFilter};
use num_complex::Complex32;
use std::f32::consts::PI;

/// GMSK modulator
#[derive(Clone, Debug)]
pub struct GmskMod {
    k: usize,      // samples/symbol
    m: usize,      // symbol delay
    bt: f32,       // bandwidth/time product
    interp: FirInterpolationFilter<f32, f32>,
    theta: f32,    // phase state
    k_inv: f32,    // 1/k
}

impl GmskMod {
    /// Create GMSK modulator
    ///
    /// # Arguments
    ///
    /// * `k` - samples per symbol (must be >= 2)
    /// * `m` - filter delay in symbols (must be >= 1)
    /// * `bt` - bandwidth-time product (must be in (0, 1))
    pub fn new(k: usize, m: usize, bt: f32) -> Result<Self> {
        if k < 2 {
            return Err(Error::Config(
                "samples/symbol must be at least 2".into(),
            ));
        }
        if m < 1 {
            return Err(Error::Config(
                "symbol delay must be at least 1".into(),
            ));
        }
        if bt <= 0.0 || bt >= 1.0 {
            return Err(Error::Config(
                "bandwidth/time product must be in (0, 1)".into(),
            ));
        }

        let interp = FirInterpolationFilter::new_prototype(
            FirFilterShape::Gmsktx,
            k,
            m,
            bt,
            0.0,
        )?;

        let mut q = Self {
            k,
            m,
            bt,
            interp,
            theta: 0.0,
            k_inv: 1.0 / k as f32,
        };

        q.reset();
        Ok(q)
    }

    /// Reset modulator state
    pub fn reset(&mut self) {
        self.theta = 0.0;
        self.interp.reset();
    }

    /// Get samples per symbol
    pub fn get_k(&self) -> usize {
        self.k
    }

    /// Get filter delay in symbols
    pub fn get_m(&self) -> usize {
        self.m
    }

    /// Get bandwidth-time product
    pub fn get_bt(&self) -> f32 {
        self.bt
    }

    /// Modulate a single symbol
    ///
    /// # Arguments
    ///
    /// * `s` - input symbol (0 or 1)
    /// * `y` - output buffer (length k)
    pub fn modulate(&mut self, s: u8, y: &mut [Complex32]) -> Result<()> {
        if y.len() < self.k {
            return Err(Error::Config(format!(
                "output buffer too small: {} < {}",
                y.len(),
                self.k
            )));
        }

        // generate sample from symbol
        let x = if s == 0 { -self.k_inv } else { self.k_inv };

        // run interpolator
        let mut phi = vec![0.0f32; self.k];
        self.interp.execute(x, &mut phi)?;

        // integrate phase state
        for i in 0..self.k {
            self.theta += phi[i];

            // ensure phase in [-pi, pi]
            if self.theta > PI {
                self.theta -= 2.0 * PI;
            }
            if self.theta < -PI {
                self.theta += 2.0 * PI;
            }

            // compute output
            y[i] = Complex32::new(self.theta.cos(), self.theta.sin());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    fn test_gmskmod_config() {
        // invalid configurations
        assert!(GmskMod::new(1, 3, 0.25).is_err()); // k too small
        assert!(GmskMod::new(2, 0, 0.25).is_err()); // m too small
        assert!(GmskMod::new(2, 3, 0.0).is_err()); // bt too small
        assert!(GmskMod::new(2, 3, 1.0).is_err()); // bt too large

        // valid configuration
        let q = GmskMod::new(4, 3, 0.25).unwrap();
        assert_eq!(q.get_k(), 4);
        assert_eq!(q.get_m(), 3);
        assert!((q.get_bt() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_gmskmod_modulate() {
        let mut q = GmskMod::new(4, 3, 0.25).unwrap();
        let mut buf = vec![Complex32::new(0.0, 0.0); 4];

        // modulate a few symbols
        for s in [0u8, 1, 1, 0, 1, 0, 0, 1] {
            q.modulate(s, &mut buf).unwrap();

            // check output is on unit circle
            for sample in &buf {
                let mag = sample.norm();
                assert!(
                    (mag - 1.0).abs() < 1e-5,
                    "output should be on unit circle, got {}",
                    mag
                );
            }
        }
    }

    #[test]
    fn test_gmskmod_phase_continuity() {
        let mut q = GmskMod::new(4, 3, 0.25).unwrap();
        let mut buf = vec![Complex32::new(0.0, 0.0); 4];
        let mut prev_phase = 0.0f32;

        // modulate several symbols and check phase continuity
        for s in [0u8, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0] {
            q.modulate(s, &mut buf).unwrap();

            for sample in &buf {
                let phase = sample.arg();
                // phase should change smoothly (allow for wrapping)
                let diff = (phase - prev_phase).abs();
                let diff = diff.min((diff - 2.0 * PI).abs()).min((diff + 2.0 * PI).abs());
                assert!(
                    diff < 1.0,
                    "phase discontinuity detected: {} -> {} (diff={})",
                    prev_phase,
                    phase,
                    diff
                );
                prev_phase = phase;
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_gmskmod_copy)]
    fn autotest_gmskmod_copy() {
        use crate::sequence::MSequence;

        let k = 5;
        let m = 3;
        let bt = 0.2345;

        let mut mod_orig = GmskMod::new(k, m, bt).unwrap();

        let num_symbols = 16;
        let mut buf_orig = vec![Complex32::new(0.0, 0.0); k];
        let mut buf_copy = vec![Complex32::new(0.0, 0.0); k];
        let mut ms = MSequence::create_default(7).unwrap();

        // run original object
        for _ in 0..num_symbols {
            let s = ms.generate_symbol(1) as u8;
            mod_orig.modulate(s, &mut buf_orig).unwrap();
        }

        // copy object
        let mut mod_copy = mod_orig.clone();

        // run through both objects and compare
        for _ in 0..num_symbols {
            let s = ms.generate_symbol(1) as u8;
            mod_orig.modulate(s, &mut buf_orig).unwrap();
            mod_copy.modulate(s, &mut buf_copy).unwrap();

            for (a, b) in buf_orig.iter().zip(buf_copy.iter()) {
                assert!(
                    (a - b).norm() < 1e-6,
                    "copy mismatch: {:?} vs {:?}",
                    a,
                    b
                );
            }
        }
    }
}
