// Gauss minimum-shift keying modulator

use crate::error::{Error, Result};
use crate::filter::{FirFilterShape, FirInterpolationFilter};
use num_complex::Complex32;
use std::f32::consts::PI;

/// Gaussian minimum-shift keying modulator.
///
/// Each binary symbol changes the phase by `+pi/2` for a one or `-pi/2` for a
/// zero. A Gaussian pulse spreads each change over neighboring symbols, which
/// narrows the spectrum. The output has unit magnitude and continuous phase.
///
/// A [`GmskDemodulator`](super::GmskDemodulator) with the same settings
/// recovers the symbols `2 * filter_delay` symbols later.
#[derive(Clone, Debug)]
#[doc(alias = "GmskMod")]
pub struct GmskModulator {
    k: usize, // samples/symbol
    m: usize, // symbol delay
    bt: f32,  // bandwidth/time product
    interp: FirInterpolationFilter<f32, f32>,
    theta: f32, // phase state
    k_inv: f32, // 1/k
}

impl GmskModulator {
    /// Creates a GMSK modulator with its phase at zero.
    ///
    /// `filter_delay` is the pulse length on each side of its center, in
    /// symbols, and is also the delay the modulator adds.
    /// `bandwidth_time_product` sets the pulse width. Smaller values give a
    /// narrower spectrum and more overlap between symbols.
    ///
    /// # Errors
    ///
    /// Returns an error when `samples_per_symbol` is less than two,
    /// `filter_delay` is zero, or `bandwidth_time_product` is outside
    /// `(0, 1)`.
    pub fn new(samples_per_symbol: usize, filter_delay: usize, bandwidth_time_product: f32) -> Result<Self> {
        if samples_per_symbol < 2 {
            return Err(Error::Config("samples/symbol must be at least 2".into()));
        }
        if filter_delay < 1 {
            return Err(Error::Config("symbol delay must be at least 1".into()));
        }
        if bandwidth_time_product <= 0.0 || bandwidth_time_product >= 1.0 {
            return Err(Error::Config("bandwidth/time product must be in (0, 1)".into()));
        }

        let interp = FirInterpolationFilter::new_prototype(
            FirFilterShape::Gmsktx,
            samples_per_symbol,
            filter_delay,
            bandwidth_time_product,
            0.0,
        )?;

        let mut q = Self {
            k: samples_per_symbol,
            m: filter_delay,
            bt: bandwidth_time_product,
            interp,
            theta: 0.0,
            k_inv: 1.0 / samples_per_symbol as f32,
        };

        q.reset();
        Ok(q)
    }

    /// Clears the pulse-shaping history and sets the phase to zero.
    pub fn reset(&mut self) {
        self.theta = 0.0;
        self.interp.reset();
    }

    /// Returns the number of samples per symbol.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the filter delay in symbols.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Returns the bandwidth-time product.
    pub fn bt(&self) -> f32 {
        self.bt
    }

    /// Modulates one binary symbol into [`Self::k`] samples.
    ///
    /// A `symbol` of zero is a zero bit and any other value is a one bit. The
    /// samples go to the start of `output`, and the rest of `output` is left
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than [`Self::k`]. The
    /// modulator and `output` are unchanged on error.
    pub fn modulate(&mut self, symbol: u8, output: &mut [Complex32]) -> Result<()> {
        if output.len() < self.k {
            return Err(Error::Config(format!("output buffer too small: {} < {}", output.len(), self.k)));
        }

        // generate sample from symbol
        let x = if symbol == 0 { -self.k_inv } else { self.k_inv };

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
            output[i] = Complex32::new(self.theta.cos(), self.theta.sin());
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
        assert!(GmskModulator::new(1, 3, 0.25).is_err()); // k too small
        assert!(GmskModulator::new(2, 0, 0.25).is_err()); // m too small
        assert!(GmskModulator::new(2, 3, 0.0).is_err()); // bt too small
        assert!(GmskModulator::new(2, 3, 1.0).is_err()); // bt too large

        // valid configuration
        let q = GmskModulator::new(4, 3, 0.25).unwrap();
        assert_eq!(q.k(), 4);
        assert_eq!(q.m(), 3);
        assert!((q.bt() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_gmskmod_modulate() {
        let mut q = GmskModulator::new(4, 3, 0.25).unwrap();
        let mut buf = vec![Complex32::new(0.0, 0.0); 4];

        // modulate a few symbols
        for s in [0u8, 1, 1, 0, 1, 0, 0, 1] {
            q.modulate(s, &mut buf).unwrap();

            // check output is on unit circle
            for sample in &buf {
                let mag = sample.norm();
                assert!((mag - 1.0).abs() < 1e-5, "output should be on unit circle, got {}", mag);
            }
        }
    }

    #[test]
    fn test_gmskmod_phase_continuity() {
        let mut q = GmskModulator::new(4, 3, 0.25).unwrap();
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
                assert!(diff < 1.0, "phase discontinuity detected: {} -> {} (diff={})", prev_phase, phase, diff);
                prev_phase = phase;
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_gmskmod_copy)]
    fn autotest_gmskmod_copy() {
        use crate::sequence::MaximalLengthSequence;

        let k = 5;
        let m = 3;
        let bt = 0.2345;

        let mut mod_orig = GmskModulator::new(k, m, bt).unwrap();

        let num_symbols = 16;
        let mut buf_orig = vec![Complex32::new(0.0, 0.0); k];
        let mut buf_copy = vec![Complex32::new(0.0, 0.0); k];
        let mut ms = MaximalLengthSequence::from_degree(7).unwrap();

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
                assert!((a - b).norm() < 1e-6, "copy mismatch: {:?} vs {:?}", a, b);
            }
        }
    }
}
