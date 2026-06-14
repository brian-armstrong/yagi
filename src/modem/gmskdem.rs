// Gauss minimum-shift keying demodulator
//
// Two implementations are available:
// 1. Fixed matched filter (default) - uses FirFilter with GMSK receive filter
// 2. Adaptive equalizer - uses Eqlms for decision-directed equalization
//
// The equalizer variant can be enabled by setting GMSKDEM_USE_EQUALIZER to true.

use crate::error::{Error, Result};
use crate::equalization::eqlms::Eqlms;
use crate::filter::{FirFilterShape, FirFilter};
use crate::filter::fir_design_prototype;
use num_complex::Complex32;

// Set to true to use adaptive equalizer instead of fixed matched filter.
// This matches liquid-dsp's GMSKDEM_USE_EQUALIZER preprocessor flag.
const GMSKDEM_USE_EQUALIZER: bool = false;

/// Internal filter state - either fixed FIR or adaptive equalizer
#[derive(Clone, Debug)]
enum FilterState {
    /// Fixed matched filter
    Fir(FirFilter<f32, f32>),
    /// Adaptive equalizer
    Equalizer(Eqlms<f32>),
}

/// GMSK demodulator
#[derive(Clone, Debug)]
pub struct GmskDem {
    k: usize,                // samples/symbol
    m: usize,                // symbol delay
    bt: f32,                 // bandwidth/time product
    k_inv: f32,              // 1/k (for equalizer training)
    filter: FilterState,
    x_prime: Complex32,      // received signal state
    num_symbols_demod: u64,
}

impl GmskDem {
    /// Create GMSK demodulator
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

        let filter = if GMSKDEM_USE_EQUALIZER {
            let mut eq = Eqlms::<f32>::new_rnyquist(FirFilterShape::Gmskrx, k, m, bt, 0.0)?;
            eq.set_bw(0.01)?; // default learning rate
            FilterState::Equalizer(eq)
        } else {
            let h = fir_design_prototype(FirFilterShape::Gmskrx, k, m, bt, 0.0)?;
            FilterState::Fir(FirFilter::new(&h)?)
        };

        let mut q = Self {
            k,
            m,
            bt,
            k_inv: 1.0 / k as f32,
            filter,
            x_prime: Complex32::new(0.0, 0.0),
            num_symbols_demod: 0,
        };

        q.reset();
        Ok(q)
    }

    /// Reset demodulator state
    pub fn reset(&mut self) {
        self.x_prime = Complex32::new(0.0, 0.0);
        self.num_symbols_demod = 0;

        match &mut self.filter {
            FilterState::Fir(f) => f.reset(),
            FilterState::Equalizer(eq) => eq.reset(),
        }
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

    /// Get number of symbols demodulated
    pub fn get_num_symbols_demod(&self) -> u64 {
        self.num_symbols_demod
    }

    /// Set equalizer bandwidth (learning rate)
    ///
    /// Only effective when using equalizer mode.
    pub fn set_eq_bw(&mut self, bw: f32) -> Result<()> {
        if bw < 0.0 || bw > 0.5 {
            return Err(Error::Config(
                "bandwidth must be in [0, 0.5]".into(),
            ));
        }

        match &mut self.filter {
            FilterState::Equalizer(eq) => eq.set_bw(bw),
            FilterState::Fir(_) => {
                Err(Error::Config(
                    "equalizer is disabled".into(),
                ))
            }
        }
    }

    /// Get equalizer bandwidth (learning rate)
    pub fn get_eq_bw(&self) -> Option<f32> {
        match &self.filter {
            FilterState::Equalizer(eq) => Some(eq.get_bw()),
            FilterState::Fir(_) => None,
        }
    }

    /// Check if using equalizer mode
    pub fn uses_equalizer(&self) -> bool {
        matches!(self.filter, FilterState::Equalizer(_))
    }

    /// Demodulate k samples to produce one symbol
    ///
    /// # Arguments
    ///
    /// * `x` - input buffer (length k)
    ///
    /// # Returns
    ///
    /// Demodulated symbol (0 or 1)
    pub fn demodulate(&mut self, x: &[Complex32]) -> Result<u8> {
        if x.len() < self.k {
            return Err(Error::Config(format!(
                "input buffer too small: {} < {}",
                x.len(),
                self.k
            )));
        }

        self.num_symbols_demod += 1;

        let d_hat = match &mut self.filter {
            FilterState::Fir(filter) => {
                let mut d_hat = 0.0f32;
                for i in 0..self.k {
                    // compute phase difference
                    let phi = (self.x_prime.conj() * x[i]).arg();
                    self.x_prime = x[i];

                    // run through matched filter
                    filter.push(phi);

                    // decimate by k - only compute output at symbol boundary
                    if i == 0 {
                        d_hat = filter.execute();
                    }
                }
                d_hat
            }
            FilterState::Equalizer(eq) => {
                let mut d_hat = 0.0f32;
                for i in 0..self.k {
                    // compute phase difference
                    let phi = (self.x_prime.conj() * x[i]).arg();
                    self.x_prime = x[i];

                    // run through equalizer
                    eq.push(phi);

                    // decimate by k - only compute output at symbol boundary
                    if i == 0 {
                        d_hat = eq.execute().unwrap_or(0.0);
                    }
                }

                // update equalizer weights after appropriate delay
                if self.num_symbols_demod >= 2 * self.m as u64 {
                    // compute expected output, scaling by samples/symbol
                    let d_prime = if d_hat > 0.0 { self.k_inv } else { -self.k_inv };
                    eq.step(d_prime, d_hat);
                }

                d_hat
            }
        };

        // make decision
        Ok(if d_hat > 0.0 { 1 } else { 0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modem::gmskmod::GmskMod;
    use crate::sequence::MSequence;
    use test_macro::autotest_annotate;

    #[test]
    fn test_gmskdem_config() {
        // invalid configurations
        assert!(GmskDem::new(1, 3, 0.25).is_err()); // k too small
        assert!(GmskDem::new(2, 0, 0.25).is_err()); // m too small
        assert!(GmskDem::new(2, 3, 0.0).is_err()); // bt too small
        assert!(GmskDem::new(2, 3, 1.0).is_err()); // bt too large

        // valid configuration
        let q = GmskDem::new(4, 3, 0.25).unwrap();
        assert_eq!(q.get_k(), 4);
        assert_eq!(q.get_m(), 3);
        assert!((q.get_bt() - 0.25).abs() < 1e-6);
        // Note: uses_equalizer() depends on GMSKDEM_USE_EQUALIZER const
        assert_eq!(q.uses_equalizer(), GMSKDEM_USE_EQUALIZER);
    }

    #[test]
    fn test_gmskdem_eq_bw() {
        let mut q = GmskDem::new(4, 3, 0.25).unwrap();

        if q.uses_equalizer() {
            // With equalizer, set_eq_bw should succeed
            assert!(q.set_eq_bw(0.01).is_ok());
            assert!((q.get_eq_bw().unwrap() - 0.01).abs() < 1e-6);
        } else {
            // With FIR filter, set_eq_bw should fail
            assert!(q.set_eq_bw(0.01).is_err());
            assert!(q.get_eq_bw().is_none());
        }
    }

    fn testbench_gmskmodem(k: usize, m: usize, bt: f32) {
        let mut modulator = GmskMod::new(k, m, bt).unwrap();
        let mut demodulator = GmskDem::new(k, m, bt).unwrap();

        let delay = m + m;
        let num_symbols = 80 + delay;

        let mut ms = MSequence::create_default(7).unwrap();
        let mut buf = vec![Complex32::new(0.0, 0.0); k];
        let mut sym_in = vec![0u8; num_symbols];
        let mut sym_out = vec![0u8; num_symbols];

        for i in 0..num_symbols {
            // generate random symbol
            sym_in[i] = ms.generate_symbol(1) as u8;

            // modulate
            modulator.modulate(sym_in[i], &mut buf).unwrap();

            // demodulate
            sym_out[i] = demodulator.demodulate(&buf).unwrap();
        }

        // count errors (accounting for delay)
        for i in delay..num_symbols {
            assert_eq!(
                sym_in[i - delay],
                sym_out[i],
                "symbol mismatch at index {}: expected {}, got {}",
                i,
                sym_in[i - delay],
                sym_out[i]
            );
        }
    }

    // base configuration
    #[test]
    #[autotest_annotate(autotest_gmskmodem_k4_m3_b025)]
    fn test_gmskmodem_k4_m3_b025() {
        testbench_gmskmodem(4, 3, 0.25);
    }

    // test different samples/symbol
    #[test]
    #[autotest_annotate(autotest_gmskmodem_k2_m3_b025)]
    fn test_gmskmodem_k2_m3_b025() {
        testbench_gmskmodem(2, 3, 0.25);
    }

    #[test]
    #[autotest_annotate(autotest_gmskmodem_k3_m3_b025)]
    fn test_gmskmodem_k3_m3_b025() {
        testbench_gmskmodem(3, 3, 0.25);
    }

    #[test]
    #[autotest_annotate(autotest_gmskmodem_k5_m3_b025)]
    fn test_gmskmodem_k5_m3_b025() {
        testbench_gmskmodem(5, 3, 0.25);
    }

    #[test]
    #[autotest_annotate(autotest_gmskmodem_k8_m3_b033)]
    fn test_gmskmodem_k8_m3_b033() {
        testbench_gmskmodem(8, 3, 0.25);
    }

    // test different filter semi-lengths
    #[test]
    #[autotest_annotate(autotest_gmskmodem_k4_m1_b025)]
    fn test_gmskmodem_k4_m1_b025() {
        testbench_gmskmodem(4, 1, 0.25);
    }

    #[test]
    #[autotest_annotate(autotest_gmskmodem_k4_m2_b025)]
    fn test_gmskmodem_k4_m2_b025() {
        testbench_gmskmodem(4, 2, 0.25);
    }

    #[test]
    #[autotest_annotate(autotest_gmskmodem_k4_m8_b025)]
    fn test_gmskmodem_k4_m8_b025() {
        testbench_gmskmodem(4, 8, 0.25);
    }

    // test different filter bandwidth factors
    #[test]
    #[autotest_annotate(autotest_gmskmodem_k4_m3_b020)]
    fn test_gmskmodem_k4_m3_b020() {
        testbench_gmskmodem(4, 3, 0.20);
    }

    // note -- this test is named 033, but it tests 0.25
    // it's the same as in liquid-dsp
    #[test]
    #[autotest_annotate(autotest_gmskmodem_k4_m3_b033)]
    fn test_gmskmodem_k4_m3_b033() {
        testbench_gmskmodem(4, 3, 0.25);
    }

    // note -- this test is named 050, but it tests 0.25
    // it's the same as in liquid-dsp
    #[test]
    #[autotest_annotate(autotest_gmskmodem_k4_m3_b050)]
    fn test_gmskmodem_k4_m3_b050() {
        testbench_gmskmodem(4, 3, 0.25);
    }

    #[test]
    #[autotest_annotate(autotest_gmskdem_copy)]
    fn autotest_gmskdem_copy() {
        use crate::random::randnf;

        let k = 5;
        let m = 3;
        let bt = 0.2345;

        let mut dem_orig = GmskDem::new(k, m, bt).unwrap();

        let num_symbols = 16;
        let mut buf = vec![Complex32::new(0.0, 0.0); k];

        // run original object
        for _ in 0..num_symbols {
            for j in 0..k {
                buf[j] = Complex32::new(randnf(), randnf());
            }
            dem_orig.demodulate(&buf).unwrap();
        }

        // copy object
        let mut dem_copy = dem_orig.clone();

        // run through both objects and compare
        for _ in 0..num_symbols {
            for j in 0..k {
                buf[j] = Complex32::new(randnf(), randnf());
            }
            let sym_orig = dem_orig.demodulate(&buf).unwrap();
            let sym_copy = dem_copy.demodulate(&buf).unwrap();
            assert_eq!(sym_orig, sym_copy);
        }
    }
}
