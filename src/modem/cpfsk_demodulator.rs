// continuous phase frequency-shift keying demodulator

use super::cpfsk_modulator::CpfskFilterType;
use crate::error::{Error, Result};
use crate::filter::{FirFilter, FirFilterShape};
use num_complex::Complex32;
use std::f32::consts::PI;

/// Continuous-phase frequency-shift keying demodulator
#[derive(Clone, Debug)]
#[doc(alias = "Cpfskdem")]
pub struct CpfskDemodulator {
    bps: usize, // bits per symbol
    k: usize,   // samples per symbol
    beta: f32,  // filter bandwidth parameter
    h: f32,     // modulation index
    filter_type: CpfskFilterType,
    m_size: usize,       // constellation size (M = 2^bps)
    symbol_delay: usize, // receiver filter delay [symbols]

    // matched filter
    mf: FirFilter<Complex32, f32>,

    // state variables
    z_prime: Complex32, // previous filtered sample
}

impl CpfskDemodulator {
    /// Create CPFSK demodulator object (frequency demodulator)
    ///
    /// # Arguments
    ///
    /// * `bits_per_symbol` - bits per symbol; must be greater than zero
    /// * `modulation_index` - modulation index; must be greater than zero
    /// * `samples_per_symbol` - samples per symbol; must be greater than one and even
    /// * `filter_delay` - filter delay in symbols; must be greater than zero
    /// * `excess_bandwidth` - filter bandwidth parameter; must be in (0, 1]
    /// * `filter_type` - filter type (e.g. CpfskFilterType::Square)
    pub fn new(
        bits_per_symbol: usize,
        modulation_index: f32,
        samples_per_symbol: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
        filter_type: CpfskFilterType,
    ) -> Result<Self> {
        // validate input
        if bits_per_symbol == 0 {
            return Err(Error::Config("bits/symbol must be greater than 0".into()));
        }
        if modulation_index <= 0.0 {
            return Err(Error::Config("modulation index must be greater than 0".into()));
        }
        if samples_per_symbol < 2 || !samples_per_symbol.is_multiple_of(2) {
            return Err(Error::Config("samples/symbol must be greater than 2 and even".into()));
        }
        if filter_delay == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if excess_bandwidth <= 0.0 || excess_bandwidth > 1.0 {
            return Err(Error::Config("filter roll-off must be in (0,1]".into()));
        }

        let bps = bits_per_symbol;
        let h = modulation_index;
        let k = samples_per_symbol;
        let m = filter_delay;
        let beta = excess_bandwidth;

        let m_size = 1 << bps;

        // create matched filter based on filter type (non-coherent demodulator)
        let (mf, symbol_delay, scale) = Self::create_matched_filter(k, m, beta, filter_type, m_size)?;

        let mut q = Self { bps, k, beta, h, filter_type, m_size, symbol_delay, mf, z_prime: Complex32::new(0.0, 0.0) };

        q.mf.set_scale(scale);
        q.reset();
        Ok(q)
    }

    /// Create demodulator object for minimum-shift keying
    ///
    /// # Arguments
    ///
    /// * `samples_per_symbol` - Samples per symbol. Must be even and greater than one.
    pub fn new_msk(samples_per_symbol: usize) -> Result<Self> {
        Self::new(1, 0.5, samples_per_symbol, 1, 1.0, CpfskFilterType::Square)
    }

    /// Create demodulator object for Gaussian minimum-shift keying
    ///
    /// # Arguments
    ///
    /// * `samples_per_symbol` - Samples per symbol. Must be even and greater than one.
    /// * `filter_delay` - Filter delay in symbols. Must be greater than zero.
    /// * `bandwidth_time_product` - Bandwidth-time product in (0, 1].
    pub fn new_gmsk(samples_per_symbol: usize, filter_delay: usize, bandwidth_time_product: f32) -> Result<Self> {
        Self::new(1, 0.5, samples_per_symbol, filter_delay, bandwidth_time_product, CpfskFilterType::Gmsk)
    }

    /// Create matched filter for non-coherent demodulation
    fn create_matched_filter(
        k: usize,
        m: usize,
        beta: f32,
        filter_type: CpfskFilterType,
        m_size: usize,
    ) -> Result<(FirFilter<Complex32, f32>, usize, f32)> {
        let gmsk_bt = beta;

        match filter_type {
            CpfskFilterType::Square => {
                let bw = 0.4;
                let symbol_delay = m;
                let mf = FirFilter::new_kaiser(2 * k * m + 1, bw, 60.0, 0.0)?;
                let scale = 2.0 * bw;
                Ok((mf, symbol_delay, scale))
            }
            CpfskFilterType::RcosFull => {
                if m_size == 2 {
                    let mf = FirFilter::new_rnyquist(FirFilterShape::Gmskrx, k, m, 0.5, 0.0)?;
                    let scale = 1.33 / k as f32;
                    let symbol_delay = m;
                    Ok((mf, symbol_delay, scale))
                } else {
                    let mf = FirFilter::new_rnyquist(FirFilterShape::Gmskrx, k / 2, 2 * m, 0.9, 0.0)?;
                    let scale = 3.25 / k as f32;
                    let symbol_delay = 0; // TODO: fix this value
                    Ok((mf, symbol_delay, scale))
                }
            }
            CpfskFilterType::RcosPartial => {
                if m_size == 2 {
                    let mf = FirFilter::new_rnyquist(FirFilterShape::Gmskrx, k, m, 0.3, 0.0)?;
                    let scale = 1.10 / k as f32;
                    let symbol_delay = m;
                    Ok((mf, symbol_delay, scale))
                } else {
                    let mf = FirFilter::new_rnyquist(FirFilterShape::Gmskrx, k / 2, 2 * m, 0.27, 0.0)?;
                    let scale = 2.90 / k as f32;
                    let symbol_delay = 0; // TODO: fix this value
                    Ok((mf, symbol_delay, scale))
                }
            }
            CpfskFilterType::Gmsk => {
                let bw = 0.5 / k as f32;
                // TODO: figure out beta value here
                let filter_beta = if m_size == 2 { 0.8 * gmsk_bt } else { 1.0 * gmsk_bt };
                let mf = FirFilter::new_rnyquist(FirFilterShape::Gmskrx, k, m, filter_beta, 0.0)?;
                let scale = 2.0 * bw;
                let symbol_delay = m;
                Ok((mf, symbol_delay, scale))
            }
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.mf.reset();
        self.z_prime = Complex32::new(0.0, 0.0);
    }

    /// Get demodulator's number of bits per symbol
    pub fn bits_per_symbol(&self) -> usize {
        self.bps
    }

    /// Get demodulator's modulation index
    pub fn modulation_index(&self) -> f32 {
        self.h
    }

    /// Get demodulator's number of samples per symbol
    pub fn samples_per_symbol(&self) -> usize {
        self.k
    }

    /// Get demodulator's filter delay [symbols]
    pub fn delay(&self) -> usize {
        self.symbol_delay
    }

    /// Get demodulator's bandwidth parameter
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Get demodulator's filter type
    pub fn filter_type(&self) -> CpfskFilterType {
        self.filter_type
    }

    /// Demodulate array of samples (non-coherent)
    ///
    /// # Arguments
    ///
    /// * `input` - input sample array [size: k x 1]
    ///
    /// # Returns
    ///
    /// Demodulated symbol
    pub fn demodulate(&mut self, input: &[Complex32]) -> Result<usize> {
        if input.len() < self.k {
            return Err(Error::Range(format!(
                "input buffer length ({}) must be at least samples/symbol ({})",
                input.len(),
                self.k
            )));
        }

        let mut sym_out = 0;

        for (i, &yi) in input.iter().enumerate() {
            // push input sample through filter
            self.mf.push(yi);

            // decimate output - only compute at first sample of symbol
            if i == 0 {
                // compute output sample
                let z = self.mf.execute();

                // compute instantaneous frequency scaled by modulation index
                let phi_hat = (self.z_prime.conj() * z).arg() / (self.h * PI);

                // estimate transmitted symbol
                let v = (phi_hat + (self.m_size - 1) as f32) * 0.5;
                sym_out = (v.round() as usize) % self.m_size;

                // save current point
                self.z_prime = z;
            }
        }

        Ok(sym_out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::CpfskModulator;
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_config)]
    fn test_cpfskmodem_config() {
        // test creating invalid modulator objects
        assert!(CpfskModulator::new(0, 0.5, 4, 12, 0.25, CpfskFilterType::Square).is_err()); // bps is less than 1
        assert!(CpfskModulator::new(1, 0.0, 4, 12, 0.25, CpfskFilterType::Square).is_err()); // h (mod index) is out of range
        assert!(CpfskModulator::new(1, 0.5, 0, 12, 0.25, CpfskFilterType::Square).is_err()); // k is too small
        assert!(CpfskModulator::new(1, 0.5, 5, 12, 0.25, CpfskFilterType::Square).is_err()); // k is not even
        assert!(CpfskModulator::new(1, 0.5, 4, 0, 0.25, CpfskFilterType::Square).is_err()); // m is too small
        assert!(CpfskModulator::new(1, 0.5, 4, 12, 0.0, CpfskFilterType::Square).is_err()); // beta is too small
        assert!(CpfskModulator::new(1, 0.5, 4, 12, 7.22, CpfskFilterType::Square).is_err()); // beta is too large

        // test creating invalid demodulator objects
        assert!(CpfskDemodulator::new(0, 0.5, 4, 12, 0.25, CpfskFilterType::Square).is_err()); // bps is less than 1
        assert!(CpfskDemodulator::new(1, 0.0, 4, 12, 0.25, CpfskFilterType::Square).is_err()); // h (mod index) is out of range
        assert!(CpfskDemodulator::new(1, 0.5, 0, 12, 0.25, CpfskFilterType::Square).is_err()); // k is too small
        assert!(CpfskDemodulator::new(1, 0.5, 5, 12, 0.25, CpfskFilterType::Square).is_err()); // k is not even
        assert!(CpfskDemodulator::new(1, 0.5, 4, 0, 0.25, CpfskFilterType::Square).is_err()); // m is too small
        assert!(CpfskDemodulator::new(1, 0.5, 4, 12, 0.0, CpfskFilterType::Square).is_err()); // beta is too small
        assert!(CpfskDemodulator::new(1, 0.5, 4, 12, 7.22, CpfskFilterType::Square).is_err()); // beta is too large

        // create modulator object and check configuration
        let mod_ = CpfskModulator::new(1, 0.5, 4, 12, 0.5, CpfskFilterType::Square).unwrap();
        assert_eq!(mod_.bits_per_symbol(), 1);
        assert_abs_diff_eq!(mod_.modulation_index(), 0.5);
        assert_eq!(mod_.samples_per_symbol(), 4);
        assert_abs_diff_eq!(mod_.beta(), 0.5);
        assert_eq!(mod_.filter_type(), CpfskFilterType::Square);

        // create demodulator object and check configuration
        let dem = CpfskDemodulator::new(1, 0.5, 4, 12, 0.5, CpfskFilterType::Square).unwrap();
        assert_eq!(dem.bits_per_symbol(), 1);
        assert_abs_diff_eq!(dem.modulation_index(), 0.5);
        assert_eq!(dem.samples_per_symbol(), 4);
        assert_abs_diff_eq!(dem.beta(), 0.5);
        assert_eq!(dem.filter_type(), CpfskFilterType::Square);
    }

    #[test]
    fn test_cpfskdem_create() {
        // valid creation
        let result = CpfskDemodulator::new(2, 0.5, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_ok());

        // invalid bps
        let result = CpfskDemodulator::new(0, 0.5, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid modulation index
        let result = CpfskDemodulator::new(2, 0.0, 4, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid k (odd)
        let result = CpfskDemodulator::new(2, 0.5, 3, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid k (too small)
        let result = CpfskDemodulator::new(2, 0.5, 1, 3, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid m
        let result = CpfskDemodulator::new(2, 0.5, 4, 0, 0.35, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        // invalid beta
        let result = CpfskDemodulator::new(2, 0.5, 4, 3, 0.0, CpfskFilterType::Gmsk);
        assert!(result.is_err());

        let result = CpfskDemodulator::new(2, 0.5, 4, 3, 1.5, CpfskFilterType::Gmsk);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpfskdem_msk() {
        let result = CpfskDemodulator::new_msk(4);
        assert!(result.is_ok());
        let dem = result.unwrap();
        assert_eq!(dem.bits_per_symbol(), 1);
        assert_abs_diff_eq!(dem.modulation_index(), 0.5);
        assert_eq!(dem.filter_type(), CpfskFilterType::Square);
    }

    #[test]
    fn test_cpfskdem_gmsk() {
        let result = CpfskDemodulator::new_gmsk(4, 3, 0.35);
        assert!(result.is_ok());
        let dem = result.unwrap();
        assert_eq!(dem.bits_per_symbol(), 1);
        assert_abs_diff_eq!(dem.modulation_index(), 0.5);
        assert_eq!(dem.filter_type(), CpfskFilterType::Gmsk);
    }

    /// Helper function for mod/demod testing
    fn cpfskmodem_test_mod_demod(mut mod_: CpfskModulator, mut dem: CpfskDemodulator) {
        let delay = mod_.delay() + dem.delay();
        let k = mod_.samples_per_symbol();
        let bps = mod_.bits_per_symbol();

        let num_symbols = 180 + delay;
        let mut buf = vec![Complex32::new(0.0, 0.0); k];
        let mut sym_in = vec![0usize; num_symbols];
        let mut sym_out = vec![0usize; num_symbols];

        // modulate, demodulate
        let mut ms = crate::sequence::MaximalLengthSequence::from_degree(7).unwrap();
        for i in 0..num_symbols {
            // generate random symbol
            sym_in[i] = ms.generate_symbol(bps as u32) as usize;

            // modulate
            mod_.modulate(sym_in[i], &mut buf).unwrap();

            // demodulate
            sym_out[i] = dem.demodulate(&buf).unwrap();
        }

        // count errors
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

    /// Helper function to create mod/dem pair and test
    fn cpfskmodem_test_harness(bps: usize, h: f32, k: usize, m: usize, beta: f32, filter_type: CpfskFilterType) {
        let mod_ = CpfskModulator::new(bps, h, k, m, beta, filter_type).unwrap();
        let dem = CpfskDemodulator::new(bps, h, k, m, beta, filter_type).unwrap();

        // ensure values match
        assert_eq!(mod_.samples_per_symbol(), k);
        assert_eq!(dem.samples_per_symbol(), k);

        // run modulation/demodulation tests
        cpfskmodem_test_mod_demod(mod_, dem);
    }

    // Square pulse shape tests
    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p5000_k4_m3_square)]
    fn test_cpfskmodem_bps1_h0p5000_k4_m3_square() {
        cpfskmodem_test_harness(1, 0.5000, 4, 3, 0.25, CpfskFilterType::Square);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p0250_k4_m3_square)]
    fn test_cpfskmodem_bps1_h0p0250_k4_m3_square() {
        cpfskmodem_test_harness(1, 0.2500, 4, 3, 0.25, CpfskFilterType::Square);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p1250_k4_m3_square)]
    fn test_cpfskmodem_bps1_h0p1250_k4_m3_square() {
        cpfskmodem_test_harness(1, 0.1250, 4, 3, 0.25, CpfskFilterType::Square);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p0625_k4_m3_square)]
    fn test_cpfskmodem_bps1_h0p0625_k4_m3_square() {
        cpfskmodem_test_harness(1, 0.0625, 4, 3, 0.25, CpfskFilterType::Square);
    }

    // Raised-cosine full tests
    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p5000_k4_m3_rcosfull)]
    fn test_cpfskmodem_bps1_h0p5000_k4_m3_rcosfull() {
        cpfskmodem_test_harness(1, 0.5000, 4, 3, 0.25, CpfskFilterType::RcosFull);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p0250_k4_m3_rcosfull)]
    fn test_cpfskmodem_bps1_h0p0250_k4_m3_rcosfull() {
        cpfskmodem_test_harness(1, 0.2500, 4, 3, 0.25, CpfskFilterType::RcosFull);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p1250_k4_m3_rcosfull)]
    fn test_cpfskmodem_bps1_h0p1250_k4_m3_rcosfull() {
        cpfskmodem_test_harness(1, 0.1250, 4, 3, 0.25, CpfskFilterType::RcosFull);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p0625_k4_m3_rcosfull)]
    fn test_cpfskmodem_bps1_h0p0625_k4_m3_rcosfull() {
        cpfskmodem_test_harness(1, 0.0625, 4, 3, 0.25, CpfskFilterType::RcosFull);
    }

    // Raised-cosine partial tests
    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p5000_k4_m3_rcospart)]
    fn test_cpfskmodem_bps1_h0p5000_k4_m3_rcospart() {
        cpfskmodem_test_harness(1, 0.5000, 4, 3, 0.25, CpfskFilterType::RcosPartial);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p0250_k4_m3_rcospart)]
    fn test_cpfskmodem_bps1_h0p0250_k4_m3_rcospart() {
        cpfskmodem_test_harness(1, 0.2500, 4, 3, 0.25, CpfskFilterType::RcosPartial);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p1250_k4_m3_rcospart)]
    fn test_cpfskmodem_bps1_h0p1250_k4_m3_rcospart() {
        cpfskmodem_test_harness(1, 0.1250, 4, 3, 0.25, CpfskFilterType::RcosPartial);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p0625_k4_m3_rcospart)]
    fn test_cpfskmodem_bps1_h0p0625_k4_m3_rcospart() {
        cpfskmodem_test_harness(1, 0.0625, 4, 3, 0.25, CpfskFilterType::RcosPartial);
    }

    // GMSK tests
    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p5000_k4_m3_gmsk)]
    fn test_cpfskmodem_bps1_h0p5000_k4_m3_gmsk() {
        cpfskmodem_test_harness(1, 0.5000, 4, 3, 0.25, CpfskFilterType::Gmsk);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p0250_k4_m3_gmsk)]
    fn test_cpfskmodem_bps1_h0p0250_k4_m3_gmsk() {
        cpfskmodem_test_harness(1, 0.2500, 4, 3, 0.25, CpfskFilterType::Gmsk);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p1250_k4_m3_gmsk)]
    fn test_cpfskmodem_bps1_h0p1250_k4_m3_gmsk() {
        cpfskmodem_test_harness(1, 0.1250, 4, 3, 0.25, CpfskFilterType::Gmsk);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p0625_k4_m3_gmsk)]
    fn test_cpfskmodem_bps1_h0p0625_k4_m3_gmsk() {
        cpfskmodem_test_harness(1, 0.0625, 4, 3, 0.25, CpfskFilterType::Gmsk);
    }

    // Different bits per symbol tests
    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps2_h0p0250_k4_m3_square)]
    fn test_cpfskmodem_bps2_h0p0250_k4_m3_square() {
        cpfskmodem_test_harness(2, 0.2500, 4, 3, 0.25, CpfskFilterType::Square);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps3_h0p1250_k4_m3_square)]
    fn test_cpfskmodem_bps3_h0p1250_k4_m3_square() {
        cpfskmodem_test_harness(3, 0.1250, 4, 3, 0.25, CpfskFilterType::Square);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps4_h0p0625_k4_m3_square)]
    fn test_cpfskmodem_bps4_h0p0625_k4_m3_square() {
        cpfskmodem_test_harness(4, 0.0625, 4, 3, 0.25, CpfskFilterType::Square);
    }

    // Different samples per symbol tests (GMSK)
    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p5_k2_m7_gmsk)]
    fn test_cpfskmodem_bps1_h0p5_k2_m7_gmsk() {
        cpfskmodem_test_harness(1, 0.5, 2, 7, 0.30, CpfskFilterType::Gmsk);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p5_k4_m7_gmsk)]
    fn test_cpfskmodem_bps1_h0p5_k4_m7_gmsk() {
        cpfskmodem_test_harness(1, 0.5, 4, 7, 0.30, CpfskFilterType::Gmsk);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p5_k6_m7_gmsk)]
    fn test_cpfskmodem_bps1_h0p5_k6_m7_gmsk() {
        cpfskmodem_test_harness(1, 0.5, 6, 7, 0.30, CpfskFilterType::Gmsk);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_bps1_h0p5_k8_m7_gmsk)]
    fn test_cpfskmodem_bps1_h0p5_k8_m7_gmsk() {
        cpfskmodem_test_harness(1, 0.5, 8, 7, 0.30, CpfskFilterType::Gmsk);
    }

    #[test]
    #[autotest_annotate(autotest_cpfskmodem_spectrum)]
    fn test_cpfskmodem_spectrum() {
        use crate::fft::SpectralPeriodogram;
        use crate::utility::test_helpers::{validate_psd_spgramcf, PsdRegion};
        use rand::Rng;

        // create modulator
        let bps = 1;
        let h = 0.5;
        let k = 4;
        let m = 3;
        let beta = 0.35;
        let filter_type = CpfskFilterType::RcosPartial;
        let mut mod_ = CpfskModulator::new(bps, h, k, m, beta, filter_type).unwrap();

        // spectral periodogram options
        let nfft = 2400;
        let num_symbols = 192000;
        let mut buf = vec![Complex32::new(0.0, 0.0); k];

        // modulate many, many symbols to warm up state
        for _ in 0..(1 << 24) {
            mod_.modulate(0, &mut buf).unwrap();
        }

        // modulate several symbols and run result through spectral estimate
        let mut periodogram = SpectralPeriodogram::<Complex32>::from_nfft(nfft).unwrap();
        let mut rng = rand::thread_rng();
        for _ in 0..num_symbols {
            let s = (rng.gen::<u32>() & ((1 << bps) - 1)) as usize;
            mod_.modulate(s, &mut buf).unwrap();
            periodogram.write(&buf);
        }

        // verify spectrum
        #[rustfmt::skip]
        let regions = [
            PsdRegion { fmin: -0.50, fmax: -0.35, pmin: 0.0, pmax: -40.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.35, fmax: -0.20, pmin: 0.0, pmax: -20.0, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.10, fmax:  0.10, pmin: 0.0, pmax:  10.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.20, fmax:  0.35, pmin: 0.0, pmax: -20.0, test_lo: false, test_hi: true },
            PsdRegion { fmin:  0.35, fmax:  0.50, pmin: 0.0, pmax: -40.0, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_spgramcf(&periodogram, &regions).unwrap());
    }
}
