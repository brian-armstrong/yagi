//! Symbol tracker/synchronizer.

use crate::agc::Agc;
use crate::equalization::LeastMeanSquaresEqualizer;
use crate::error::{Error, Result};
use crate::filter::{FirFilterShape, SymbolSynchronizer};
use crate::modem::{Modem, ModulationScheme};
use crate::nco::{Nco, NcoBackend};
use num_complex::{Complex32, ComplexFloat};

// number of symbols to receive before enabling the equalizer update
const SYMTRACK_EQ_WARMUP: usize = 200;

/// Equalizer strategy used by [`SymbolTracker`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum EqStrategy {
    /// constant modulus
    ConstantModulus,
    /// decision directed
    DecisionDirected,
    /// disabled
    Off,
}

/// Recovers symbols from a continuous stream of linearly modulated samples.
///
/// The tracker needs no preamble. Each input sample passes through automatic
/// gain control and symbol timing recovery, which outputs two samples per
/// symbol. A phase-locked oscillator removes the carrier offset, and an
/// adaptive equalizer produces one output per symbol. A demodulator drives the
/// carrier loop and, in decision-directed mode, the equalizer. The output is
/// the equalized complex symbol, not a hard decision.
///
/// All loops start from scratch and take time to converge, so the first
/// outputs are unreliable. Equalizer adaptation begins after the first 200
/// symbols. The default adaptation is constant-modulus. [`Self::set_bandwidth`]
/// sets how fast the loops respond.
#[doc(alias = "symtrack")]
pub struct SymbolTracker {
    // parameters
    filter_type: Option<FirFilterShape>, // filter type (None => kaiser fallback)
    k: usize,                            // samples/symbol
    m: usize,                            // filter semi-length
    beta: f32,                           // filter excess bandwidth
    mod_scheme: ModulationScheme,        // demodulator scheme
    bw: f32,                             // tracking bandwidth

    // automatic gain control
    agc: Agc<Complex32>,

    // symbol timing recovery
    symsync: SymbolSynchronizer<Complex32>,
    symsync_index: usize, // symsync output sample index

    // equalizer/decimator
    eq: LeastMeanSquaresEqualizer<Complex32>,
    eq_strategy: EqStrategy,

    // nco/phase-locked loop (carrier recovery)
    nco: Nco,

    // demodulator
    demod: Modem,

    // state and counters
    num_syms_rx: usize, // number of symbols recovered
}

impl SymbolTracker {
    /// Creates a tracker for `samples_per_symbol` input samples per symbol.
    ///
    /// `filter_shape` is the transmitter's pulse shape, which sets the matched
    /// filter. `None` selects a fixed Kaiser design that ignores
    /// `excess_bandwidth`, though it is still range-checked. `filter_delay` is
    /// the filter semi-length in symbols, and `excess_bandwidth` is the rolloff
    /// factor. The loop bandwidth starts at 0.9.
    ///
    /// # Errors
    ///
    /// Returns an error when `samples_per_symbol` is less than two,
    /// `filter_delay` is zero, `excess_bandwidth` is outside `(0, 1]`,
    /// `modulation_scheme` has no built-in constellation, or the selected
    /// shape rejects the configuration.
    pub fn new(
        filter_shape: Option<FirFilterShape>,
        samples_per_symbol: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
        modulation_scheme: ModulationScheme,
    ) -> Result<Self> {
        // validate input
        if samples_per_symbol < 2 {
            return Err(Error::Config("filter samples/symbol must be at least 2".into()));
        }
        if filter_delay == 0 {
            return Err(Error::Config("filter delay must be greater than zero".into()));
        }
        if excess_bandwidth <= 0.0 || excess_bandwidth > 1.0 {
            return Err(Error::Config("filter excess bandwidth must be in (0,1]".into()));
        }
        if modulation_scheme == ModulationScheme::Unknown {
            return Err(Error::Config("invalid modulation scheme".into()));
        }

        // create automatic gain control
        let agc = Agc::new();

        // create symbol synchronizer (output rate: 2 samples per symbol)
        let mut symsync = match filter_shape {
            None => SymbolSynchronizer::new_kaiser(samples_per_symbol, filter_delay, 16)?,
            Some(shape) => {
                SymbolSynchronizer::new_rnyquist(shape, samples_per_symbol, filter_delay, excess_bandwidth, 16)?
            }
        };
        symsync.set_output_rate(2)?;

        // create equalizer as default low-pass filter with integer symbol delay
        // (2 samples/symbol)
        let eq_len = 2 * 4 + 1;
        let eq = LeastMeanSquaresEqualizer::new_lowpass(eq_len, 0.45)?;

        // nco and phase-locked loop
        let nco = Nco::new(NcoBackend::InterpolatedLookupTable);

        // demodulator
        let demod = Modem::new(modulation_scheme)?;

        let mut q = Self {
            filter_type: filter_shape,
            k: samples_per_symbol,
            m: filter_delay,
            beta: excess_bandwidth,
            mod_scheme: modulation_scheme,
            bw: 0.0,
            agc,
            symsync,
            symsync_index: 0,
            eq,
            eq_strategy: EqStrategy::ConstantModulus,
            nco,
            demod,
            num_syms_rx: 0,
        };

        // set default bandwidth
        q.set_bandwidth(0.9)?;

        // reset and return main object
        q.reset();
        Ok(q)
    }

    /// Creates a QPSK tracker at two samples per symbol.
    ///
    /// The matched filter is [`FirFilterShape::Arkaiser`] with a semi-length
    /// of 7 symbols and a rolloff of 0.3. This never returns an error.
    pub fn new_default() -> Result<Self> {
        Self::new(
            Some(FirFilterShape::Arkaiser),
            2,   // samples/symbol
            7,   // filter delay
            0.3, // filter excess bandwidth
            ModulationScheme::Qpsk,
        )
    }

    /// Returns every loop to its initial state.
    ///
    /// The carrier frequency returns to zero, which discards any
    /// [`Self::adjust_frequency`] seed. The loop bandwidth, equalizer mode,
    /// and modulation scheme are kept.
    pub fn reset(&mut self) {
        // reset objects
        self.agc.reset();
        self.symsync.reset();
        self.eq.reset();
        self.nco.reset();
        self.demod.reset();

        // reset internal counters
        self.symsync_index = 0;
        self.num_syms_rx = 0;
    }

    /// Returns the matched-filter shape, or `None` for the fixed Kaiser
    /// design.
    pub fn filter_shape(&self) -> Option<FirFilterShape> {
        self.filter_type
    }

    /// Returns the number of input samples per symbol.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the matched-filter semi-length in symbols.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Returns the rolloff factor given at construction.
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Returns the demodulation scheme.
    pub fn modulation_scheme(&self) -> ModulationScheme {
        self.mod_scheme
    }

    /// Changes the demodulation scheme.
    ///
    /// [`ModulationScheme::Unknown`] selects BPSK, although [`Self::new`]
    /// rejects it. The loops keep their state.
    ///
    /// # Errors
    ///
    /// Returns an error when `modulation_scheme` has no built-in constellation.
    /// The scheme is unchanged on error.
    pub fn set_modulation_scheme(&mut self, modulation_scheme: ModulationScheme) -> Result<()> {
        // an unknown scheme falls back to BPSK, matching liquid
        let modulation_scheme = if modulation_scheme == ModulationScheme::Unknown {
            ModulationScheme::Bpsk
        } else {
            modulation_scheme
        };

        // re-create modem (yagi has no in-place recreate)
        self.demod = Modem::new(modulation_scheme)?;
        self.mod_scheme = modulation_scheme;
        Ok(())
    }

    /// Returns the loop bandwidth set by [`Self::set_bandwidth`].
    pub fn bandwidth(&self) -> f32 {
        self.bw
    }

    /// Sets how fast all the loops respond.
    ///
    /// `bandwidth` scales every loop together. The AGC and equalizer receive
    /// `0.02 * bandwidth`, and the timing and carrier loops receive
    /// `0.001 * bandwidth`.
    /// Larger values converge faster and let more noise through. The
    /// nominal range is `[0, 1]`, and the default is 0.9. Values up to 50 are
    /// accepted.
    ///
    /// # Errors
    ///
    /// Returns an error when `bandwidth` is negative, above 50, or NaN. The
    /// loops are unchanged on error, but [`Self::bandwidth`] reports the
    /// rejected value unless `bandwidth` was negative.
    pub fn set_bandwidth(&mut self, bandwidth: f32) -> Result<()> {
        // validate input
        if bandwidth < 0.0 {
            return Err(Error::Config("bandwidth must be in [0,1]".into()));
        }

        self.bw = bandwidth;

        // set bandwidths accordingly
        let agc_bandwidth = 0.02 * bandwidth;
        let symsync_bandwidth = 0.001 * bandwidth;
        let eq_bandwidth = 0.02 * bandwidth;
        let pll_bandwidth = 0.001 * bandwidth;

        // automatic gain control
        self.agc.set_bandwidth(agc_bandwidth)?;

        // symbol timing recovery
        self.symsync.set_lf_bw(symsync_bandwidth)?;

        // equalizer
        self.eq.set_bw(eq_bandwidth)?;

        // phase-locked loop
        self.nco.pll_set_bandwidth(pll_bandwidth);
        Ok(())
    }

    /// Adds `frequency_offset` to the carrier frequency estimate.
    ///
    /// The carrier loop runs after timing recovery, at two samples per symbol,
    /// so `frequency_offset` is in radians per half symbol. This matches
    /// radians per input sample only when `k` is two. To seed the loop from an
    /// offset in radians per input sample, such as [`FrameDetector::dphi`],
    /// multiply it by `k / 2` first.
    ///
    /// [`FrameDetector::dphi`]: crate::framing::FrameDetector::dphi
    pub fn adjust_frequency(&mut self, frequency_offset: f32) {
        self.nco.adjust_frequency(frequency_offset);
    }

    /// Adds `phase_offset` radians to the carrier phase estimate.
    pub fn adjust_phase(&mut self, phase_offset: f32) {
        self.nco.adjust_phase(phase_offset);
    }

    /// Adapts the equalizer toward a constant output magnitude.
    ///
    /// This mode needs no symbol decisions, so it works before the carrier
    /// loop has locked. It is the default.
    pub fn set_equalizer_constant_modulus(&mut self) {
        self.eq_strategy = EqStrategy::ConstantModulus;
    }

    /// Adapts the equalizer toward the nearest constellation point.
    ///
    /// This mode relies on correct symbol decisions, so use it once the loops
    /// have locked.
    pub fn set_equalizer_decision_directed(&mut self) {
        self.eq_strategy = EqStrategy::DecisionDirected;
    }

    /// Stops equalizer adaptation and keeps the current coefficients.
    pub fn disable_equalizer_adaptation(&mut self) {
        self.eq_strategy = EqStrategy::Off;
    }

    /// Processes one input sample and appends any recovered symbols to
    /// `output`.
    ///
    /// Returns the number of symbols appended. This is usually zero or one,
    /// averaging one per `k` input samples. This never returns an error.
    ///
    /// # Warning
    ///
    /// Once equalizer adaptation has begun, a run of zero-valued input that
    /// fills the equalizer history can make the equalizer coefficients NaN.
    /// Constant-modulus mode always fails this way, and decision-directed mode
    /// can. Every later output is then NaN until [`Self::reset`]. Disable
    /// adaptation with [`Self::disable_equalizer_adaptation`] across gaps in
    /// the signal.
    pub fn execute(&mut self, input: Complex32, output: &mut Vec<Complex32>) -> Result<usize> {
        let mut num_outputs = 0;

        // run sample through automatic gain control
        let v = self.agc.execute(input)?;

        // symbol synchronizer
        let symsync_output = self.symsync.execute(&[v])?;

        // process each output sample
        for &sample in symsync_output {
            // update phase-locked loop
            self.nco.step();
            let v = self.nco.mix_down(sample);

            // equalizer/decimator
            self.eq.push(v);

            // decimate result, noting that symsync outputs at exactly 2 samples/symbol
            self.symsync_index += 1;
            if self.symsync_index.is_multiple_of(2) {
                continue;
            }

            // increment number of symbols received
            self.num_syms_rx += 1;

            // compute equalizer filter output; updating coefficients is dependent
            // upon strategy
            let d_hat = self.eq.execute()?;

            // demodulate result, apply phase correction
            let _sym_out = self.demod.demodulate(d_hat)?;
            let phase_error = self.demod.demodulator_phase_error();

            // update pll
            self.nco.pll_step(phase_error);

            // update equalizer independent of the signal: estimate error assuming
            // constant modulus signal
            // TODO: check lock conditions of previous object to determine when to
            // run equalizer
            if self.num_syms_rx > SYMTRACK_EQ_WARMUP && self.eq_strategy != EqStrategy::Off {
                let d_prime = match self.eq_strategy {
                    EqStrategy::ConstantModulus => d_hat / d_hat.abs(),
                    EqStrategy::DecisionDirected => self.demod.demodulator_sample(),
                    EqStrategy::Off => unreachable!(),
                };
                self.eq.step(d_prime, d_hat);
            }

            // save result to output
            output.push(d_hat);
            num_outputs += 1;
        }

        Ok(num_outputs)
    }

    /// Processes a block of input samples and appends the recovered symbols to
    /// `output`.
    ///
    /// Returns the number of symbols appended. The result matches calling
    /// [`Self::execute`] for each sample. This never returns an error.
    pub fn execute_block(&mut self, input: &[Complex32], output: &mut Vec<Complex32>) -> Result<usize> {
        let mut num_written = 0;
        for &xi in input {
            num_written += self.execute(xi, output)?;
        }
        Ok(num_written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::SymbolStream;
    use test_macro::autotest_annotate;

    fn testbench_symtrack_cccf(k: usize, m: usize, beta: f32, ms: ModulationScheme) {
        let ftype = FirFilterShape::Arkaiser;
        let num_symbols = 6000; // number of data symbols
        let noise_floor = -30.0f32; // noise floor [dB]
        let snr_db = 30.0f32; // signal-to-noise ratio [dB]
        let bandwidth = 0.15f32; // loop filter bandwidth
        let dphi = 0.02f32; // carrier frequency offset [radians/sample]
        let phi = 2.1f32; // carrier phase offset [radians]

        let buf_len = 800; // buffer size

        // create stream generator (symstream delay is 2*m)
        let mut gen = SymbolStream::new_linear(ftype, k, 2 * m, beta, ms).unwrap();

        // create channel emulator and add impairments
        let mut chan = crate::channel::Channel::new();
        chan.add_awgn(noise_floor, snr_db);
        chan.add_carrier_offset(dphi, phi);

        // create symbol tracking synchronizer
        let mut symtrack = SymbolTracker::new(Some(ftype), k, m, beta, ms).unwrap();
        symtrack.set_bandwidth(bandwidth).unwrap();

        let mut total_samples = 0;
        let mut num_symbols_evm = 0;
        let mut evm = 0.0f32;
        let mut demod = Modem::new(ms).unwrap(); // for checking output EVM

        let mut buf_0 = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_1: Vec<Complex32> = Vec::new();
        while total_samples < num_symbols * k {
            // write samples to buffer
            gen.write_samples(&mut buf_0).unwrap();

            // apply channel (in place)
            let input = buf_0.clone();
            chan.execute_block(&input, &mut buf_0).unwrap();

            // run resulting stream through synchronizer
            buf_1.clear();
            let num_symbols_sync = symtrack.execute_block(&buf_0, &mut buf_1).unwrap();
            total_samples += buf_len;

            // check output EVM over the last 20% of the run
            if total_samples as f32 > 0.80 * (num_symbols * k) as f32 {
                for &sym in buf_1.iter().take(num_symbols_sync) {
                    demod.demodulate(sym).unwrap();
                    let err = demod.demodulator_evm();
                    evm += err * err;
                }
                num_symbols_evm += num_symbols_sync;
            }
        }

        // verify output constellation EVM is reasonably low
        let evm = 10.0 * (evm / num_symbols_evm as f32).log10();
        assert!(evm < -15.0, "EVM {} not below -15 dB", evm);
    }

    #[test]
    #[autotest_annotate(autotest_symtrack_cccf_bpsk)]
    fn symtrack_cccf_bpsk() {
        testbench_symtrack_cccf(2, 12, 0.25, ModulationScheme::Bpsk);
    }

    #[test]
    #[autotest_annotate(autotest_symtrack_cccf_qpsk)]
    fn symtrack_cccf_qpsk() {
        testbench_symtrack_cccf(2, 12, 0.25, ModulationScheme::Qpsk);
    }

    #[test]
    #[autotest_annotate(autotest_symtrack_cccf_config_invalid)]
    fn symtrack_cccf_config_invalid() {
        // invalid create configurations
        assert!(SymbolTracker::new(Some(FirFilterShape::Rrcos), 1, 12, 0.25, ModulationScheme::Qpsk).is_err());
        assert!(SymbolTracker::new(Some(FirFilterShape::Rrcos), 2, 0, 0.25, ModulationScheme::Qpsk).is_err());
        assert!(SymbolTracker::new(Some(FirFilterShape::Rrcos), 2, 12, 2.00, ModulationScheme::Qpsk).is_err());
        assert!(SymbolTracker::new(Some(FirFilterShape::Rrcos), 2, 12, 0.25, ModulationScheme::Unknown).is_err());

        // create proper object but test invalid internal configurations
        let mut q = SymbolTracker::new_default().unwrap();
        assert!(q.set_bandwidth(-1.0).is_err());
    }

    #[test]
    #[autotest_annotate(autotest_symtrack_cccf_config_valid)]
    fn symtrack_cccf_config_valid() {
        // create proper object and test configuration methods
        let mut q = SymbolTracker::new(Some(FirFilterShape::Arkaiser), 4, 12, 0.25, ModulationScheme::Qam64).unwrap();

        // test valid configurations
        q.adjust_phase(0.1);
        q.set_equalizer_constant_modulus();
        q.set_equalizer_decision_directed();
        q.disable_equalizer_adaptation();

        // test access methods
        assert_eq!(q.k(), 4);
        assert_eq!(q.m(), 12);
        assert_eq!(q.beta(), 0.25);
        assert_eq!(q.filter_shape(), Some(FirFilterShape::Arkaiser));
        assert_eq!(q.modulation_scheme(), ModulationScheme::Qam64);

        // test setting bandwidth
        q.set_bandwidth(0.1).unwrap();
        assert_eq!(q.bandwidth(), 0.1);

        // test setting modulation scheme
        q.set_modulation_scheme(ModulationScheme::Apsk16).unwrap();
        assert_eq!(q.modulation_scheme(), ModulationScheme::Apsk16);
    }
}
