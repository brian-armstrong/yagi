//! Frame detector and synchronizer.
//!
//! Uses a correlation method to detect a synchronization pattern, estimate
//! carrier frequency and phase offsets as well as timing phase, then correct for
//! these impairments in a simple interface suitable for custom frame recovery.

use crate::error::{Error, Result};
use crate::filter::{FirFilterShape, FirPolyphaseFilter};
use crate::framing::FrameDetector;
use crate::nco::{Nco, NcoBackend};
use num_complex::Complex32;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Detect, // detect frame
    Sync,   // apply carrier offset correction and matched filter
}

/// Result of [`QuadratureSynchronizer::execute`].
///
/// `symbols` is either empty or exactly one block of
/// [`QuadratureSynchronizer::block_len`] recovered symbols. It borrows the
/// synchronizer's storage. Continue with `&input[consumed..]`.
#[derive(Debug)]
#[doc(alias = "qdsync_callback")]
pub struct QuadratureSynchronizerOutput<'a> {
    /// Number of input samples consumed.
    pub consumed: usize,
    /// One block of recovered symbols, or empty if no block was ready.
    pub symbols: &'a [Complex32],
}

/// Result of [`QuadratureSynchronizer::execute_symbols_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadratureSynchronizerBlockOutput {
    /// Number of input samples consumed. Continue with `&input[consumed..]`.
    pub consumed: usize,
    /// Number of recovered symbols copied, a multiple of the block length.
    pub symbols_written: usize,
}

/// Detects a preamble and recovers the symbols of the frame that follows.
///
/// A [`FrameDetector`] searches the input for the preamble. On detection, the
/// synchronizer sets a carrier mixer, a gain, and a matched-filter timing
/// phase from the detector's estimates. It then filters and decimates the
/// input to one symbol per `samples_per_symbol` samples. The corrections are
/// fixed at detection. There is no tracking afterward, so a residual carrier
/// offset rotates later symbols and a timing drift accumulates.
///
/// Symbols are delivered in blocks of [`Self::block_len`], 64 by default. The
/// first `sequence.len()` symbols are the preamble given to [`Self::new_linear`],
/// followed by the payload. The synchronizer does not know the frame length
/// and keeps producing symbols after the frame ends. Call [`Self::reset`]
/// after the frame to search for the next preamble.
#[doc(alias = "qdsync")]
#[derive(Clone)]
pub struct QuadratureSynchronizer {
    k: usize, // samples per symbol
    m: usize, // filter semi-length

    detector: FrameDetector, // detector

    // status variables
    state: State,
    symbol_counter: usize, // total number of symbols received including preamble

    mixer: Nco, // coarse carrier frequency recovery

    // timing recovery objects, states
    mf: FirPolyphaseFilter<Complex32>, // matched filter/decimator
    npfb: usize,                       // number of filters in the bank
    mf_counter: i32,                   // matched filter output timer
    pfb_index: usize,                  // filterbank index

    // symbol buffer
    buf_out_len: usize,           // symbols delivered per block
    pending: VecDeque<Complex32>, // recovered symbols awaiting delivery
    block: Vec<Complex32>,        // scratch: one block handed back by execute()
}

impl QuadratureSynchronizer {
    /// Creates a synchronizer for a linearly modulated preamble.
    ///
    /// # Arguments
    ///
    /// - `sequence`: preamble symbols in transmission order.
    /// - `filter_shape`: the transmitter's pulse shape. The receiver uses the
    ///   matching matched filter.
    /// - `samples_per_symbol`: samples per symbol, in `2..=80`.
    /// - `filter_delay`: filter semi-length in symbols, in `1..=100`.
    /// - `excess_bandwidth`: rolloff factor, in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error when `sequence` is empty, a parameter is outside its
    /// range, or the selected shape rejects the configuration.
    pub fn new_linear(
        sequence: &[Complex32],
        filter_shape: FirFilterShape,
        samples_per_symbol: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
    ) -> Result<Self> {
        // validate input
        if sequence.is_empty() {
            return Err(Error::Config("sequence length cannot be zero".into()));
        }

        // create detector
        let detector =
            FrameDetector::new_linear(sequence, filter_shape, samples_per_symbol, filter_delay, excess_bandwidth)?;

        // create down-converter for carrier phase tracking
        let mixer = Nco::new(NcoBackend::LookupTable);

        // create symbol timing recovery filterbank
        let npfb = 256; // number of filters in the bank
        let mf =
            FirPolyphaseFilter::new_rnyquist(filter_shape, npfb, samples_per_symbol, filter_delay, excess_bandwidth)?;

        // block size for delivering recovered symbols (user can re-size later)
        let buf_out_len = 64;

        let mut q = Self {
            k: samples_per_symbol,
            m: filter_delay,
            detector,
            state: State::Detect,
            symbol_counter: 0,
            mixer,
            mf,
            npfb,
            mf_counter: 0,
            pfb_index: 0,
            buf_out_len,
            pending: VecDeque::new(),
            block: Vec::with_capacity(buf_out_len),
        };

        q.reset();
        Ok(q)
    }

    /// Returns to searching for a preamble.
    ///
    /// Buffered input, filter history, and undelivered symbols are discarded.
    /// The threshold, search range, and block length are kept. The estimate
    /// getters keep reporting the last detection until the next one.
    pub fn reset(&mut self) {
        self.detector.reset();
        self.state = State::Detect;
        self.symbol_counter = 0;
        self.pending.clear();
        self.mf.reset();
    }

    /// Returns whether a preamble has been detected since the last reset.
    pub fn is_detected(&self) -> bool {
        self.state == State::Sync
    }

    /// Returns the number of symbols in each delivered block.
    pub fn block_len(&self) -> usize {
        self.buf_out_len
    }

    /// Returns the detection threshold for the normalized correlation peak.
    pub fn threshold(&self) -> f32 {
        self.detector.threshold()
    }

    /// Sets the detection threshold for the normalized correlation peak.
    ///
    /// See [`FrameDetector::set_threshold`]. The default is 0.5.
    ///
    /// # Errors
    ///
    /// Returns an error when `threshold` is outside `(0, 2]`. The threshold is
    /// unchanged on error.
    pub fn set_threshold(&mut self, threshold: f32) -> Result<()> {
        self.detector.set_threshold(threshold)
    }

    /// Returns the carrier offset search range in radians per sample.
    pub fn range(&self) -> f32 {
        self.detector.range()
    }

    /// Sets the carrier offset search range in radians per sample.
    ///
    /// See [`FrameDetector::set_range`]. The default is 0.3.
    ///
    /// # Errors
    ///
    /// Returns an error when `carrier_offset_range` is outside `[0, 0.5]`.
    /// The range is unchanged on error.
    pub fn set_range(&mut self, carrier_offset_range: f32) -> Result<()> {
        self.detector.set_range(carrier_offset_range)
    }

    /// Sets the number of symbols in each delivered block.
    ///
    /// Undelivered symbols are kept and delivered in blocks of the new size.
    ///
    /// # Errors
    ///
    /// Returns an error when `block_len` is zero. The block length is unchanged
    /// on error.
    pub fn set_block_len(&mut self, block_len: usize) -> Result<()> {
        if block_len == 0 {
            return Err(Error::Config("block length must be greater than 0".into()));
        }

        self.buf_out_len = block_len;
        Ok(())
    }

    /// Returns the normalized correlation peak of the last detection.
    pub fn rxy(&self) -> f32 {
        self.detector.rxy()
    }

    /// Returns the fractional timing offset of the last detection, in samples.
    pub fn tau(&self) -> f32 {
        self.detector.tau()
    }

    /// Returns the channel gain of the last detection, as a linear amplitude
    /// factor.
    pub fn gamma(&self) -> f32 {
        self.detector.gamma()
    }

    /// Returns the carrier frequency offset of the last detection, in radians
    /// per sample.
    pub fn dphi(&self) -> f32 {
        self.detector.dphi()
    }

    /// Returns the carrier phase of the last detection, in radians.
    pub fn phi(&self) -> f32 {
        self.detector.phi()
    }

    /// Processes input until a block of symbols is ready or the input runs
    /// out.
    ///
    /// When a block is ready, the call returns it at once, and
    /// `consumed` says how much input was used. Call again with
    /// `&input[consumed..]` to continue. When the input runs out first,
    /// `symbols` is empty and `consumed` is `input.len()`.
    ///
    /// Several blocks can become ready from one sample, for example when a
    /// detection releases buffered input. The extra blocks come back from the
    /// following calls with `consumed` of zero, even when `input` is empty.
    /// Keep calling until `symbols` is empty and all input is consumed.
    ///
    /// This never returns an error.
    ///
    pub fn execute<'a>(&'a mut self, input: &[Complex32]) -> Result<QuadratureSynchronizerOutput<'a>> {
        // deliver a full block already buffered from a previous call before
        // consuming any new input
        if self.pending.len() >= self.buf_out_len {
            self.take_block();
            return Ok(QuadratureSynchronizerOutput { consumed: 0, symbols: &self.block });
        }

        for (i, &x) in input.iter().enumerate() {
            match self.state {
                State::Detect => self.execute_detect(x)?,
                State::Sync => self.step(x),
            }

            if self.pending.len() >= self.buf_out_len {
                self.take_block();
                return Ok(QuadratureSynchronizerOutput { consumed: i + 1, symbols: &self.block });
            }
        }

        Ok(QuadratureSynchronizerOutput { consumed: input.len(), symbols: &[] })
    }

    // move one block of recovered symbols from the pending queue into `block`
    fn take_block(&mut self) {
        self.block.clear();
        for _ in 0..self.buf_out_len {
            self.block.push(self.pending.pop_front().unwrap());
        }
    }

    /// Processes input and copies whole blocks of symbols into `symbols`.
    ///
    /// Processing stops when the input runs out or `symbols` has no room for
    /// another block of [`Self::block_len`]. Space after the last whole block is
    /// left unchanged. With room for less than one block, nothing is consumed.
    ///
    /// Undelivered blocks are copied only while there is input left. With an
    /// empty `input`, this returns zero counts even if a block is ready. Use
    /// [`Self::execute`] to collect it.
    ///
    /// This never returns an error.
    ///
    pub fn execute_symbols_into(
        &mut self,
        input: &[Complex32],
        symbols: &mut [Complex32],
    ) -> Result<QuadratureSynchronizerBlockOutput> {
        let mut consumed = 0;
        let mut symbols_written = 0;

        while consumed < input.len() && symbols_written + self.buf_out_len <= symbols.len() {
            let out = self.execute(&input[consumed..])?;
            consumed += out.consumed;
            if !out.symbols.is_empty() {
                symbols[symbols_written..symbols_written + out.symbols.len()].copy_from_slice(out.symbols);
                symbols_written += out.symbols.len();
            } else {
                // input exhausted without a full block
                break;
            }
        }

        Ok(QuadratureSynchronizerBlockOutput { consumed, symbols_written })
    }

    //
    // internal methods
    //

    // execute synchronizer, seeking the preamble sequence
    fn execute_detect(&mut self, x: Complex32) -> Result<()> {
        // push through pre-demod synchronizer; on detection this returns the
        // buffered samples to re-process
        let detected = self.detector.execute(x).map(|v| v.to_vec());

        if let Some(v) = detected {
            // get estimates
            let tau_hat = self.detector.tau().clamp(-0.5, 0.5);
            let gamma_hat = self.detector.gamma();
            let dphi_hat = self.detector.dphi();
            let phi_hat = self.detector.phi();

            // set appropriate filterbank index
            self.mf_counter = self.k as i32 - 2;
            let mut index = (tau_hat * self.npfb as f32) as i32;
            if index < 0 {
                self.mf_counter += 1;
                index += self.npfb as i32;
            }
            self.pfb_index = index as usize;

            // output filter scale
            self.mf.set_scale(Complex32::new(1.0 / (self.k as f32 * gamma_hat), 0.0));

            // set frequency/phase of mixer
            self.mixer.set_frequency(dphi_hat);
            self.mixer.set_phase(phi_hat);

            // update state
            self.state = State::Sync;

            // run buffered samples through the synchronizer (any recovered symbols
            // accumulate; a block filling here is drained on the next execute call)
            for &s in &v {
                self.step(s);
            }
        }
        Ok(())
    }

    // step receiver mixer, matched filter, decimator
    fn step(&mut self, x: Complex32) {
        // mix sample down
        let v = self.mixer.mix_down(x);
        self.mixer.step();

        // push sample into filterbank
        self.mf.push(v);
        let v = self.mf.execute(self.pfb_index).unwrap();

        // increment counter to determine if a sample is available
        self.mf_counter += 1;
        let sample_available = self.mf_counter >= self.k as i32 - 1;

        // output sample if available
        if sample_available {
            // decrement counter by k samples/symbol
            self.mf_counter -= self.k as i32;

            // append to output
            self.buf_append(v);
        }
    }

    // enqueue a recovered symbol
    fn buf_append(&mut self, x: Complex32) {
        // account for filter delay
        self.symbol_counter += 1;
        if self.symbol_counter <= 2 * self.m {
            return;
        }

        self.pending.push_back(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{FirInterpolationFilter, FractionalDelay};
    use crate::random::randnf;
    use std::f32::consts::FRAC_1_SQRT_2;
    use test_macro::autotest_annotate;

    // collect recovered symbols from a QuadratureSynchronizer, capped at `cap`
    fn drain_all(q: &mut QuadratureSynchronizer, buf: &[Complex32], out: &mut Vec<Complex32>, cap: usize) {
        let mut rest = buf;
        while !rest.is_empty() {
            let o = q.execute(rest).unwrap();
            if !o.symbols.is_empty() && out.len() < cap {
                let take = o.symbols.len().min(cap - out.len());
                out.extend_from_slice(&o.symbols[..take]);
            }
            rest = &rest[o.consumed..];
        }
    }

    fn testbench_qdsync_linear(k: usize, m: usize, beta: f32) {
        // options
        let seq_len = 1200; // total number of sync symbols
        let ftype = FirFilterShape::Arkaiser;
        let nstd = 0.001f32;
        let tau = 0.400f32; // fractional sample timing offset

        // generate synchronization sequence (QPSK symbols)
        let mut seq_tx = vec![Complex32::new(0.0, 0.0); seq_len];
        for s in seq_tx.iter_mut() {
            let re = if rand::random::<bool>() { 1.0 } else { -1.0 } * FRAC_1_SQRT_2;
            let im = if rand::random::<bool>() { 1.0 } else { -1.0 } * FRAC_1_SQRT_2;
            *s = Complex32::new(re, im);
        }

        // create sync object, only using the first 240 symbols as preamble
        let mut q = QuadratureSynchronizer::new_linear(&seq_tx[..240], ftype, k, m, beta).unwrap();
        q.set_range(0.001).unwrap();

        // recovered symbols
        let mut seq_rx: Vec<Complex32> = Vec::new();

        // create interpolator and delay
        let mut interp = FirInterpolationFilter::<Complex32>::new_prototype(ftype, k, m, beta, 0.0).unwrap();
        let mut delay = FractionalDelay::<Complex32, f32>::new(100, 8, 64).unwrap();
        delay.set_delay(10.0 * k as f32 + tau).unwrap();

        // run signal through the sync object
        let mut buf = vec![Complex32::new(0.0, 0.0); k];
        for i in 0..(4 * seq_len + 2 * m + 50) {
            let s = if i < seq_len { seq_tx[i] } else { Complex32::new(0.0, 0.0) };
            interp.execute(s, &mut buf).unwrap();
            let input = buf.clone();
            delay.execute_block(&input, &mut buf).unwrap();
            for bj in buf.iter_mut() {
                *bj += nstd * Complex32::new(randnf(), randnf()) * FRAC_1_SQRT_2;
            }
            drain_all(&mut q, &buf, &mut seq_rx, seq_len);
        }

        let rxy_hat = q.rxy();
        let tau_hat = q.tau();
        let gamma_hat = q.gamma();
        let dphi_hat = q.dphi();
        let phi_hat = q.phi();

        // compute error in terms of offset from unity magnitude
        let mut rmse = 0.0f32;
        for &r in &seq_rx {
            let e = r.norm() - 1.0;
            rmse += e * e;
        }
        rmse = 10.0 * (rmse / seq_len as f32).log10();

        assert!(rmse < -30.0, "rmse {} not below -30 dB", rmse);
        assert!(rxy_hat > 0.75, "rxy {} not above 0.75", rxy_hat);
        assert!((tau_hat - tau).abs() < 0.10, "tau error {}", (tau_hat - tau).abs());
        assert!(gamma_hat > 0.75, "gamma {} not above 0.75", gamma_hat);
        assert!(dphi_hat.abs() < 1e-3, "dphi {}", dphi_hat);
        assert!(phi_hat.abs() < 0.4, "phi {}", phi_hat);
    }

    #[test]
    #[autotest_annotate(autotest_qdsync_cccf_k2)]
    fn qdsync_cccf_k2() {
        testbench_qdsync_linear(2, 7, 0.3);
    }

    #[test]
    #[autotest_annotate(autotest_qdsync_cccf_k3)]
    fn qdsync_cccf_k3() {
        testbench_qdsync_linear(3, 7, 0.3);
    }

    #[test]
    #[autotest_annotate(autotest_qdsync_cccf_k4)]
    fn qdsync_cccf_k4() {
        testbench_qdsync_linear(4, 7, 0.3);
    }

    #[test]
    #[autotest_annotate(autotest_qdsync_set_buf_len)]
    fn qdsync_set_buf_len() {
        let seq_len = 2400;
        let k = 2;
        let m = 12;
        let beta = 0.3f32;
        let ftype = FirFilterShape::Arkaiser;

        // generate synchronization sequence (unit-magnitude random phase)
        let mut seq_tx = vec![Complex32::new(0.0, 0.0); seq_len];
        for s in seq_tx.iter_mut() {
            let theta = rand::random::<f32>() * 2.0 * std::f32::consts::PI;
            *s = Complex32::from_polar(1.0, theta);
        }

        // create sync object using only the first 120 symbols as preamble
        let mut q = QuadratureSynchronizer::new_linear(&seq_tx[..120], ftype, k, m, beta).unwrap();
        q.set_range(0.001).unwrap();

        let mut interp = FirInterpolationFilter::<Complex32>::new_prototype(ftype, k, m, beta, 0.0).unwrap();
        let mut seq_rx: Vec<Complex32> = Vec::new();

        let mut buf = vec![Complex32::new(0.0, 0.0); k];
        for i in 0..(seq_len + 20 * m + 200) {
            let s = if i < seq_len { seq_tx[i] } else { Complex32::new(0.0, 0.0) };
            interp.execute(s, &mut buf).unwrap();
            drain_all(&mut q, &buf, &mut seq_rx, seq_len);

            // periodically resize the output block length
            if i % 7 == 0 {
                q.set_block_len(13 + 17 * (i % 11)).unwrap();
            }
        }

        let rxy_hat = q.rxy();
        let gamma_hat = q.gamma();
        let dphi_hat = q.dphi();

        let mut rmse = 0.0f32;
        for &r in &seq_rx {
            let e = r.norm() - 1.0;
            rmse += e * e;
        }
        rmse = 10.0 * (rmse / seq_len as f32).log10();

        assert!(rmse < -30.0, "rmse {rmse} not below -30 dB");
        assert!(rxy_hat > 0.75, "rxy {rxy_hat} not above 0.75");
        assert!(gamma_hat > 0.75, "gamma {gamma_hat} not above 0.75");
        assert!(dphi_hat.abs() < 1e-3, "dphi {dphi_hat}");
    }

    #[test]
    #[autotest_annotate(autotest_qdsync_cccf_copy)]
    fn qdsync_cccf_copy() {
        let seq_len = 2400;
        let split = 1033;
        let k = 2;
        let m = 12;
        let beta = 0.25f32;
        let ftype = FirFilterShape::Arkaiser;
        let nstd = 0.001f32;

        // generate random frame sequence (QPSK)
        let mut seq_tx = vec![Complex32::new(0.0, 0.0); seq_len];
        for s in seq_tx.iter_mut() {
            let re = if rand::random::<bool>() { 1.0 } else { -1.0 } * FRAC_1_SQRT_2;
            let im = if rand::random::<bool>() { 1.0 } else { -1.0 } * FRAC_1_SQRT_2;
            *s = Complex32::new(re, im);
        }

        let mut q_orig = QuadratureSynchronizer::new_linear(&seq_tx[..240], ftype, k, m, beta).unwrap();
        let mut interp = FirInterpolationFilter::<Complex32>::new_prototype(ftype, k, m, beta, 0.0).unwrap();

        let mut rx_orig: Vec<Complex32> = Vec::new();
        let mut rx_copy: Vec<Complex32> = Vec::new();

        let mut buf = vec![Complex32::new(0.0, 0.0); k];
        let make_buf = |interp: &mut FirInterpolationFilter<Complex32>, i: usize, buf: &mut [Complex32]| {
            let s = if i < seq_len { seq_tx[i] } else { Complex32::new(0.0, 0.0) };
            interp.execute(s, buf).unwrap();
            for bj in buf.iter_mut() {
                *bj += nstd * Complex32::new(randnf(), randnf()) * FRAC_1_SQRT_2;
            }
        };

        // feed the first part through the original only
        for i in 0..split {
            make_buf(&mut interp, i, &mut buf);
            drain_all(&mut q_orig, &buf, &mut rx_orig, seq_len);
        }

        let mut q_copy = q_orig.clone();
        rx_copy.extend_from_slice(&rx_orig);

        // run the remaining stream through both
        for i in split..(seq_len + 20 * m * 40) {
            make_buf(&mut interp, i, &mut buf);
            let saved = buf.clone();
            drain_all(&mut q_orig, &saved, &mut rx_orig, seq_len);
            drain_all(&mut q_copy, &saved, &mut rx_copy, seq_len);
        }

        // both should have recovered the full sequence, identically
        assert_eq!(rx_orig.len(), seq_len);
        assert_eq!(rx_copy.len(), seq_len);
        assert_eq!(rx_orig, rx_copy);
    }

    #[test]
    #[autotest_annotate(autotest_qdsync_cccf_config)]
    fn qdsync_cccf_config() {
        // invalid create: empty sequence
        assert!(QuadratureSynchronizer::new_linear(&[], FirFilterShape::Arkaiser, 4, 12, 0.25).is_err());

        // create a proper object and test configurations
        let seq: Vec<Complex32> = [
            1.0f32, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
            -1.0,
        ]
        .iter()
        .map(|&r| Complex32::new(r, 0.0))
        .collect();
        let mut q = QuadratureSynchronizer::new_linear(&seq, FirFilterShape::Arkaiser, 4, 12, 0.25).unwrap();

        // set/get threshold
        q.set_threshold(0.654321).unwrap();
        assert_eq!(q.threshold(), 0.654321);

        // set/get range
        q.set_range(0.007220).unwrap();
        assert_eq!(q.range(), 0.007220);

        // set invalid buffer length
        assert!(q.set_block_len(0).is_err());
        // valid buffer length
        assert!(q.set_block_len(32).is_ok());
        assert_eq!(q.block_len(), 32);

        // query properties
        assert!(!q.is_detected());

        // empty input produces no symbols and leaves output unchanged
        let sentinel = Complex32::new(-9.0, -9.0);
        let mut symbols = [sentinel; 64];
        let output = q.execute_symbols_into(&[], &mut symbols).unwrap();
        assert_eq!(output.consumed, 0);
        assert_eq!(output.symbols_written, 0);
        assert_eq!(symbols, [sentinel; 64]);
    }
}
