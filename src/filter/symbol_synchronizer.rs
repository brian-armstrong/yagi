use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::IirSecondOrderSection;
use crate::filter::{self, FirFilterShape, FirPolyphaseFilter, Sample};

/// Symbol timing recovery for a stream with a known nominal symbol rate.
///
/// The synchronizer runs the input through a polyphase matched filter and
/// picks the filter phase closest to the ideal sampling instant. A second
/// filter bank holds the derivative of the matched filter. Its output drives
/// a timing-error estimate, which a loop filter smooths once per symbol to
/// steer the sampling phase and rate.
///
/// [`Self::execute`] consumes input samples and produces output at the rate
/// set with [`Self::set_output_rate`], one sample per symbol by default. Each
/// output is the matched-filter output divided by `samples_per_symbol`.
#[derive(Clone, Debug)]
#[doc(alias = "Symsync")]
pub struct SymbolSynchronizer<T> {
    k: usize,     // samples/symbol (input)
    k_out: usize, // samples/symbol (output)

    npfb: usize,                     // number of filters in symsync
    mf: FirPolyphaseFilter<T, f32>,  // matched filter
    dmf: FirPolyphaseFilter<T, f32>, // derivative matched filter
    b: usize,                        // filterbank index
    bf: f32,                         // filterbank index (fractional)
    tau: f32,                        // fractional sample offset
    tau_decim: f32,                  // fractional sample offset (decimated)

    rate: f32, // internal resampling rate
    del: f32,  // fractional delay step

    q: f32,                          // timing error
    q_hat: f32,                      // filtered timing error
    decim_counter: usize,            // decimation counter
    pll: IirSecondOrderSection<f32>, // loop filter
    rate_adjustment: f32,            // rate adjustment factor
    is_locked: bool,                 // synchronizer locked flag
}

impl<T> SymbolSynchronizer<T>
where
    T: Sample + std::ops::Mul<f32, Output = T>,
    [T]: DotProd<f32, Output = T>,
{
    /// Creates a symbol synchronizer from matched-filter coefficients.
    ///
    /// `samples_per_symbol` is the nominal number of input samples per symbol.
    /// `coefficients` is the prototype for all `num_filters` phases, in time
    /// order, and its length must be `m * num_filters + 1` for some `m >= 1`.
    ///
    /// The derivative filter is computed from the coefficients by central
    /// differences. When every coefficient times its derivative is zero, as
    /// with constant coefficients, the derivative filter becomes nonfinite
    /// and no error is returned.
    ///
    /// The synchronizer starts unlocked, with one output sample per symbol and
    /// a loop bandwidth of 0.01.
    ///
    /// # Errors
    ///
    /// Returns an error when `samples_per_symbol` is less than two,
    /// `num_filters` is zero, `coefficients` is empty, or its length does not
    /// have the required form.
    ///
    /// # Panics
    ///
    /// Panics when `coefficients` has exactly one element. That length passes
    /// the form check, but the derivative needs at least two taps.
    pub fn new(samples_per_symbol: usize, num_filters: usize, coefficients: &[f32]) -> Result<Self> {
        if samples_per_symbol < 2 {
            return Err(Error::Config("samples/symbol must be at least 2".into()));
        }
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than 0".into()));
        }
        if coefficients.is_empty() {
            return Err(Error::Config("filter length must be greater than 0".into()));
        }
        let h_len = coefficients.len();
        if !(h_len - 1).is_multiple_of(num_filters) {
            return Err(Error::Config("filter length must be of the form: h_len = m*k + 1".into()));
        }

        // same as set_output_rate(1)
        let k_out = 1;
        let rate = samples_per_symbol as f32 / k_out as f32;
        let del = rate;

        let npfb = num_filters;

        let mut dh = vec![0.0f32; h_len];
        let mut hdh_max = 0.0;
        for i in 0..h_len {
            if i == 0 {
                dh[i] = coefficients[i + 1] - coefficients[h_len - 1];
            } else if i == h_len - 1 {
                dh[i] = coefficients[0] - coefficients[i - 1];
            } else {
                dh[i] = coefficients[i + 1] - coefficients[i - 1];
            }

            if (coefficients[i] * dh[i]).abs() > hdh_max || i == 0 {
                hdh_max = (coefficients[i] * dh[i]).abs();
            }
        }

        for dhi in &mut dh[..h_len] {
            *dhi *= 0.06f32 / hdh_max;
        }

        let mf = FirPolyphaseFilter::new(npfb, coefficients)?;
        let dmf = FirPolyphaseFilter::new(npfb, &dh)?;

        let a_coeff = [1.0, 0.0, 0.0];
        let b_coeff = [0.0, 0.0, 0.0];
        let pll = IirSecondOrderSection::new(&b_coeff, &a_coeff)?;

        let mut q = Self {
            k: samples_per_symbol,
            k_out,
            npfb,
            mf,
            dmf,
            b: 0,
            bf: 0.0,
            tau: 0.0,
            tau_decim: 0.0,
            rate,
            del,
            q: 0.0,
            q_hat: 0.0,
            decim_counter: 0,
            pll,
            rate_adjustment: 0.0f32,
            is_locked: false,
        };

        q.reset();
        q.set_lf_bw(0.01f32)?;
        q.unlock();

        Ok(q)
    }

    /// Creates a symbol synchronizer from a pulse-shaping prototype.
    ///
    /// The matched filter is designed from `filter_shape`, usually a
    /// root-Nyquist shape that matches the transmitter.
    ///
    /// # Arguments
    ///
    /// - `samples_per_symbol`: nominal input samples per symbol, at least two.
    /// - `filter_delay`: filter span on each side of its center, in symbols.
    ///   Must be nonzero.
    /// - `excess_bandwidth`: rolloff factor, in `[0, 1]`.
    /// - `num_filters`: number of filter phases. The timing resolution is
    ///   `1 / num_filters` of an input sample. Must be nonzero.
    ///
    /// # Errors
    ///
    /// Returns an error when `samples_per_symbol` is less than two,
    /// `filter_delay` or `num_filters` is zero, `excess_bandwidth` is outside
    /// its range, or the selected prototype rejects the configuration.
    ///
    /// # Panics
    ///
    /// Panics when `samples_per_symbol * num_filters` or the filter length
    /// overflows `usize` and overflow checks are enabled. Without overflow
    /// checks, the length wraps and the design is invalid.
    pub fn new_rnyquist(
        filter_shape: FirFilterShape,
        samples_per_symbol: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
        num_filters: usize,
    ) -> Result<Self> {
        if samples_per_symbol < 2 {
            return Err(Error::Config("samples/symbol must be at least 2".into()));
        }
        if filter_delay == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if excess_bandwidth < 0.0 || excess_bandwidth > 1.0 {
            return Err(Error::Config("excess bandwidth factor must be in [0,1]".into()));
        }
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than 0".into()));
        }

        let h = filter::fir::design::fir_design_prototype(
            filter_shape,
            samples_per_symbol * num_filters,
            filter_delay,
            excess_bandwidth,
            0.0,
        )?;

        Self::new(samples_per_symbol, num_filters, &h)
    }

    /// Creates a symbol synchronizer with a Kaiser-windowed matched filter.
    ///
    /// # Arguments
    ///
    /// - `samples_per_symbol`: nominal input samples per symbol, at least two.
    /// - `filter_delay`: filter span on each side of its center, in symbols.
    ///   Must be nonzero.
    /// - `excess_bandwidth`: must lie in `(0, 1]`, but does not change the
    ///   filter.
    /// - `num_filters`: number of filter phases. The timing resolution is
    ///   `1 / num_filters` of an input sample. Must be nonzero.
    ///
    /// The cutoff is fixed at `0.75 / samples_per_symbol` cycles per input
    /// sample and the stopband attenuation at 40 dB.
    ///
    /// # Errors
    ///
    /// Returns an error when `samples_per_symbol` is less than two,
    /// `filter_delay` or `num_filters` is zero, or `excess_bandwidth` is
    /// outside its range.
    ///
    /// # Panics
    ///
    /// Panics when the filter length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps and the design is
    /// invalid.
    pub fn new_kaiser(
        samples_per_symbol: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
        num_filters: usize,
    ) -> Result<Self> {
        if samples_per_symbol < 2 {
            return Err(Error::Config("samples/symbol must be at least 2".into()));
        }
        if filter_delay == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if excess_bandwidth <= 0.0 || excess_bandwidth > 1.0 {
            return Err(Error::Config("excess bandwidth factor must be in [0,1]".into()));
        }
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than 0".into()));
        }

        let h_len = 2 * num_filters * samples_per_symbol * filter_delay + 1;

        let fc = 0.75f32;
        let as_ = 40.0f32;
        let mut h = filter::fir::design::fir_design_kaiser(
            h_len,
            fc / (samples_per_symbol as f32 * num_filters as f32),
            as_,
            0.0,
        )?;

        for c in h.iter_mut() {
            *c *= 2.0 * fc;
        }

        Self::new(samples_per_symbol, num_filters, &h)
    }

    /// Clears the timing and loop state and the history of both matched
    /// filters.
    ///
    /// The timing increment returns to its nominal value for the current output
    /// rate, which discards any [`Self::set_fractional_rate`] setting. The lock
    /// setting, loop bandwidth, and output rate are kept.
    pub fn reset(&mut self) {
        self.mf.reset();
        self.dmf.reset();
        self.rate = self.k as f32 / self.k_out as f32;
        self.del = self.rate;
        self.b = 0;
        self.bf = 0.0;
        self.tau = 0.0;
        self.tau_decim = 0.0;
        self.q = 0.0;
        self.q_hat = 0.0;
        self.decim_counter = 0;
        self.pll.reset();
    }

    /// Freezes the timing loop at its current rate and phase.
    ///
    /// The synchronizer keeps producing output but stops updating its timing
    /// estimate.
    pub fn lock(&mut self) {
        self.is_locked = true;
    }

    /// Resumes timing-loop updates after [`Self::lock`].
    pub fn unlock(&mut self) {
        self.is_locked = false;
    }

    /// Returns `true` when the timing loop is frozen by [`Self::lock`].
    pub fn is_locked(&self) -> bool {
        self.is_locked
    }

    /// Sets the number of output samples per symbol.
    ///
    /// The timing increment returns to its nominal value for the new rate,
    /// which discards any loop correction. The filter history and timing phase
    /// are kept.
    ///
    /// The count of outputs since the last timing update is also kept. The
    /// loop updates when that count reaches the output rate, so lowering the
    /// rate below the current count stops timing recovery for good. Call
    /// [`Self::reset`] after lowering the output rate to clear the count.
    ///
    /// # Errors
    ///
    /// Returns an error when `output_rate` is zero.
    pub fn set_output_rate(&mut self, output_rate: usize) -> Result<()> {
        if output_rate == 0 {
            return Err(Error::Config("output rate must be greater than 0".into()));
        }
        self.k_out = output_rate;
        self.rate = self.k as f32 / self.k_out as f32;
        self.del = self.rate;
        Ok(())
    }

    /// Sets the number of input samples consumed per output sample.
    ///
    /// Use this for an output rate that is not a whole number of samples per
    /// symbol. It replaces the increment set by [`Self::set_output_rate`], and
    /// [`Self::reset`] restores that increment.
    ///
    /// # Errors
    ///
    /// Returns an error when `rate` is zero or negative.
    pub fn set_fractional_rate(&mut self, rate: f32) -> Result<()> {
        if rate <= 0.0 {
            return Err(Error::Config("rate must be greater than 0".into()));
        }
        self.rate = rate;
        self.del = self.rate;
        Ok(())
    }

    /// Sets the normalized timing-loop bandwidth.
    ///
    /// `bandwidth` must lie in `[0, 1]`. The default is 0.01.
    ///
    /// # Errors
    ///
    /// Returns an error when `bandwidth` is outside `[0, 1]`. The loop is
    /// unchanged on error.
    pub fn set_lf_bw(&mut self, bandwidth: f32) -> Result<()> {
        if bandwidth < 0.0 || bandwidth > 1.0 {
            return Err(Error::Config("bandwidth must be in [0,1]".into()));
        }

        let alpha = 1.0 - bandwidth;
        let beta = 0.22 * bandwidth;
        let a = 0.5;
        let b = 0.495;

        let b_coeff = [beta, 0.0, 0.0];
        let a_coeff = [1.0 - a * alpha, -b * alpha, 0.0];

        self.pll.set_coefficients(&b_coeff, &a_coeff)?;

        self.rate_adjustment = 0.5 * bandwidth;
        Ok(())
    }

    /// Returns the fractional timing offset, in input samples, at the most
    /// recent loop update.
    #[allow(clippy::misnamed_getters)]
    pub fn tau(&self) -> f32 {
        self.tau_decim
    }

    /// Returns the number of input samples consumed per output sample.
    ///
    /// This is the nominal rate plus the slow correction the loop has
    /// accumulated.
    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// Returns the timing increment used for the next output sample.
    ///
    /// This is [`Self::rate`] plus the current filtered timing error.
    pub fn del(&self) -> f32 {
        self.del
    }

    /// Returns the most recent timing-error estimate, clamped to `[-1, 1]`.
    pub fn q(&self) -> f32 {
        self.q
    }

    /// Returns the timing-error estimate after the loop filter.
    pub fn q_hat(&self) -> f32 {
        self.q_hat
    }

    /// Synchronizes a block of input samples and returns the number of outputs.
    ///
    /// The count depends on the timing state and is roughly
    /// `input.len() / rate()`, so leave some room in `output` beyond that.
    /// Only the returned number of samples is written.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` fills before all input is consumed. By
    /// then the synchronizer has consumed part of the input and written part of
    /// `output`, and the number of samples written is lost.
    pub fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let mut ny = 0;

        for &xi in input.iter() {
            let k = self.step(xi, &mut output[ny..])?;
            ny += k;
        }

        Ok(ny)
    }

    fn step(&mut self, x: T, y: &mut [T]) -> Result<usize> {
        self.mf.push(x);
        self.dmf.push(x);

        let mut mf: T;
        let mut dmf: T;
        let mut ny = 0;

        while self.b < self.npfb {
            if ny >= y.len() {
                return Err(Error::Range("symsync output buffer full".into()));
            }

            mf = self.mf.execute(self.b)?;
            y[ny] = mf / (self.k as f32).into();

            if self.decim_counter == self.k_out {
                self.decim_counter = 0;

                if self.is_locked {
                    continue;
                }

                dmf = self.dmf.execute(self.b)?;
                self.advance_internal_loop(mf, dmf);
                self.tau_decim = self.tau;
            }

            self.decim_counter += 1;
            self.tau += self.del;
            self.bf = self.tau * self.npfb as f32;
            self.b = self.bf.round() as usize;
            ny += 1;
        }

        self.tau -= 1.0;
        self.bf -= self.npfb as f32;
        self.b -= self.npfb;

        Ok(ny)
    }

    fn advance_internal_loop(&mut self, mf: T, dmf: T) {
        self.q = (mf.conj() * dmf).re();
        self.q = self.q.clamp(-1.0, 1.0);

        self.q_hat = self.pll.execute(self.q);

        self.rate += self.rate_adjustment * self.q_hat;
        self.del = self.rate + self.q_hat;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::ArbitraryResampler;
    use crate::filter::FirInterpolationFilter;
    use crate::random::randnf;
    use crate::sequence::MaximalLengthSequence;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_symsync_copy)]
    fn test_symsync_copy() {
        // create base object
        let mut q0 = SymbolSynchronizer::<Complex32>::new_rnyquist(FirFilterShape::Arkaiser, 5, 7, 0.25, 64).unwrap();
        q0.set_lf_bw(0.02).unwrap();

        // run samples through filter
        // NOTE: we don't care that the input is noise; just that both objects
        //       produce the same output
        let buf_len = 640;
        let mut buf = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); buf_len];

        for i in 0..buf_len {
            buf[i] = Complex32::new(randnf(), randnf());
        }

        let _nw_0 = q0.execute(&buf, &mut buf_0).unwrap();

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters and check equality
        for i in 0..buf_len {
            buf[i] = Complex32::new(randnf(), randnf());
        }

        let nw_0 = q0.execute(&buf, &mut buf_0).unwrap();
        let nw_1 = q1.execute(&buf, &mut buf_1).unwrap();

        // check that the same number of samples were written
        assert_eq!(nw_0, nw_1);

        // check output sample values
        assert_eq!(buf_0[..nw_0], buf_1[..nw_1]);

        // check other internal properties
        assert_eq!(q0.tau(), q1.tau());
    }

    #[test]
    fn test_symsync_reset_matches_new() {
        let new = || {
            let mut q =
                SymbolSynchronizer::<Complex32>::new_rnyquist(FirFilterShape::Arkaiser, 5, 7, 0.25, 64).unwrap();
            q.set_lf_bw(0.02).unwrap();
            q
        };

        let buf_len = 640;
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); buf_len];

        let mut q0 = new();
        let warmup: Vec<_> = (0..buf_len).map(|_| Complex32::new(randnf(), randnf())).collect();
        q0.execute(&warmup, &mut buf_0).unwrap();
        q0.reset();

        let mut q1 = new();

        let buf: Vec<_> = (0..buf_len).map(|_| Complex32::new(randnf(), randnf())).collect();
        let nw_0 = q0.execute(&buf, &mut buf_0).unwrap();
        let nw_1 = q1.execute(&buf, &mut buf_1).unwrap();

        assert_eq!(nw_0, nw_1);
        assert_eq!(buf_0[..nw_0], buf_1[..nw_1]);
        assert_eq!(q0.tau(), q1.tau());
    }

    // test errors and invalid configuration
    #[test]
    #[autotest_annotate(autotest_symsync_config)]
    fn test_symsync_config() {
        // test copying/creating invalid objects
        // assert!(SymbolSynchronizer::<Complex32>::copy(&None).is_err());

        assert!(SymbolSynchronizer::<Complex32>::new(0, 12, &[]).is_err()); // k is too small
        assert!(SymbolSynchronizer::<Complex32>::new(2, 0, &[]).is_err()); // M is too small
        assert!(SymbolSynchronizer::<Complex32>::new(2, 12, &[]).is_err()); // coefficients cannot be empty
        assert!(SymbolSynchronizer::<Complex32>::new(2, 12, &[0.0; 48]).is_err()); // coefficient count is invalid

        assert!(SymbolSynchronizer::<Complex32>::new_rnyquist(FirFilterShape::Rrcos, 0, 12, 0.2, 48).is_err()); // k is too small
        assert!(SymbolSynchronizer::<Complex32>::new_rnyquist(FirFilterShape::Rrcos, 2, 0, 0.2, 48).is_err()); // m is too small
        assert!(SymbolSynchronizer::<Complex32>::new_rnyquist(FirFilterShape::Rrcos, 2, 12, 7.2, 48).is_err()); // beta is too large
        assert!(SymbolSynchronizer::<Complex32>::new_rnyquist(FirFilterShape::Rrcos, 2, 12, 0.2, 0).is_err()); // M is too small

        assert!(SymbolSynchronizer::<Complex32>::new_kaiser(0, 12, 0.2, 48).is_err()); // k is too small
        assert!(SymbolSynchronizer::<Complex32>::new_kaiser(2, 0, 0.2, 48).is_err()); // m is too small
        assert!(SymbolSynchronizer::<Complex32>::new_kaiser(2, 12, 7.2, 48).is_err()); // beta is too large
        assert!(SymbolSynchronizer::<Complex32>::new_kaiser(2, 12, 0.2, 0).is_err()); // M is too small

        // create valid object
        let mut q = SymbolSynchronizer::<Complex32>::new_kaiser(2, 12, 0.2, 48).unwrap();
        // assert!(q.print().is_ok());

        // check lock state
        q.lock();
        assert!(q.is_locked());
        q.unlock();
        assert!(!q.is_locked());

        // check invalid properties
        assert!(q.set_output_rate(0).is_err());
        assert!(q.set_lf_bw(-1.0).is_err());
    }

    fn symsync_crcf_test(method: &str, _k: usize, _m: usize, _beta: f32, _tau: f32, _rate: f32) {
        // options
        let tol: f32 = 0.2; // error tolerance
        let k: usize = _k; // samples/symbol (input)
        let m: usize = _m; // filter delay (symbols)
        let beta: f32 = _beta; // filter excess bandwidth factor
        let num_filters: usize = 32; // number of filters in the bank

        let num_symbols_init: usize = 200; // number of initial symbols
        let num_symbols_test: usize = 100; // number of testing symbols

        // transmit filter type
        let ftype_tx = if method == "rnyquist" { FirFilterShape::Arkaiser } else { FirFilterShape::Kaiser };

        let bt: f32 = 0.02; // loop filter bandwidth
        let mut tau: f32 = _tau; // fractional symbol offset
        let rate: f32 = _rate; // resampled rate

        // derived values
        let num_symbols = num_symbols_init + num_symbols_test;
        let num_samples = k * num_symbols;
        let num_samples_resamp = (num_samples as f32 * rate * 1.1).ceil() as u32 + 4;

        // compute delay
        while tau < 0.0 {
            // ensure positive tau
            tau += 1.0;
        }
        let g = k as f32 * tau; // number of samples offset
        let ds = g.floor() as i32; // additional symbol delay
        let mut dt = g - ds as f32; // fractional sample offset
        if dt > 0.5 {
            // force dt to be in [0.5,0.5]
            dt -= 1.0;
        }

        // allocate arrays
        let mut s = vec![Complex32::new(0.0, 0.0); num_symbols as usize]; // data symbols
        let mut x = vec![Complex32::new(0.0, 0.0); num_samples as usize]; // interpolated samples
        let mut y = vec![Complex32::new(0.0, 0.0); num_samples_resamp as usize]; // resampled data (resamp_crcf)
        let mut z = vec![Complex32::new(0.0, 0.0); num_symbols as usize + 64]; // synchronized symbols

        // generate pseudo-random QPSK symbols
        // NOTE: by using an m-sequence generator this sequence will be identical
        //       each time this test is run
        let mut ms = MaximalLengthSequence::from_degree(10).unwrap();
        for i in 0..num_symbols as usize {
            let si = ms.generate_symbol(1);
            let sq = ms.generate_symbol(1);
            s[i] = Complex32::new(
                if si == 0 { 1.0 } else { -1.0 } * std::f32::consts::FRAC_1_SQRT_2,
                if sq == 0 { 1.0 } else { -1.0 } * std::f32::consts::FRAC_1_SQRT_2,
            );
        }

        //
        // create and run interpolator
        //

        // design interpolating filter
        let mut interp = FirInterpolationFilter::<Complex32>::new_prototype(ftype_tx, k as usize, m, beta, dt).unwrap();

        // interpolate block of samples
        interp.execute_block(&s[..num_symbols as usize], &mut x[..num_samples as usize]).unwrap();

        //
        // run resampler
        //

        // create resampler
        let resamp_len = 10 * k; // resampling filter semi-length (filter delay)
        let resamp_bw = 0.45; // resampling filter bandwidth
        let resamp_as = 60.0; // resampling filter stop-band attenuation
        let resamp_npfb = 64; // number of filters in bank
        let mut resamp =
            ArbitraryResampler::<Complex32>::new(rate, resamp_len, resamp_bw, resamp_as, resamp_npfb).unwrap();

        // run resampler on block
        let ny = resamp.execute_block(&x[..num_samples as usize], &mut y[..]).unwrap();

        //
        // create and run symbol synchronizer
        //

        // create symbol synchronizer
        let mut sync = if method == "rnyquist" {
            SymbolSynchronizer::<Complex32>::new_rnyquist(ftype_tx, k, m, beta, num_filters).unwrap()
        } else {
            SymbolSynchronizer::<Complex32>::new_kaiser(k, m, beta, num_filters).unwrap()
        };

        // set loop filter bandwidth
        sync.set_lf_bw(bt).unwrap();

        // execute on entire block of samples
        let nz = sync.execute(&y[..ny], &mut z).unwrap();

        // compute total delay through system
        // (initial filter + resampler + matched filter)
        let delay = m + 10 + m;

        // compare (and print) results
        for i in (nz - num_symbols_test as usize)..nz {
            // compute error
            let err = (z[i] - s[i - delay as usize]).norm();

            // assert that error is below tolerance
            assert!(err < tol, "Error {} exceeds tolerance {} at index {}", err, tol, i);
        }
    }

    // autotest scenarios (root-Nyquist)
    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_0)]
    fn symsync_crcf_scenario_0() {
        symsync_crcf_test("rnyquist", 2, 7, 0.35, 0.00, 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_1)]
    fn symsync_crcf_scenario_1() {
        symsync_crcf_test("rnyquist", 2, 7, 0.35, -0.25, 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_2)]
    fn symsync_crcf_scenario_2() {
        symsync_crcf_test("rnyquist", 2, 7, 0.35, -0.25, 1.0001);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_3)]
    fn symsync_crcf_scenario_3() {
        symsync_crcf_test("rnyquist", 2, 7, 0.35, -0.25, 0.9999);
    }

    // autotest scenarios (Nyquist)
    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_4)]
    fn symsync_crcf_scenario_4() {
        symsync_crcf_test("nyquist", 2, 7, 0.35, 0.00, 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_5)]
    fn symsync_crcf_scenario_5() {
        symsync_crcf_test("nyquist", 2, 7, 0.35, -0.25, 1.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_6)]
    fn symsync_crcf_scenario_6() {
        symsync_crcf_test("nyquist", 2, 7, 0.35, -0.25, 1.0001);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_crcf_scenario_7)]
    fn symsync_crcf_scenario_7() {
        symsync_crcf_test("nyquist", 2, 7, 0.35, -0.25, 0.9999);
    }

    #[derive(Clone, Copy)]
    enum ErrorLimit {
        Peak(f32),
        Rms(f32),
    }

    fn symsync_rrrf_test_peak(method: &str, k: usize, m: usize, beta: f32, tau: f32, rate: f32, expected_rate: f32) {
        symsync_rrrf_test_with_limit(method, k, m, beta, tau, rate, expected_rate, ErrorLimit::Peak(0.20));
    }

    fn symsync_rrrf_test_rms(method: &str, k: usize, m: usize, beta: f32, tau: f32, rate: f32, expected_rate: f32) {
        symsync_rrrf_test_with_limit(method, k, m, beta, tau, rate, expected_rate, ErrorLimit::Rms(0.10));
    }

    fn symsync_rrrf_test_with_limit(
        method: &str,
        k: usize,
        m: usize,
        beta: f32,
        tau: f32,
        rate: f32,
        expected_rate: f32,
        error_limit: ErrorLimit,
    ) {
        // options
        let num_filters = 32; // number of filters in the bank

        let num_symbols_init = 400; // number of initial symbols
        let num_symbols_test = 100; // number of testing symbols

        // transmit filter type
        let ftype_tx = if method == "rnyquist" { FirFilterShape::Arkaiser } else { FirFilterShape::Kaiser };

        let bt = 0.01f32; // loop filter bandwidth
        let mut tau = tau; // fractional symbol offset
        let rate = rate; // resampled rate

        // derived values
        let num_symbols = num_symbols_init + num_symbols_test;
        let num_samples = k * num_symbols;
        let num_samples_resamp = (num_samples as f32 * rate * 1.1f32).ceil() as usize + 4;

        // compute delay
        while tau < 0.0 {
            // ensure positive tau
            tau += 1.0;
        }
        let g = k as f32 * tau; // number of samples offset
        let ds = g.floor() as i32; // additional symbol delay
        let mut dt = g - ds as f32; // fractional sample offset
        if dt > 0.5 {
            // force dt to be in [0.5,0.5]
            dt -= 1.0;
        }

        // allocate arrays
        let mut s = vec![0.0f32; num_symbols]; // data symbols
        let mut x = vec![0.0f32; num_samples]; // interpolated samples
        let mut y = vec![0.0f32; num_samples_resamp]; // resampled data (resamp_rrrf)
        let mut z = vec![0.0f32; num_symbols + 64]; // synchronized symbols

        // generate pseudo-random BPSK symbols
        // NOTE: by using an m-sequence generator this sequence will be identical
        //       each time this test is run
        let mut ms = MaximalLengthSequence::from_degree(10).unwrap();
        for i in 0..num_symbols {
            s[i] = if ms.generate_symbol(1) == 0 { 1.0 } else { -1.0 };
        }

        //
        // create and run interpolator
        //

        // design interpolating filter
        let mut interp = FirInterpolationFilter::<f32>::new_prototype(ftype_tx, k as usize, m, beta, dt).unwrap();

        // interpolate block of samples
        interp.execute_block(&s[..num_symbols], &mut x).unwrap();

        //
        // run resampler
        //

        // create resampler
        let resamp_len = 10 * k; // resampling filter semi-length (filter delay)
        let resamp_bw = 0.45f32; // resampling filter bandwidth
        let resamp_as = 60.0f32; // resampling filter stop-band attenuation
        let resamp_npfb = 64; // number of filters in bank
        let mut resamp = ArbitraryResampler::<f32>::new(rate, resamp_len, resamp_bw, resamp_as, resamp_npfb).unwrap();

        // run resampler on block
        let ny = resamp.execute_block(&x[..num_samples], &mut y).unwrap();

        //
        // create and run symbol synchronizer
        //

        // create symbol synchronizer
        let mut sync = if method == "rnyquist" {
            SymbolSynchronizer::<f32>::new_rnyquist(ftype_tx, k, m, beta, num_filters).unwrap()
        } else {
            SymbolSynchronizer::<f32>::new_kaiser(k, m, beta, num_filters).unwrap()
        };

        // set loop filter bandwidth
        sync.set_lf_bw(bt).unwrap();

        // set fractional rate
        sync.set_fractional_rate(expected_rate).unwrap();

        // execute on entire block of samples
        let nz = sync.execute(&y[..ny], &mut z).unwrap();

        // compute total delay through system
        // (initial filter + resampler + matched filter)
        let delay = m + 10 + m;

        // compare results
        let mut error_squared = 0.0f32;
        for i in (nz - num_symbols_test)..nz {
            let err = (z[i] - s[i - delay]).abs();
            error_squared += err * err;
            if let ErrorLimit::Peak(tolerance) = error_limit {
                assert!(err < tolerance, "Error {} exceeds tolerance {}", err, tolerance);
            }
        }
        if let ErrorLimit::Rms(tolerance) = error_limit {
            let rms_error = (error_squared / num_symbols_test as f32).sqrt();
            assert!(rms_error < tolerance, "RMS error {} exceeds tolerance {}", rms_error, tolerance);
        }
    }

    // autotest scenarios (root-Nyquist)
    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_0)]
    fn symsync_rrrf_scenario_0() {
        symsync_rrrf_test_peak("rnyquist", 2, 7, 0.35, 0.00, 1.0, 2.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_1)]
    fn symsync_rrrf_scenario_1() {
        symsync_rrrf_test_peak("rnyquist", 2, 7, 0.35, -0.25, 1.0, 2.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_2)]
    fn symsync_rrrf_scenario_2() {
        symsync_rrrf_test_peak("rnyquist", 2, 7, 0.35, -0.25, 1.0001, 2.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_3)]
    fn symsync_rrrf_scenario_3() {
        symsync_rrrf_test_peak("rnyquist", 2, 7, 0.35, -0.25, 0.9999, 2.0);
    }

    // autotest scenarios (Nyquist)
    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_4)]
    fn symsync_rrrf_scenario_4() {
        symsync_rrrf_test_peak("nyquist", 2, 7, 0.35, 0.00, 1.0, 2.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_5)]
    fn symsync_rrrf_scenario_5() {
        symsync_rrrf_test_peak("nyquist", 2, 7, 0.35, -0.25, 1.0, 2.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_6)]
    fn symsync_rrrf_scenario_6() {
        symsync_rrrf_test_peak("nyquist", 2, 7, 0.35, -0.25, 1.0001, 2.0);
    }

    #[test]
    #[autotest_annotate(autotest_symsync_rrrf_scenario_7)]
    fn symsync_rrrf_scenario_7() {
        symsync_rrrf_test_peak("nyquist", 2, 7, 0.35, -0.25, 0.9999, 2.0);
    }

    #[test]
    fn symsync_rrrf_scenario_8() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.998, 1.996);
    }

    #[test]
    fn symsync_rrrf_scenario_9() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.998, 1.994);
    }

    #[test]
    fn symsync_rrrf_scenario_10() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.998, 1.998);
    }

    #[test]
    fn symsync_rrrf_scenario_11() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.99, 1.98);
    }

    #[test]
    fn symsync_rrrf_scenario_12() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.99, 1.981);
    }

    #[test]
    fn symsync_rrrf_scenario_13() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.99, 1.979);
    }

    #[test]
    fn symsync_rrrf_scenario_14() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.98, 1.96);
    }

    #[test]
    fn symsync_rrrf_scenario_15() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.98, 1.962);
    }

    #[test]
    fn symsync_rrrf_scenario_16() {
        symsync_rrrf_test_rms("nyquist", 2, 7, 0.35, -0.25, 0.98, 1.958);
    }

    // Test that when rate < 1.0 (more outputs than inputs), execute returns an
    // error if the output buffer is too small, rather than panicking.
    #[test]
    fn test_symsync_output_buffer_full() {
        let mut sync = SymbolSynchronizer::<Complex32>::new_kaiser(2, 7, 0.3, 32).unwrap();

        // Set rate to 0.5 - this means ~2 outputs per input
        sync.set_fractional_rate(0.5).unwrap();
        assert!(sync.rate() < 1.0);

        // Input buffer with 100 samples
        let x: Vec<Complex32> = (0..100).map(|i| Complex32::new(i as f32 * 0.1, 0.0)).collect();

        // Output buffer sized for normal decimation (1 output per 2 inputs = 50)
        // With rate=0.5, we'd get ~2 outputs per input, so 100 inputs -> ~200 outputs
        // but we only have space for 50
        let mut y = vec![Complex32::new(0.0, 0.0); 50];

        // This should return an error because the output buffer is too small
        let result = sync.execute(&x, &mut y);
        assert!(result.is_err(), "expected error, got Ok with {} outputs", result.unwrap());
        assert!(result.unwrap_err().to_string().contains("output buffer full"));
    }
}
