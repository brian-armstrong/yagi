use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::{self, FirPolyphaseFilterBank, Sample};
use crate::math::nextpow2;

const PHASE_SCALE: u128 = 1 << 24;

/// Resampler that changes the sample rate by an arbitrary positive ratio.
///
/// The rate is the output rate divided by the input rate, from 0.004 to 250.
/// A polyphase filter bank computes each output with the filter phase at or
/// just before its fractional position between input samples. A new resampler
/// starts with zero-valued history and zero phase.
///
/// The phase advances by a step rounded to 24-bit fixed point, so the actual
/// rate can differ slightly from [`Self::rate`]. The number of outputs for a
/// given input length therefore depends on the current phase. Use
/// [`Self::num_output`] to size an output buffer before each call.
///
/// The filter is normalized so that a constant input settles to a gain of
/// about one.
#[derive(Clone, Debug)]
#[doc(alias = "Resamp")]
pub struct ArbitraryResampler<T, Coeff = T> {
    m: usize,
    r: f32,
    step: u32,
    phase: u32,
    bits_index: usize,
    w: Window<T>,
    bank: FirPolyphaseFilterBank<T, Coeff>,
}

impl<T, Coeff> ArbitraryResampler<T, Coeff>
where
    Coeff: Sample,
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates an arbitrary-rate resampler with a Kaiser-windowed filter.
    ///
    /// # Arguments
    ///
    /// - `rate`: output rate divided by input rate, in `[0.004, 250]`.
    /// - `filter_semi_length`: filter span on each side of the center tap, in
    ///   input samples. Must be nonzero.
    /// - `cutoff_frequency`: in cycles per input sample, in `(0, 0.5)`.
    /// - `stopband_attenuation`: in dB. Must be positive.
    /// - `num_filters`: number of filter phases, rounded up to a power of two
    ///   and then required to lie in `[2, 65536]`. More phases place outputs
    ///   closer to their exact positions.
    ///
    /// When `rate` is below one, keep `cutoff_frequency` below `rate / 2`,
    /// with room for the transition band. A higher cutoff lets frequencies
    /// above the output Nyquist frequency alias. The constructor does not
    /// enforce this.
    ///
    /// `num_filters` is cast to `u32` before rounding, so values above
    /// `u32::MAX` are truncated.
    ///
    /// # Errors
    ///
    /// Returns an error when `rate` is nonfinite or outside `[0.004, 250]`,
    /// `filter_semi_length` is zero, `cutoff_frequency` is nonfinite or outside
    /// its range, `stopband_attenuation` is nonfinite, zero or negative, or
    /// `num_filters` is outside its range.
    ///
    /// # Panics
    ///
    /// Panics when the filter length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps and the design is
    /// invalid.
    pub fn new(
        rate: f32,
        filter_semi_length: usize,
        cutoff_frequency: f32,
        stopband_attenuation: f32,
        num_filters: usize,
    ) -> Result<Self> {
        if !(0.004..=250.0).contains(&rate) {
            return Err(Error::Config("resampling rate must be in [0.004,250]".into()));
        }
        if filter_semi_length == 0 {
            return Err(Error::Config("filter semi-length must be greater than zero".into()));
        }
        if !(cutoff_frequency > 0.0 && cutoff_frequency < 0.5) {
            return Err(Error::Config("filter cutoff must be in (0,0.5)".into()));
        }
        if !stopband_attenuation.is_finite() || stopband_attenuation <= 0.0 {
            return Err(Error::Config("filter stop-band suppression must be finite and greater than zero".into()));
        }

        let bits = nextpow2(num_filters as u32)? as usize;
        if !(1..=16).contains(&bits) {
            return Err(Error::Config("number of filter banks must be in (2^0,2^16)".into()));
        }

        let num_filters = 1 << bits;

        // design filter
        let n = 2 * filter_semi_length * num_filters + 1;
        let hf = filter::fir::design::fir_design_kaiser(
            n,
            cutoff_frequency / (num_filters as f32),
            stopband_attenuation,
            0.0,
        )?;

        // normalize filter coefficients by DC gain
        let gain = hf.iter().sum::<f32>();
        let gain = (num_filters as f32) / gain;

        // copy to type-specific array, applying gain
        let h: Vec<Coeff> = hf.iter().map(|&x| (x * gain).into()).collect();

        let bank = FirPolyphaseFilterBank::new(num_filters, &h)?;
        let w = Window::new(bank.filter_len())?;

        let mut q = Self { m: filter_semi_length, r: rate, step: 0, phase: 0, bits_index: bits, w, bank };

        q.set_rate(rate)?;

        q.reset();
        Ok(q)
    }

    /// Creates an arbitrary-rate resampler with a default filter.
    ///
    /// The filter spans 7 input samples on each side of its center tap, with
    /// a cutoff of `min(0.5 * rate, 0.49)` cycles per input sample, 60 dB of
    /// stopband attenuation, and 64 phases.
    ///
    /// Below a rate of 0.98, the cutoff follows the output Nyquist frequency.
    /// At higher rates, it stays at 0.49. Use [`Self::new`] to choose a narrower
    /// cutoff or a longer filter when the transition band needs more room.
    ///
    /// # Errors
    ///
    /// Returns an error when `rate` is nonfinite or outside `[0.004, 250]`.
    pub fn from_rate(rate: f32) -> Result<Self> {
        let m = 7;
        let fc = (0.5 * rate).min(0.49);
        let as_ = 60.0;
        let npfb = 64;

        Self::new(rate, m, fc, as_, npfb)
    }

    /// Clears the sample history to zeros and sets the phase to zero.
    ///
    /// The rate is unchanged.
    pub fn reset(&mut self) {
        self.phase = 0;
        self.w.reset();
    }

    /// Returns the filter delay in input samples.
    pub fn delay(&self) -> usize {
        self.m
    }

    /// Changes the rate while keeping the history and phase.
    ///
    /// [`Self::rate`] then reports `rate` exactly, while the phase step is
    /// rounded.
    ///
    /// The filter is not redesigned, so the cutoff rule on [`Self::new`] now
    /// applies to the new rate. A resampler built at rate 1 with a cutoff of
    /// 0.4 and then set to rate 0.25 passes a tone at 0.3 cycles per input
    /// sample almost unattenuated, and it aliases. Choose the cutoff for the
    /// lowest rate the resampler will use.
    ///
    /// # Errors
    ///
    /// Returns an error when `rate` is nonfinite or outside `[0.004, 250]`.
    /// The resampler is unchanged on error.
    pub fn set_rate(&mut self, rate: f32) -> Result<()> {
        if !(0.004..=250.0).contains(&rate) {
            return Err(Error::Config("resampling rate must be in [0.004,250]".into()));
        }

        self.r = rate;
        self.step = ((1 << 24) as f32 / self.r).round() as u32;

        Ok(())
    }

    /// Returns the requested output rate divided by the input rate.
    ///
    /// The actual rate can differ slightly because the phase step is rounded.
    pub fn rate(&self) -> f32 {
        self.r
    }

    /// Multiplies the rate by `gamma` while keeping the history and phase.
    ///
    /// The filter is not redesigned, as with [`Self::set_rate`].
    ///
    /// # Errors
    ///
    /// Returns an error when `gamma` is nonfinite, zero or negative, or the
    /// new rate is outside `[0.004, 250]`. The resampler is unchanged on error.
    pub fn adjust_rate(&mut self, gamma: f32) -> Result<()> {
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(Error::Config(format!(
                "resampling adjustment ({}) must be finite and greater than zero",
                gamma
            )));
        }

        self.set_rate(self.r * gamma)
    }

    // pub fn set_timing_phase(&mut self, _tau: f32) -> Result<()> {
    //     unimplemented!()
    // }

    // pub fn adjust_timing_phase(&mut self, _delta: f32) -> Result<()> {
    //     unimplemented!()
    // }

    /// Returns the number of output samples that `num_input` more input
    /// samples produce.
    ///
    /// The count depends on the current phase, so compute it immediately
    /// before each call. It is the output length that [`Self::execute_block`]
    /// requires for that input. A count too large for `usize` saturates at
    /// `usize::MAX`.
    pub fn num_output(&self, num_input: usize) -> usize {
        // after consuming n inputs and producing k outputs, the accumulator is
        // phase + k*step - n*PHASE_SCALE

        let distance = (num_input as u128 * PHASE_SCALE).saturating_sub(self.phase as u128);
        let count = distance.div_ceil(self.step as u128);
        count.min(usize::MAX as u128) as usize
    }

    /// Returns the largest number of input samples whose output fits in
    /// `max_output` samples.
    ///
    /// The count depends on the current phase. A count too large for `usize`
    /// saturates at `usize::MAX`.
    pub fn max_input(&self, max_output: usize) -> usize {
        // k <= max_output when
        // n*PHASE_SCALE <= max_output*step + phase.
        let count = (max_output as u128 * self.step as u128 + self.phase as u128) / PHASE_SCALE;
        count.min(usize::MAX as u128) as usize
    }

    /// Consumes one input sample and writes the outputs it produces.
    ///
    /// Depending on the rate and phase, one input produces zero or more
    /// outputs. `output` must hold at least `num_output(1)` samples. Returns
    /// the number of samples written. The rest of `output` is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is too short. The resampler and `output`
    /// are unchanged on error.
    pub fn execute(&mut self, input: T, output: &mut [T]) -> Result<usize> {
        if output.len() < self.num_output(1) {
            return Err(Error::Config("output length too small for resampler output".into()));
        }
        self.w.push(input);

        let mut n = 0;
        while self.phase <= 0x00ffffff {
            let index = self.phase >> (24 - self.bits_index);
            output[n] = self.bank.execute(index as usize, self.w.read())?;
            n += 1;
            self.phase += self.step;
        }
        self.phase -= 1 << 24;

        Ok(n)
    }

    /// Consumes a block of input samples and writes the outputs they produce.
    ///
    /// `output` must hold at least `num_output(input.len())` samples. Returns
    /// the number of samples written. The rest of `output` is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is too short. The resampler and `output`
    /// are unchanged on error.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if output.len() < self.num_output(input.len()) {
            return Err(Error::Config("output length too small for resampler output".into()));
        }
        let mut ny = 0;
        let step = self.step;
        let bits_index = self.bits_index;
        let phase = &mut self.phase;
        let bank = &self.bank;
        let filter_len = bank.filter_len();
        debug_assert!((0x00ffffff >> (24 - bits_index)) < bank.num_filters());

        self.w.execute_block_contiguous(input, |_, samples| {
            for history in samples.windows(filter_len) {
                while *phase <= 0x00ffffff {
                    let index = *phase >> (24 - bits_index);
                    output[ny] = unsafe { bank.execute_unchecked(index as usize, history) };
                    ny += 1;
                    *phase += step;
                }
                *phase -= 1 << 24;
            }
        });

        Ok(ny)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
    use crate::utility::test_helpers::{validate_psd_signal, PsdRegion};
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    #[test]
    fn test_resamp_from_rate_defaults() {
        for rate in [0.2, 0.731, 0.98, 1.0, 1.414, 2.3] {
            let mut default = ArbitraryResampler::<Complex32>::from_rate(rate).unwrap();
            let mut reference =
                ArbitraryResampler::<Complex32>::new(rate, 7, (0.5 * rate).min(0.49), 60.0, 64).unwrap();
            let input: Vec<_> = (0..257)
                .map(|i| {
                    let t = i as f32;
                    Complex32::new((0.31 * t).sin(), (0.73 * t).cos())
                })
                .collect();
            for block in input.chunks(37) {
                let required = reference.num_output(block.len());
                assert_eq!(default.num_output(block.len()), required);
                let mut expected = vec![Complex32::default(); required];
                let mut actual = vec![Complex32::default(); required];
                assert_eq!(reference.execute_block(block, &mut expected).unwrap(), required);
                assert_eq!(default.execute_block(block, &mut actual).unwrap(), required);
                assert_eq!(actual, expected, "rate={rate}");
            }
        }
    }

    #[test]
    fn test_resamp_rate_validation() {
        for rate in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            -1.0,
            f32::from_bits(0.004f32.to_bits() - 1),
            f32::from_bits(250.0f32.to_bits() + 1),
        ] {
            assert!(ArbitraryResampler::<f32>::new(rate, 7, 0.25, 60.0, 64).is_err());
            assert!(ArbitraryResampler::<f32>::from_rate(rate).is_err());
        }
        for rate in [0.004, 250.0] {
            for q in [
                ArbitraryResampler::<f32>::new(rate, 7, 0.25, 60.0, 64).unwrap(),
                ArbitraryResampler::<f32>::from_rate(rate).unwrap(),
            ] {
                assert_eq!(q.rate(), rate);
                assert_eq!(q.num_output(1), if rate == 250.0 { 250 } else { 1 });
            }
        }
    }

    #[test]
    fn test_resamp_rejects_nonfinite_filter_parameters() {
        for cutoff in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.5] {
            assert!(matches!(ArbitraryResampler::<f32>::new(1.0, 7, cutoff, 60.0, 64), Err(Error::Config(_))));
        }
        for attenuation in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -f32::EPSILON] {
            assert!(matches!(ArbitraryResampler::<f32>::new(1.0, 7, 0.25, attenuation, 64), Err(Error::Config(_))));
        }
    }

    #[test]
    fn test_resamp_rejected_rate_change_preserves_stream() {
        let mut initial = ArbitraryResampler::<f32>::new(1.37, 7, 0.25, 60.0, 64).unwrap();
        let warmup = [1.0, -0.5, 0.25];
        initial.execute_block(&warmup, &mut vec![0.0; initial.num_output(warmup.len())]).unwrap();
        let input = [0.75, -0.25, 0.5, -1.0, 0.0];

        for adjust in [false, true] {
            let mut q = initial.clone();
            let mut reference = initial.clone();
            for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, -1.0, f32::MIN_POSITIVE, f32::MAX] {
                let result = if adjust { q.adjust_rate(rate) } else { q.set_rate(rate) };
                assert!(result.is_err());
                assert_eq!(q.rate(), reference.rate());

                let required = reference.num_output(input.len());
                assert_eq!(q.num_output(input.len()), required);
                let mut expected = vec![0.0; required];
                let mut actual = expected.clone();
                assert_eq!(reference.execute_block(&input, &mut expected).unwrap(), required);
                assert_eq!(q.execute_block(&input, &mut actual).unwrap(), required);
                assert_eq!(actual, expected);
            }
        }
    }

    fn testbench_resamp_crcf(r: f32, as_db: f32, _id: usize) {
        // options
        let bw: f32 = 0.25; // target output signal bandwidth
        let tol: f32 = 0.5; // output PSD error tolerance [dB]
        let m: usize = 20; // resampler semi-length
        let npfb: usize = 2048; // number of filters in bank
        let fc: f32 = 0.45; // resampler cut-off frequency

        // create resampler
        let mut resamp = ArbitraryResampler::<Complex32>::new(r, m, fc, as_db, npfb).unwrap();

        // generate pulse with sharp transition and very narrow side-lobes
        let p = (40.0 / r) as usize;
        let pulse_len = 4 * p + 1;
        let pulse = filter::fir::design::fir_design_kaiser(pulse_len, 0.5 * r * bw, 120.0, 0.0).unwrap();

        // allocate buffers and copy input
        let num_input = pulse_len + 2 * m + 1;
        let num_output = resamp.num_output(num_input);
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); num_input];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); num_output];

        for i in 0..num_input {
            buf_0[i] = if i < pulse_len { Complex32::new(pulse[i] * bw, 0.0) } else { Complex32::new(0.0, 0.0) };
        }

        // resample
        let nw = resamp.execute_block(&buf_0, &mut buf_1).unwrap();

        // verify result
        #[rustfmt::skip]
        let regions = vec![
            PsdRegion { fmin:      -0.5, fmax: -0.6 * bw, pmin:  0.0, pmax: -as_db + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.4 * bw, fmax:  0.4 * bw, pmin: -tol, pmax:          tol, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.6 * bw, fmax:       0.5, pmin:  0.0, pmax: -as_db + tol, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_signal(&buf_1[..nw], &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_00)]
    fn test_resamp_crcf_00() {
        testbench_resamp_crcf(0.127115323, 60.0, 0);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_01)]
    fn test_resamp_crcf_01() {
        testbench_resamp_crcf(0.373737373, 60.0, 1);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_02)]
    fn test_resamp_crcf_02() {
        testbench_resamp_crcf(0.676543210, 60.0, 2);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_03)]
    fn test_resamp_crcf_03() {
        testbench_resamp_crcf(0.973621947, 60.0, 3);
    }

    // #[test]
    // fn test_resamp_crcf_04() {
    //     testbench_resamp_crcf(1.023832447, 60.0, 4);
    // }
    // #[test]
    // fn test_resamp_crcf_05() {
    //     testbench_resamp_crcf(2.182634827, 60.0, 5);
    // }
    // #[test]
    // fn test_resamp_crcf_06() {
    //     testbench_resamp_crcf(8.123980823, 60.0, 6);
    // }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_10)]
    fn test_resamp_crcf_10() {
        testbench_resamp_crcf(0.127115323, 80.0, 10);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_11)]
    fn test_resamp_crcf_11() {
        testbench_resamp_crcf(0.373737373, 80.0, 11);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_12)]
    fn test_resamp_crcf_12() {
        testbench_resamp_crcf(0.676543210, 80.0, 12);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_13)]
    fn test_resamp_crcf_13() {
        testbench_resamp_crcf(0.973621947, 80.0, 13);
    }

    fn testbench_resamp_crcf_num_output(rate: f32, npfb: usize) {
        if cfg!(test) {
            println!("testing resamp_crcf_get_num_output() with r={}, npfb={}", rate, npfb);
        }

        // create object
        let fc = 0.4f32;
        let as_db = 60.0f32;
        let m = 20;
        let mut resamp = ArbitraryResampler::<Complex32>::new(rate, m, fc, as_db, npfb).unwrap();

        // sizes to test in sequence
        let sizes = [1, 2, 3, 20, 7, 64, 4, 4, 4, 27];

        // allocate buffers (over-provision output to help avoid segmentation faults on error)
        let max_input = 64;
        let max_output = 16 + (4.0 * max_input as f32 * rate) as usize;
        let buf_0 = vec![Complex32::new(0.0, 0.0); max_input];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); max_output];

        // run numerous blocks
        for _b in 0..8 {
            for (_i, &size) in sizes.iter().enumerate() {
                let num_input = size;
                let num_output = resamp.num_output(num_input);
                let num_written = resamp.execute_block(&buf_0[..num_input], &mut buf_1).unwrap();
                assert_eq!(num_output, num_written);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_0)]
    fn test_resamp_crcf_num_output_0() {
        testbench_resamp_crcf_num_output(1.00, 64);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_1)]
    fn test_resamp_crcf_num_output_1() {
        testbench_resamp_crcf_num_output(1.00, 256);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_2)]
    fn test_resamp_crcf_num_output_2() {
        testbench_resamp_crcf_num_output(0.50, 256);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_3)]
    fn test_resamp_crcf_num_output_3() {
        testbench_resamp_crcf_num_output(2f32.sqrt(), 256);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_4)]
    fn test_resamp_crcf_num_output_4() {
        testbench_resamp_crcf_num_output(17f32.sqrt(), 16);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_5)]
    fn test_resamp_crcf_num_output_5() {
        testbench_resamp_crcf_num_output(1.0 / std::f32::consts::PI, 64);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_6)]
    fn test_resamp_crcf_num_output_6() {
        testbench_resamp_crcf_num_output(5.0f32.exp(), 64);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_7)]
    fn test_resamp_crcf_num_output_7() {
        testbench_resamp_crcf_num_output((-5.0f32).exp(), 64);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_copy)]
    fn test_resamp_crcf_copy() {
        // create object with irregular parameters
        let rate = 0.71239213987520f32;
        let mut q0 = ArbitraryResampler::<Complex32>::new(rate, 17, 0.37, 60.0, 64).unwrap();

        // run samples through filter
        let num_samples = 80;
        let mut y0 = vec![Complex32::new(0.0, 0.0); 1];
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let _nw0 = q0.execute(v, &mut y0).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters and check equality
        let mut y1 = vec![Complex32::new(0.0, 0.0); 1];
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let nw0 = q0.execute(v, &mut y0).unwrap();
            let nw1 = q1.execute(v, &mut y1).unwrap();

            // check that either 0 or 1 samples were written
            assert!(nw0 < 2);
            assert!(nw1 < 2);

            // check that the same number of samples were written
            assert_eq!(nw0, nw1);

            // check output sample values
            if nw0 == 1 && nw1 == 1 {
                assert_eq!(y0, y1);
            }
        }
    }

    fn testbench_resamp_crcf_max_input(rate: f32, npfb: usize) {
        // create object
        let fc = 0.4f32;
        let as_db = 60.0f32;
        let m = 20;
        let mut resamp = ArbitraryResampler::<Complex32>::new(rate, m, fc, as_db, npfb).unwrap();

        // allocate buffers
        let max_input = 1024;
        let max_output = 16 + (4.0 * max_input as f32 * rate.max(1.0)) as usize;
        let buf_0 = vec![Complex32::new(0.0, 0.0); max_input];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); max_output];

        // run blocks and verify max_input is consistent with num_output
        for output_limit in [1, 2, 3, 5, 10, 20, 50, 100] {
            let num_input = resamp.max_input(output_limit);
            let num_output = resamp.num_output(num_input);

            // max_input returns the max inputs that produce at most output_limit outputs
            assert!(
                num_output <= output_limit,
                "rate={}, limit={}, num_input={}, num_output={}",
                rate,
                output_limit,
                num_input,
                num_output
            );

            // verify that one more input would exceed the limit
            let num_output_more = resamp.num_output(num_input + 1);
            assert!(
                num_output_more > output_limit,
                "rate={}, limit={}, num_input+1={}, num_output_more={}",
                rate,
                output_limit,
                num_input + 1,
                num_output_more
            );

            // actually run the resampler to verify
            let num_written = resamp.execute_block(&buf_0[..num_input], &mut buf_1).unwrap();
            assert_eq!(num_output, num_written);
        }
    }

    #[test]
    fn test_resamp_crcf_max_input_0() {
        testbench_resamp_crcf_max_input(1.00, 64);
    }

    #[test]
    fn test_resamp_crcf_max_input_1() {
        testbench_resamp_crcf_max_input(0.50, 256);
    }

    #[test]
    fn test_resamp_crcf_max_input_2() {
        testbench_resamp_crcf_max_input(2.0, 256);
    }

    #[test]
    fn test_resamp_crcf_max_input_3() {
        testbench_resamp_crcf_max_input(0.127115323, 64);
    }

    #[test]
    fn test_resamp_crcf_max_input_4() {
        testbench_resamp_crcf_max_input(std::f32::consts::PI, 64);
    }

    #[test]
    fn test_resamp_crcf_max_input_5() {
        testbench_resamp_crcf_max_input(1.0 / std::f32::consts::PI, 64);
    }

    #[test]
    fn test_resamp_crcf_block_matches() {
        for &rate in &[0.37f32, 0.73, 1.0, 1.37, 4.1] {
            let mut q_sample = ArbitraryResampler::<Complex32, f32>::new(rate, 7, 0.4, 60.0, 64).unwrap();
            let mut q_block = q_sample.clone();

            let chunks = [0usize, 1, 5, 13, 14, 15, 64, 3, 97];
            let total: usize = chunks.iter().sum();
            let input: Vec<_> = (0..total)
                .map(|i| Complex32::new(((i + 3) as f32 * 0.11).cos(), ((i + 5) as f32 * 0.07).sin()))
                .collect();

            let mut offset = 0;
            for &chunk_len in &chunks {
                let x = &input[offset..offset + chunk_len];
                offset += chunk_len;

                let num_output = q_sample.num_output(x.len());
                assert_eq!(q_block.num_output(x.len()), num_output);

                let mut expected = vec![Complex32::default(); num_output];
                let mut nw = 0;
                for &xi in x {
                    nw += q_sample.execute(xi, &mut expected[nw..]).unwrap();
                }
                assert_eq!(nw, num_output);

                let mut actual = vec![Complex32::default(); num_output];
                assert_eq!(q_block.execute_block(x, &mut actual).unwrap(), num_output);
                assert_eq!(actual, expected, "rate={rate}, chunk_len={chunk_len}");
            }
        }
    }

    #[test]
    fn test_resamp_rrrf_block_matches() {
        let rate = 1.37;
        let mut q_sample = ArbitraryResampler::<f32, f32>::new(rate, 7, 0.4, 60.0, 64).unwrap();
        let mut q_block = q_sample.clone();
        let input: Vec<_> = (0..257).map(|i| ((i + 3) as f32 * 0.11).cos()).collect();

        let num_output = q_sample.num_output(input.len());
        let mut expected = vec![0.0; num_output];
        let mut nw = 0;
        for &xi in &input {
            nw += q_sample.execute(xi, &mut expected[nw..]).unwrap();
        }

        let mut actual = vec![0.0; num_output];
        assert_eq!(q_block.execute_block(&input, &mut actual).unwrap(), nw);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_resamp_execute_capacity_without_advancing() {
        let mut q = ArbitraryResampler::<f32, f32>::from_rate(2.0).unwrap();
        let mut q_ref = q.clone();
        let required = q.num_output(1);
        assert!(q.execute(1.0, &mut vec![0.0; required - 1]).is_err());

        let mut actual = vec![-9.0; required + 2];
        let mut expected = vec![0.0; required];
        assert_eq!(q.execute(1.0, &mut actual).unwrap(), required);
        q_ref.execute(1.0, &mut expected).unwrap();
        assert_eq!(&actual[..required], &expected);
        assert_eq!(&actual[required..], &[-9.0, -9.0]);

        let mut q = ArbitraryResampler::<f32, f32>::from_rate(2.0).unwrap();
        let mut q_ref = q.clone();
        let input = [1.0, -0.5, 0.25];
        let required = q.num_output(input.len());
        assert!(q.execute_block(&input, &mut vec![0.0; required - 1]).is_err());
        let mut actual = vec![-9.0; required + 2];
        let mut expected = vec![0.0; required];
        assert_eq!(q.execute_block(&input, &mut actual).unwrap(), required);
        q_ref.execute_block(&input, &mut expected).unwrap();
        assert_eq!(&actual[..required], &expected);
        assert_eq!(&actual[required..], &[-9.0, -9.0]);
        assert_eq!(q.execute_block(&[], &mut actual).unwrap(), 0);
    }

    fn iterative_num_output(phase: u32, step: u32, num_input: usize) -> usize {
        const PHASE_SCALE: u128 = 1 << 24;

        let mut phase = phase as u128;
        let mut num_output = 0;
        for _ in 0..num_input {
            while phase < PHASE_SCALE {
                num_output += 1;
                phase += step as u128;
            }
            phase -= PHASE_SCALE;
        }
        num_output
    }

    fn iterative_max_input(phase: u32, step: u32, max_output: usize) -> usize {
        const PHASE_SCALE: u128 = 1 << 24;

        let mut phase = phase as u128;
        let mut output_count = 0;
        let mut num_input = 0;
        loop {
            while phase < PHASE_SCALE {
                output_count += 1;
                if output_count > max_output {
                    return num_input;
                }
                phase += step as u128;
            }
            phase -= PHASE_SCALE;
            num_input += 1;
        }
    }

    #[test]
    fn test_resamp_output_count_closed_forms() {
        const PHASE_SCALE: u32 = 1 << 24;

        let mut slow = ArbitraryResampler::<f32, f32>::from_rate(0.004).unwrap();
        slow.execute(0.0, &mut [0.0]).unwrap();
        let post_rate_change_phase = slow.phase;

        for rate in [0.004, 0.127115323, 0.5, 1.0, 2f32.sqrt(), 2.0, 250.0] {
            let mut q = ArbitraryResampler::<f32, f32>::from_rate(rate).unwrap();
            let phases = [0, PHASE_SCALE - 1, PHASE_SCALE, q.step - 1, post_rate_change_phase];

            for phase in phases {
                q.phase = phase;

                for num_input in [0, 1, 2, 3, 17, 257] {
                    assert_eq!(
                        q.num_output(num_input),
                        iterative_num_output(phase, q.step, num_input),
                        "rate={rate}, phase={phase}, num_input={num_input}",
                    );
                }

                for max_output in [0, 1, 2, 7, 64] {
                    assert_eq!(
                        q.max_input(max_output),
                        iterative_max_input(phase, q.step, max_output),
                        "rate={rate}, phase={phase}, max_output={max_output}",
                    );
                }
            }
        }
    }
}
