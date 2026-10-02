use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::{self, FirFilterShape, FirPolyphaseFilter, Sample};
use crate::math::gcd;

/// Resampler that changes the sample rate by an exact rational ratio.
///
/// The output rate is `p / q` times the input rate. The resampler works in
/// execution blocks. Each block consumes [`Self::q`] input samples and writes
/// [`Self::p`] output samples. A new resampler starts with zero-valued
/// history.
///
/// The designed-filter constructors reduce the ratio to lowest terms before
/// building the filter. The common factor they remove becomes
/// [`Self::block_len`], so one block still covers the ratio as it was given.
/// [`Self::interp`] and [`Self::decim`] report the reduced factors.
#[derive(Clone, Debug)]
#[doc(alias = "Rresamp")]
pub struct RationalResampler<T, Coeff = T> {
    p: usize,
    q: usize,
    coefficient_count: usize,
    block_len: usize,
    pfb: FirPolyphaseFilter<T, Coeff>,
}

impl<T, Coeff> RationalResampler<T, Coeff>
where
    Coeff: Sample,
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates a rational resampler from existing filter coefficients.
    ///
    /// The coefficients describe a prototype filter running at the
    /// interpolated rate, which is `interpolation_factor` times the input
    /// rate. The rate factors are used as given, even when they share a common
    /// divisor, so [`Self::block_len`] is one. The output scale starts at one.
    ///
    /// The midpoint of the coefficients sets [`Self::input_delay`] and
    /// [`Self::output_delay`]. That is the true delay only for a linear-phase
    /// filter.
    ///
    /// # Errors
    ///
    /// Returns an error when either rate factor is zero or `coefficients` is
    /// empty.
    pub fn new(interpolation_factor: usize, decimation_factor: usize, coefficients: &[Coeff]) -> Result<Self> {
        if interpolation_factor == 0 {
            return Err(Error::Config("interpolation rate must be greater than zero".into()));
        }
        if decimation_factor == 0 {
            return Err(Error::Config("decimation rate must be greater than zero".into()));
        }

        let pfb = FirPolyphaseFilter::new(interpolation_factor, coefficients)?;

        let mut q = Self {
            p: interpolation_factor,
            q: decimation_factor,
            coefficient_count: coefficients.len(),
            block_len: 1,
            pfb,
        };

        q.reset();
        Ok(q)
    }

    /// Creates a rational resampler with a Kaiser-windowed filter.
    ///
    /// # Arguments
    ///
    /// - `filter_semi_length`: filter span on each side of the center tap, in
    ///   input samples.
    /// - `bandwidth`: cutoff in cycles per input sample, in `(0, 0.5]`. A
    ///   finite negative value selects the Nyquist frequency of the slower
    ///   of the input and output rates.
    /// - `stopband_attenuation`: in dB. Must be positive.
    ///
    /// When decimating, a cutoff above the output Nyquist frequency lets
    /// aliasing through. The constructor accepts such a cutoff. A negative
    /// `bandwidth` puts the cutoff exactly at the output Nyquist frequency, so
    /// the transition band lies above it and still aliases. With a rate of 1/4,
    /// a semi-length of 12 and 60 dB of attenuation, a tone at 0.126 cycles per
    /// input sample aliases only about 6.2 dB below the DC level. Leave room for
    /// the transition band below the output Nyquist frequency to prevent this.
    ///
    /// The output scale is set so that a constant input settles to a gain of
    /// about `sqrt(decimation_factor / interpolation_factor)`. Use
    /// [`Self::set_scale`] to change the level.
    ///
    /// The rate factors are reduced to lowest terms before the filter is
    /// designed.
    ///
    /// # Errors
    ///
    /// Returns an error when either rate factor is zero, `bandwidth` is
    /// nonfinite, zero or above 0.5, or `stopband_attenuation` is zero or
    /// negative.
    ///
    /// # Panics
    ///
    /// Panics when the filter length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps and the design is
    /// invalid.
    pub fn new_kaiser(
        interpolation_factor: usize,
        decimation_factor: usize,
        filter_semi_length: usize,
        bandwidth: f32,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        let gcd = gcd(interpolation_factor, decimation_factor)?;
        let interpolation_factor = interpolation_factor / gcd;
        let decimation_factor = decimation_factor / gcd;

        if !bandwidth.is_finite() {
            return Err(Error::Config("bandwidth must be finite".into()));
        }
        let bandwidth = if bandwidth < 0.0 {
            if interpolation_factor > decimation_factor {
                0.5
            } else {
                0.5 * interpolation_factor as f32 / decimation_factor as f32
            }
        } else if bandwidth > 0.5 {
            return Err(Error::Config(format!("invalid bandwidth ({}), must be less than 0.5", bandwidth)));
        } else {
            bandwidth
        };

        let h_len = 2 * interpolation_factor * filter_semi_length + 1;
        let hf = filter::fir::design::fir_design_kaiser(
            h_len,
            bandwidth / interpolation_factor as f32,
            stopband_attenuation,
            0.0,
        )?;

        let h: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();

        let mut q = Self::new(interpolation_factor, decimation_factor, &h)?;
        q.set_scale((2.0 * bandwidth * ((q.q as f32) / (q.p as f32)).sqrt()).into());
        q.block_len = gcd;

        Ok(q)
    }

    /// Creates a rational resampler from a pulse-shaping prototype.
    ///
    /// A symbol here is one sample at the slower of the input and output
    /// rates. The filter spans `filter_semi_length` symbols on each side of its
    /// center tap, and its cutoff sits at the Nyquist frequency of the slower
    /// rate. `excess_bandwidth` is the rolloff factor of the selected
    /// prototype.
    ///
    /// For most prototypes, the output scale is set so that a constant input
    /// settles to a gain of about `sqrt(decimation_factor /
    /// interpolation_factor)`. The GMSK prototypes are an exception. Use
    /// [`Self::set_scale`] to change the level.
    ///
    /// The rate factors are reduced to lowest terms before the filter is
    /// designed.
    ///
    /// # Errors
    ///
    /// Returns an error when either rate factor is zero or the selected
    /// prototype rejects `filter_semi_length` or `excess_bandwidth`.
    ///
    /// # Panics
    ///
    /// Panics when the filter length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps and the design is
    /// invalid.
    pub fn new_prototype(
        filter_shape: FirFilterShape,
        interpolation_factor: usize,
        decimation_factor: usize,
        filter_semi_length: usize,
        excess_bandwidth: f32,
    ) -> Result<Self> {
        let gcd = gcd(interpolation_factor, decimation_factor)?;
        let interpolation_factor = interpolation_factor / gcd;
        let decimation_factor = decimation_factor / gcd;

        let is_decimating = interpolation_factor < decimation_factor;
        let prototype_rate = if is_decimating { decimation_factor } else { interpolation_factor };
        let hf = filter::fir::design::fir_design_prototype(
            filter_shape,
            prototype_rate,
            filter_semi_length,
            excess_bandwidth,
            0.0,
        )?;

        let h: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();

        let mut q = Self::new(interpolation_factor, decimation_factor, &h)?;
        q.block_len = gcd;

        let rate = q.rate();
        q.set_scale((if is_decimating { rate.sqrt() } else { 1.0 / rate.sqrt() }).into());

        Ok(q)
    }

    /// Creates a rational resampler with a default Kaiser-windowed filter.
    ///
    /// The filter spans 12 input samples on each side of its center tap, with
    /// a cutoff of 0.5 cycles per input sample and 60 dB of stopband
    /// attenuation. A constant input settles to a gain of about
    /// `sqrt(decimation_factor / interpolation_factor)`.
    ///
    /// The cutoff does not depend on the ratio. When decimating, it passes
    /// frequencies above the output Nyquist frequency, which then alias. Use
    /// [`Self::new_kaiser`] with a lower or negative `bandwidth` to prevent
    /// that.
    ///
    /// # Errors
    ///
    /// Returns an error when either rate factor is zero.
    ///
    /// # Panics
    ///
    /// Panics when the filter length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps and the design is
    /// invalid.
    pub fn new_kaiser_simple(interpolation_factor: usize, decimation_factor: usize) -> Result<Self> {
        Self::new_kaiser(interpolation_factor, decimation_factor, 12, 0.5, 60.0)
    }

    /// Clears the sample history to zeros.
    ///
    /// The scale is unchanged.
    pub fn reset(&mut self) {
        self.pfb.reset()
    }

    /// Sets the factor applied to every output sample.
    pub fn set_scale(&mut self, scale: Coeff) {
        self.pfb.set_scale(scale)
    }

    /// Returns the factor applied to every output sample.
    pub fn scale(&self) -> Coeff {
        self.pfb.scale()
    }

    /// Returns the filter delay in input samples.
    ///
    /// The delay is the midpoint of the prototype filter, measured at the input
    /// rate. It matches the group delay only for a linear-phase filter.
    pub fn input_delay(&self) -> f32 {
        (self.coefficient_count - 1) as f32 / (2.0 * self.p as f32)
    }

    /// Returns the filter delay in output samples.
    ///
    /// This is the same delay as [`Self::input_delay`], measured at the output
    /// rate.
    pub fn output_delay(&self) -> f32 {
        (self.coefficient_count - 1) as f32 / (2.0 * self.q as f32)
    }

    /// Returns the number of reduced-ratio steps in one execution block.
    ///
    /// The value is one for [`Self::new`]. For the designed-filter
    /// constructors, it is the common factor removed from the given ratio.
    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// Returns the output rate divided by the input rate.
    pub fn rate(&self) -> f32 {
        self.p as f32 / self.q as f32
    }

    /// Returns the number of output samples in one execution block.
    pub fn p(&self) -> usize {
        self.p * self.block_len
    }

    /// Returns the interpolation factor in lowest terms.
    pub fn interp(&self) -> usize {
        self.p
    }

    /// Returns the number of input samples in one execution block.
    pub fn q(&self) -> usize {
        self.q * self.block_len
    }

    /// Returns the decimation factor in lowest terms.
    pub fn decim(&self) -> usize {
        self.q
    }

    /// Returns the number of output samples that `num_input` input samples
    /// produce.
    ///
    /// Only complete execution blocks count. Input left over after the last
    /// complete block produces no output.
    ///
    /// # Panics
    ///
    /// Panics when the count overflows `usize` and overflow checks are enabled.
    /// Without overflow checks, the count wraps.
    pub fn num_output(&self, num_input: usize) -> usize {
        let input_block_size = self.q * self.block_len;
        let output_block_size = self.p * self.block_len;
        let num_blocks = num_input / input_block_size;
        num_blocks * output_block_size
    }

    /// Returns the largest number of input samples whose output fits in
    /// `max_output` samples.
    ///
    /// The count is always a whole number of execution blocks.
    ///
    /// # Panics
    ///
    /// Panics when the count overflows `usize` and overflow checks are enabled.
    /// Without overflow checks, the count wraps.
    pub fn max_input(&self, max_output: usize) -> usize {
        let input_block_size = self.q * self.block_len;
        let output_block_size = self.p * self.block_len;
        let max_blocks = max_output / output_block_size;
        max_blocks * input_block_size
    }

    /// Adds samples to the history without producing output.
    pub fn write(&mut self, buf: &[T]) {
        self.pfb.write(buf)
    }

    /// Consumes one execution block and writes its output.
    ///
    /// `input` must hold at least [`Self::q`] samples and `output` at least
    /// [`Self::p`]. Extra input is ignored and extra output is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the block size overflows `usize`, `input` is too
    /// short, or `output` is too short. The resampler and `output` are
    /// unchanged on error.
    pub fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        let q = self.q;
        let p = self.p;
        let required_input = self
            .block_len
            .checked_mul(q)
            .ok_or_else(|| Error::Range("resampler input length overflow".into()))?;
        let required_output = self
            .block_len
            .checked_mul(p)
            .ok_or_else(|| Error::Range("resampler output length overflow".into()))?;
        if input.len() < required_input {
            return Err(Error::Config("input length must be at least block_len * q".into()));
        }
        if output.len() < required_output {
            return Err(Error::Config("output length must be at least block_len * p".into()));
        }
        for i in 0..self.block_len {
            self.execute_primitive(&input[i * q..(i + 1) * q], &mut output[i * p..(i + 1) * p])?;
        }
        Ok(())
    }

    /// Consumes complete execution blocks and writes their output.
    ///
    /// `input.len()` must be a multiple of [`Self::q`], and `output` must hold
    /// at least `input.len() / q() * p()` samples. Extra output is left
    /// unchanged. Returns the number of samples written.
    ///
    /// # Errors
    ///
    /// Returns an error when a block size or output length overflows `usize`,
    /// `input` contains a partial block, or `output` is too short. The
    /// resampler and `output` are unchanged for these errors.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let input_block_size = self
            .block_len
            .checked_mul(self.q)
            .ok_or_else(|| Error::Range("resampler input block length overflow".into()))?;
        let output_block_size = self
            .block_len
            .checked_mul(self.p)
            .ok_or_else(|| Error::Range("resampler output block length overflow".into()))?;
        if !input.len().is_multiple_of(input_block_size) {
            return Err(Error::Config("input length must be a multiple of q()".into()));
        }
        let n = input.len() / input_block_size;
        let required_output = n
            .checked_mul(output_block_size)
            .ok_or_else(|| Error::Range("resampler output length overflow".into()))?;
        if output.len() < required_output {
            return Err(Error::Config("output length must be at least n * p()".into()));
        }
        for i in 0..n {
            self.execute(
                &input[i * input_block_size..(i + 1) * input_block_size],
                &mut output[i * output_block_size..(i + 1) * output_block_size],
            )?;
        }
        Ok(required_output)
    }

    fn execute_primitive(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        let mut index = 0;
        let mut n = 0;
        for &xi in &x[..self.q] {
            self.pfb.push(xi);

            while index < self.p {
                y[n] = self.pfb.execute(index)?;
                n += 1;
                index += self.q;
            }

            index -= self.p;
        }

        if index != 0 {
            return Err(Error::Internal(format!("index={} (expected 0)", index)));
        } else if n != self.p {
            return Err(Error::Internal(format!("n={} (expected P={})", n, self.p)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::SpectralPeriodogram;
    use crate::framing::ArbitraryRateSymbolStream;
    use crate::math::{hamming, WindowType};
    use crate::modem::ModulationScheme;
    use crate::utility::test_helpers::{validate_psd_spectrum, PsdRegion};
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    fn test_harness_rresamp_crcf_part(p: usize, q: usize, m: usize, n: usize) {
        // semi-fixed options
        let tol = 1e-12f32; // error tolerance (should be basically zero)
        let bw = 0.5f32; // resampling filter bandwidth
        let as_ = 60.0f32; // resampling filter stop-band attenuation [dB]

        // create two identical resampler objects
        let mut q0 = RationalResampler::<Complex32>::new_kaiser(p, q, m, bw, as_).unwrap();
        let mut q1 = RationalResampler::<Complex32>::new_kaiser(p, q, m, bw, as_).unwrap();

        // full input, output buffers
        let mut buf_in = vec![num_complex::Complex32::new(0.0, 0.0); 2 * q * n];
        let mut buf_out_0 = vec![num_complex::Complex32::new(0.0, 0.0); 2 * p * n];
        let mut buf_out_1 = vec![num_complex::Complex32::new(0.0, 0.0); 2 * p * n];

        // generate input signal (pulse, but can really be anything)
        for i in 0..(2 * q * n) {
            buf_in[i] = hamming(i, 2 * q * n).unwrap()
                * num_complex::Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * 0.037 * i as f32);
        }

        // run resampler normally in one large block (2*q*n inputs, 2*p*n outputs)
        q0.execute_block(&buf_in, &mut buf_out_0).unwrap();

        // reset and run with separate resamplers (e.g. in two threads)
        q0.reset();
        // first block runs as normal
        q0.execute_block(&buf_in[..q * n], &mut buf_out_1[..p * n]).unwrap();
        // initialize second block with q*m samples to account for delay
        for i in 0..m {
            q1.write(&buf_in[q * n - (m - i) * q..q * n - (m - i - 1) * q]);
        }
        // run remainder of second block as normal
        q1.execute_block(&buf_in[q * n..], &mut buf_out_1[p * n..]).unwrap();

        // compare output buffers between normal and partitioned operation
        for i in 0..(2 * p * n) {
            assert_abs_diff_eq!(buf_out_0[i].re, buf_out_1[i].re, epsilon = tol);
            assert_abs_diff_eq!(buf_out_0[i].im, buf_out_1[i].im, epsilon = tol);
        }
    }

    // actual tests
    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P1_Q5)]
    fn test_rresamp_crcf_part_p1_q5() {
        test_harness_rresamp_crcf_part(1, 5, 15, 20);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P2_Q5)]
    fn test_rresamp_crcf_part_p2_q5() {
        test_harness_rresamp_crcf_part(2, 5, 15, 20);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P3_Q5)]
    fn test_rresamp_crcf_part_p3_q5() {
        test_harness_rresamp_crcf_part(3, 5, 15, 20);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P6_Q5)]
    fn test_rresamp_crcf_part_p6_q5() {
        test_harness_rresamp_crcf_part(6, 5, 15, 20);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P8_Q5)]
    fn test_rresamp_crcf_part_p8_q5() {
        test_harness_rresamp_crcf_part(8, 5, 15, 20);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P9_Q5)]
    fn test_rresamp_crcf_part_p9_q5() {
        test_harness_rresamp_crcf_part(9, 5, 15, 20);
    }

    fn test_rresamp_crcf(method: &str, interp: usize, decim: usize, m: usize, bw: f32, as_: f32) {
        // options
        let n = 800000; // number of output samples to analyze
        let nfft = 800; // number of bins in transform
        let tol = 0.5f32; // error tolerance [dB]

        // create resampler with rate interp/decim
        let mut resamp = match method {
            "baseline" => RationalResampler::<Complex32, f32>::new_kaiser(interp, decim, m, bw, as_).unwrap(),
            "default" => RationalResampler::<Complex32, f32>::new_kaiser_simple(interp, decim).unwrap(),
            _ => {
                let ftype: FirFilterShape = method.parse().unwrap();
                let beta = bw; // rename to avoid confusion
                RationalResampler::<Complex32, f32>::new_prototype(ftype, interp, decim, m, beta).unwrap()
            }
        };

        let r = resamp.rate();

        // create and configure objects
        let bw = 0.2f32; // target output bandwidth
        let mut q = SpectralPeriodogram::<Complex32>::new(nfft, WindowType::Hann, nfft / 2, nfft / 4).unwrap();
        let mut gen =
            ArbitraryRateSymbolStream::new_linear(FirFilterShape::Kaiser, r * bw, 25, 0.2, ModulationScheme::Qpsk)
                .unwrap();
        gen.set_gain((bw * r).sqrt());

        // generate samples and push through spgram object
        let mut buf_0 = vec![num_complex::Complex32::new(0.0, 0.0); decim]; // input buffer
        let mut buf_1 = vec![num_complex::Complex32::new(0.0, 0.0); interp]; // output buffer
        while q.num_samples_total() < n {
            // generate block of samples
            gen.write_samples(&mut buf_0).unwrap();

            // resample
            resamp.execute(&buf_0, &mut buf_1).unwrap();

            // run samples through the spgram object
            q.write(&buf_1);
        }

        // verify result
        let psd = q.psd();
        #[rustfmt::skip]
        let regions = vec![
            PsdRegion { fmin: -0.5,    fmax: -0.6*bw, pmin:     0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.4*bw, fmax: 0.4*bw,  pmin: 0.0-tol, pmax:  0.0+tol, test_lo: true,  test_hi: true },
            PsdRegion { fmin: 0.6*bw,  fmax: 0.5,     pmin:     0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_spectrum(&psd, nfft, &regions).unwrap());
    }

    // baseline tests using create_kaiser() method
    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P1_Q5)]
    fn test_rresamp_crcf_baseline_p1_q5() {
        test_rresamp_crcf("baseline", 1, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P2_Q5)]
    fn test_rresamp_crcf_baseline_p2_q5() {
        test_rresamp_crcf("baseline", 2, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P3_Q5)]
    fn test_rresamp_crcf_baseline_p3_q5() {
        test_rresamp_crcf("baseline", 3, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P6_Q5)]
    fn test_rresamp_crcf_baseline_p6_q5() {
        test_rresamp_crcf("baseline", 6, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P8_Q5)]
    fn test_rresamp_crcf_baseline_p8_q5() {
        test_rresamp_crcf("baseline", 8, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P9_Q5)]
    fn test_rresamp_crcf_baseline_p9_q5() {
        test_rresamp_crcf("baseline", 9, 5, 15, -1.0, 60.0);
    }

    // tests using new_kaiser_simple() method
    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P1_Q5)]
    fn test_rresamp_crcf_default_p1_q5() {
        test_rresamp_crcf("default", 1, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P2_Q5)]
    fn test_rresamp_crcf_default_p2_q5() {
        test_rresamp_crcf("default", 2, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P3_Q5)]
    fn test_rresamp_crcf_default_p3_q5() {
        test_rresamp_crcf("default", 3, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P6_Q5)]
    fn test_rresamp_crcf_default_p6_q5() {
        test_rresamp_crcf("default", 6, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P8_Q5)]
    fn test_rresamp_crcf_default_p8_q5() {
        test_rresamp_crcf("default", 8, 5, 15, -1.0, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P9_Q5)]
    fn test_rresamp_crcf_default_p9_q5() {
        test_rresamp_crcf("default", 9, 5, 15, -1.0, 60.0);
    }

    // tests using create_prototype() method
    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_arkaiser_P3_Q5)]
    fn test_rresamp_crcf_arkaiser_p3_q5() {
        test_rresamp_crcf("arkaiser", 3, 5, 40, 0.2, 50.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_arkaiser_P5_Q3)]
    fn test_rresamp_crcf_arkaiser_p5_q3() {
        test_rresamp_crcf("arkaiser", 5, 3, 40, 0.2, 50.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_rrcos_P3_Q5)]
    fn test_rresamp_crcf_rrcos_p3_q5() {
        test_rresamp_crcf("rrcos", 3, 5, 40, 0.2, 50.0);
    }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_rrcos_P5_Q3)]
    fn test_rresamp_crcf_rrcos_p5_q3() {
        test_rresamp_crcf("rrcos", 5, 3, 40, 0.2, 50.0);
    }

    fn testbench_rresamp_crcf_num_output(interp: usize, decim: usize) {
        let mut resamp = RationalResampler::<Complex32, f32>::new_kaiser_simple(interp, decim).unwrap();
        let q = resamp.q();
        let p = resamp.p();

        // allocate buffers
        let max_blocks = 20;
        let buf_in = vec![Complex32::new(0.0, 0.0); max_blocks * q];
        let mut buf_out = vec![Complex32::new(0.0, 0.0); max_blocks * p];

        // test various block counts
        for num_blocks in 1..=max_blocks {
            let num_input = num_blocks * q;
            let expected_output = num_blocks * p;

            assert_eq!(resamp.num_output(num_input), expected_output);

            // actually run the resampler to verify
            resamp.execute_block(&buf_in[..num_input], &mut buf_out[..expected_output]).unwrap();
        }
    }

    #[test]
    fn test_rresamp_crcf_num_output_0() {
        testbench_rresamp_crcf_num_output(1, 5);
    }

    #[test]
    fn test_rresamp_crcf_num_output_1() {
        testbench_rresamp_crcf_num_output(3, 5);
    }

    #[test]
    fn test_rresamp_crcf_num_output_2() {
        testbench_rresamp_crcf_num_output(5, 3);
    }

    #[test]
    fn test_rresamp_crcf_num_output_3() {
        testbench_rresamp_crcf_num_output(8, 5);
    }

    fn testbench_rresamp_crcf_max_input(interp: usize, decim: usize) {
        let resamp = RationalResampler::<Complex32, f32>::new_kaiser_simple(interp, decim).unwrap();
        let q = resamp.q();

        // test various output limits
        for output_limit in [1, 2, 5, 10, 20, 50, 100] {
            let max_input = resamp.max_input(output_limit);
            let num_output = resamp.num_output(max_input);

            // max_input returns the max inputs that produce at most output_limit outputs
            assert!(
                num_output <= output_limit,
                "interp={}, decim={}, limit={}, max_input={}, num_output={}",
                interp,
                decim,
                output_limit,
                max_input,
                num_output
            );

            // verify that one more block would exceed the limit
            let next_input = max_input + q;
            let next_output = resamp.num_output(next_input);
            assert!(
                next_output > output_limit,
                "interp={}, decim={}, limit={}, next_input={}, next_output={}",
                interp,
                decim,
                output_limit,
                next_input,
                next_output
            );
        }
    }

    #[test]
    fn test_rresamp_crcf_max_input_0() {
        testbench_rresamp_crcf_max_input(1, 5);
    }

    #[test]
    fn test_rresamp_crcf_max_input_1() {
        testbench_rresamp_crcf_max_input(3, 5);
    }

    #[test]
    fn test_rresamp_crcf_max_input_2() {
        testbench_rresamp_crcf_max_input(5, 3);
    }

    #[test]
    fn test_rresamp_crcf_max_input_3() {
        testbench_rresamp_crcf_max_input(8, 5);
    }

    fn assert_impulse_delay(mut resamp: RationalResampler<f32, f32>, expected_input: f32, expected_output: f32) {
        assert_abs_diff_eq!(resamp.input_delay(), expected_input, epsilon = 1e-5);
        assert_abs_diff_eq!(resamp.output_delay(), expected_output, epsilon = 1e-5);
        assert_abs_diff_eq!(resamp.output_delay(), resamp.input_delay() * resamp.rate(), epsilon = 1e-5);

        let num_blocks = 200;
        let mut input = vec![0.0f32; num_blocks * resamp.q()];
        let mut output = vec![0.0f32; num_blocks * resamp.p()];
        input[0] = 1.0;
        resamp.execute_block(&input, &mut output).unwrap();

        let mut energy = 0.0f64;
        let mut weighted_index = 0.0f64;
        for (index, &sample) in output.iter().enumerate() {
            let sample_energy = sample as f64 * sample as f64;
            energy += sample_energy;
            weighted_index += index as f64 * sample_energy;
        }
        let measured_output = (weighted_index / energy) as f32;
        let measured_input = measured_output / resamp.rate();
        assert_abs_diff_eq!(measured_input, expected_input, epsilon = 1e-4);
        assert_abs_diff_eq!(measured_output, expected_output, epsilon = 1e-4);
    }

    #[test]
    fn test_rresamp_kaiser_delay_sample_domains() {
        assert_impulse_delay(RationalResampler::new_kaiser(3, 5, 40, -1.0, 60.0).unwrap(), 40.0, 24.0);
        assert_impulse_delay(RationalResampler::new_kaiser(5, 3, 40, -1.0, 60.0).unwrap(), 40.0, 200.0 / 3.0);
    }

    #[test]
    fn test_rresamp_prototype_delay_sample_domains() {
        // test rate 3/5
        assert_impulse_delay(
            RationalResampler::new_prototype(FirFilterShape::Arkaiser, 3, 5, 40, 0.2).unwrap(),
            (5.0 * 40.0) / 3.0,
            40.0,
        );
        // test rate 5/3
        assert_impulse_delay(
            RationalResampler::new_prototype(FirFilterShape::Arkaiser, 5, 3, 40, 0.2).unwrap(),
            40.0,
            (5.0 * 40.0) / 3.0,
        );
    }

    #[test]
    fn test_rresamp_crcf_num_output_non_coprime_rate() {
        testbench_rresamp_crcf_num_output(6, 4);
    }

    #[test]
    fn test_rresamp_block_non_coprime_boundaries() {
        let mut single = RationalResampler::<f32, f32>::new_kaiser_simple(6, 4).unwrap();
        let mut single_ref = single.clone();
        let input = [1.0, -0.5, 0.25, 0.75];
        assert!(single.execute(&input[..3], &mut [0.0; 6]).is_err());
        assert!(single.execute(&input, &mut [0.0; 5]).is_err());
        let mut actual = [-9.0; 8];
        let mut expected = [0.0; 6];
        single.execute(&input, &mut actual).unwrap();
        single_ref.execute(&input, &mut expected).unwrap();
        assert_eq!(&actual[..6], &expected);
        assert_eq!(&actual[6..], &[-9.0, -9.0]);

        let mut q = RationalResampler::<f32, f32>::new_kaiser_simple(6, 4).unwrap();
        let mut q_ref = q.clone();

        let mut rejected = [-9.0; 6];
        assert!(matches!(q.execute_block(&input[..3], &mut rejected), Err(Error::Config(_))));
        assert_eq!(rejected, [-9.0; 6]);
        assert!(matches!(q.execute_block(&input, &mut rejected[..5]), Err(Error::Config(_))));
        assert_eq!(rejected, [-9.0; 6]);

        let mut actual = [-9.0; 8];
        let mut expected = [0.0; 6];
        assert_eq!(q.execute_block(&input, &mut actual).unwrap(), 6);
        q_ref.execute_block(&input, &mut expected).unwrap();
        assert_eq!(&actual[..6], &expected);
        assert_eq!(&actual[6..], &[-9.0, -9.0]);
        assert_eq!(q.execute_block(&[], &mut actual).unwrap(), 0);
    }
}
