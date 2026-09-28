use crate::buffer::Window;
use crate::dotprod::{DotProd, DotProduct};
use crate::error::{Error, Result};
use crate::filter;
use crate::filter::Sample;

/// Half-band filter that resamples by two or splits a signal into two bands.
///
/// The same filter supports four operations. [`Self::decim_execute`] halves
/// the sample rate and [`Self::interp_execute`] doubles it.
/// [`Self::analyzer_execute`] splits pairs of samples into low and high
/// subbands, and [`Self::synthesizer_execute`] combines them again.
/// [`Self::filter_execute`] produces both bands at the original rate. Each
/// operation has a block form. A new filter starts with zero-valued history.
///
/// All operations share one filter history. Call [`Self::reset`] before
/// switching an instance to a different operation.
///
/// With a zero center frequency and the default scale of one, decimation
/// has a gain of about two for a constant input and interpolation has a gain
/// of about one. Use [`Self::set_scale`] to change the level.
#[derive(Clone, Debug)]
#[doc(alias = "Resamp2")]
pub struct HalfBandResampler<T, Coeff = T> {
    m: usize,

    dp: DotProduct<T, Coeff>,

    w0: Window<T>,
    w1: Window<T>,
    decim_phase: Vec<T>,
    scale: Coeff,

    toggle: bool,
}

impl<T, Coeff> HalfBandResampler<T, Coeff>
where
    Coeff: Sample,
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates a half-band filter.
    ///
    /// The prototype filter has `4 * filter_semi_length + 1` taps.
    /// `center_frequency` shifts the passband to that frequency and must lie in
    /// `[-0.5, 0.5]`. It is in cycles per sample at the filter's full rate,
    /// which is the input rate except when interpolating, where it is the
    /// output rate. Stopband attenuation is given in dB.
    ///
    /// # Errors
    ///
    /// Returns an error when `filter_semi_length` is less than two,
    /// `center_frequency` is outside its range, `stopband_attenuation` is
    /// negative, or the filter design rejects the attenuation.
    ///
    /// # Panics
    ///
    /// Panics when the filter length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps and the design is
    /// invalid.
    pub fn new(filter_semi_length: usize, center_frequency: f32, stopband_attenuation: f32) -> Result<Self> {
        if filter_semi_length < 2 {
            return Err(Error::Config("filter semi-length must be at least 2".into()));
        }
        if center_frequency < -0.5 || center_frequency > 0.5 {
            return Err(Error::Config(format!("center frequency ({}) must be in [-0.5,0.5]", center_frequency)));
        }
        if stopband_attenuation < 0.0 {
            return Err(Error::Config(format!(
                "stopband attenuation ({}) must be greater than zero",
                stopband_attenuation
            )));
        }

        let h_len = 4 * filter_semi_length + 1;
        let mut h = vec![Coeff::zero(); h_len];
        let hf =
            filter::fir::design::fir_design_pm_halfband_stopband_attenuation(filter_semi_length, stopband_attenuation)?;

        for (i, hi) in h.iter_mut().enumerate() {
            let t = i as f32 - (h_len - 1) as f32 / 2.0;
            *hi = Coeff::halfband_coefficient(hf[i], t, center_frequency);
        }

        let h1_len = 2 * filter_semi_length;
        let mut h1 = vec![Coeff::zero(); h1_len];
        for (i, h1i) in h1.iter_mut().enumerate() {
            *h1i = h[h_len - 2 * i - 2];
        }

        let w0 = Window::new(2 * filter_semi_length)?;
        let w1 = Window::new(2 * filter_semi_length)?;

        let mut q = Self {
            m: filter_semi_length,
            dp: DotProduct::new(&h1)?,
            w0,
            w1,
            decim_phase: Vec::new(),
            scale: Coeff::one(),
            toggle: false,
        };

        q.reset();
        Ok(q)
    }

    /// Clears the sample history to zeros and restarts the phase that
    /// [`Self::filter_execute`] tracks.
    ///
    /// The scale is unchanged.
    pub fn reset(&mut self) {
        self.w0.reset();
        self.w1.reset();
        self.toggle = false;
    }

    /// Sets the factor applied to every output sample.
    pub fn set_scale(&mut self, scale: Coeff) {
        self.scale = scale;
    }

    /// Returns the factor applied to every output sample.
    pub fn scale(&self) -> Coeff {
        self.scale
    }

    /// Returns the nominal filter delay, `2 * filter_semi_length - 1`.
    ///
    /// The value is in samples at the filter's full rate. The center tap of
    /// the prototype sits one sample later, at `2 * filter_semi_length`.
    pub fn delay(&self) -> usize {
        2 * self.m - 1
    }

    /// Grows the workspace used by [`Self::decim_execute_block`].
    ///
    /// Later calls with up to `n` input samples then run without allocating.
    pub fn reserve_decim_block(&mut self, n: usize) {
        let phase_len = n / 2;
        if self.decim_phase.len() < phase_len {
            self.decim_phase.resize(phase_len, T::default());
        }
    }

    /// Filters one sample into low-pass and high-pass outputs.
    ///
    /// Both outputs stay at the input rate. The returned pair is
    /// `(low_pass, high_pass)`.
    pub fn filter_execute(&mut self, input: T) -> Result<(T, T)> {
        let (yi, yq) = if !self.toggle {
            self.w0.push(input);
            let yi = self.w0.index(self.m - 1)?;
            let r = self.w1.read();
            let yq = self.dp.execute(r);
            (yi, yq)
        } else {
            self.w1.push(input);
            let yi = self.w1.index(self.m - 1)?;
            let r = self.w0.read();
            let yq = self.dp.execute(r);
            (yi, yq)
        };

        self.toggle = !self.toggle;

        let y0 = Into::<T>::into(0.5) * (yi + yq) * self.scale;
        let y1 = Into::<T>::into(0.5) * (yi - yq) * self.scale;
        Ok((y0, y1))
    }

    /// Splits two input samples into one low-band and one high-band sample.
    ///
    /// The low-band sample goes to `output[0]` and the high-band sample to
    /// `output[1]`. Extra input is ignored and extra output is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics when `input` or `output` holds fewer than two samples. When
    /// `output` is the short one, the history has already changed.
    pub fn analyzer_execute(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        self.w1.push(Into::<T>::into(0.5) * input[0]);
        let r = self.w1.read();
        let y1 = self.dp.execute(r);

        self.w0.push(Into::<T>::into(0.5) * input[1]);
        let y0 = self.w0.index(self.m - 1)?;

        output[0] = (y1 + y0) * self.scale;
        output[1] = (y1 - y0) * self.scale;
        Ok(())
    }

    /// Combines one low-band and one high-band sample into two output samples.
    ///
    /// This reverses [`Self::analyzer_execute`], taking the low-band sample
    /// from `input[0]` and the high-band sample from `input[1]`. Extra input
    /// is ignored and extra output is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics when `input` or `output` holds fewer than two samples. When
    /// `output` is the short one, the history has already changed.
    pub fn synthesizer_execute(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        let x0 = input[0] + input[1];
        let x1 = input[0] - input[1];

        self.w0.push(x0);
        output[0] = self.w0.index(self.m - 1)? * self.scale;

        self.w1.push(x1);
        let r = self.w1.read();
        output[1] = self.dp.execute(r) * self.scale;

        Ok(())
    }

    /// Consumes two input samples and returns one output sample.
    ///
    /// Extra input is ignored.
    ///
    /// # Panics
    ///
    /// Panics when `input` holds fewer than two samples. The history has
    /// already changed when `input` holds exactly one.
    pub fn decim_execute(&mut self, input: &[T]) -> Result<T> {
        self.w1.push(input[0]);
        let r = self.w1.read();
        let y1 = self.dp.execute(r);

        self.w0.push(input[1]);
        let y0 = self.w0.index(self.m - 1)?;

        let y = (y0 + y1) * self.scale;
        Ok(y)
    }

    /// Consumes one input sample and writes two output samples.
    ///
    /// Extra output is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics when `output` holds fewer than two samples. The history and
    /// `output[0]` may already have changed.
    pub fn interp_execute(&mut self, input: T, output: &mut [T]) -> Result<()> {
        self.w0.push(input);
        output[0] = self.w0.index(self.m - 1)? * self.scale;

        self.w1.push(input);
        let r = self.w1.read();
        output[1] = self.dp.execute(r) * self.scale;

        Ok(())
    }

    /// Filters a block of samples into interleaved low-pass and high-pass
    /// outputs.
    ///
    /// `output` must hold at least `2 * input.len()` samples. Each input
    /// sample produces a low-pass sample followed by a high-pass sample. Extra
    /// output is left unchanged. Returns the number of samples written,
    /// `2 * input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is too short. The filter and `output`
    /// are unchanged on error.
    pub fn filter_execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let n_out = input.len().checked_mul(2).ok_or_else(|| Error::Range("filter output length overflow".into()))?;
        if output.len() < n_out {
            return Err(Error::Config(format!("output length ({}) must be at least {}", output.len(), n_out,)));
        }
        for (i, &xi) in input.iter().enumerate() {
            let (y0, y1) = self.filter_execute(xi)?;
            output[2 * i] = y0;
            output[2 * i + 1] = y1;
        }
        Ok(n_out)
    }

    /// Splits a block of input pairs into interleaved low-band and high-band
    /// samples.
    ///
    /// Each pair of input samples produces one low-band sample followed by one
    /// high-band sample, as in [`Self::analyzer_execute`]. A trailing odd
    /// input sample is ignored. `output` must hold one sample for each input
    /// sample in a complete pair. Extra output is left unchanged. Returns the
    /// number of samples written.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is too short. The filter and `output` are
    /// unchanged on error.
    pub fn analyzer_execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let n = input.len() / 2;
        let n_out = 2 * n;
        if output.len() < n_out {
            return Err(Error::Config(format!("output length ({}) must be at least {}", output.len(), n_out,)));
        }
        for i in 0..n {
            self.analyzer_execute(&input[2 * i..2 * i + 2], &mut output[2 * i..2 * i + 2])?;
        }
        Ok(n_out)
    }

    /// Combines a block of interleaved low-band and high-band samples.
    ///
    /// This reverses [`Self::analyzer_execute_block`]. Each input pair
    /// produces two output samples, as in [`Self::synthesizer_execute`]. A
    /// trailing odd input sample is ignored. `output` must hold one sample for
    /// each input sample in a complete pair. Extra output is left unchanged.
    /// Returns the number of samples written.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is too short. The filter and `output` are
    /// unchanged on error.
    pub fn synthesizer_execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let n = input.len() / 2;
        let n_out = 2 * n;
        if output.len() < n_out {
            return Err(Error::Config(format!("output length ({}) must be at least {}", output.len(), n_out,)));
        }
        for i in 0..n {
            self.synthesizer_execute(&input[2 * i..2 * i + 2], &mut output[2 * i..2 * i + 2])?;
        }
        Ok(n_out)
    }

    /// Consumes a block of input samples and writes one output per pair.
    ///
    /// `input` must have an even length, and `output` must hold at least
    /// `input.len() / 2` samples. Extra output is left unchanged. Returns the
    /// number of samples written, `input.len() / 2`.
    ///
    /// The results can differ slightly from repeated
    /// [`Self::decim_execute`] calls, because the block dot product may sum in
    /// a different order. The call grows an internal workspace when needed.
    /// Use [`Self::reserve_decim_block`] to allocate it in advance.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` has an odd length or `output` is too
    /// short. The filter and `output` are unchanged on error.
    pub fn decim_execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if !input.len().is_multiple_of(2) {
            return Err(Error::Config("decimation input length must be a multiple of 2".into()));
        }
        let n = input.len() / 2;
        if output.len() < n {
            return Err(Error::Config(format!("output length ({}) must be at least {}", output.len(), n,)));
        }

        self.reserve_decim_block(input.len());

        let m = self.m;
        let scale = self.scale;
        let phase = &mut self.decim_phase[..n];

        // there are two branches to take care of. the even branch
        // computes a dotprod while the odd branch is just delayed
        // by `m` samples. both branches are `n` samples long.

        // pack the even samples into a contiguous phase
        for (i, pair) in input[..2 * n].as_chunks::<2>().0.iter().enumerate() {
            phase[i] = pair[0];
        }

        // compute the dotprod of the even samples as a single block
        // we're storing these in output, but output will still need the contribution
        // from the odd samples
        let dp = &self.dp;
        self.w1.execute_block_contiguous(phase, |indices, history| {
            dp.execute_block(history, &mut output[indices]);
        });

        // for the odd samples, we'll fetch the sample and then add and apply scale

        // the first `m` samples of the odd branch are fetched from w0, not input
        let prefix_len = n.min(m);
        for (yi, &odd) in output[..prefix_len].iter_mut().zip(&self.w0.read()[m..m + prefix_len]) {
            *yi = (*yi + odd) * scale;
        }

        let direct_end = n.saturating_sub(m);
        let retain_start = n.saturating_sub(self.w0.len());
        for (i, pair) in input[..2 * n].as_chunks::<2>().0.iter().enumerate() {
            if i < direct_end {
                output[i + m] = (output[i + m] + pair[1]) * scale;
            }
            if i >= retain_start {
                self.w0.push(pair[1]);
            }
        }

        Ok(n)
    }

    /// Consumes a block of input samples and writes two outputs for each.
    ///
    /// `output` must hold at least `2 * input.len()` samples. Extra output is
    /// left unchanged. Returns the number of samples written,
    /// `2 * input.len()`.
    ///
    /// The results can differ slightly from repeated
    /// [`Self::interp_execute`] calls, because the block dot product may sum in
    /// a different order.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is too short. The filter and `output`
    /// are unchanged on error.
    pub fn interp_execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let n_out = input
            .len()
            .checked_mul(2)
            .ok_or_else(|| Error::Range("interpolation output length overflow".into()))?;
        if output.len() < n_out {
            return Err(Error::Config(format!("output length ({}) must be at least {}", output.len(), n_out,)));
        }

        let m = self.m;
        let scale = self.scale;

        // compute the odd branch contiguously into the first half of output
        let dp = &self.dp;
        self.w1.execute_block_contiguous(input, |indices, history| {
            dp.execute_block(history, &mut output[indices]);
        });

        // expand the temporary results in place in reverse
        // this prevents overwriting of the dotprod in the first half of output
        let history = self.w0.read();
        for i in (0..input.len()).rev() {
            // the even output branch is an m-sample delay
            let even = if i < m { history[m + i] } else { input[i - m] };
            // the odd output branch is the dotprod we computed above
            let odd = output[i];
            output[2 * i] = even * scale;
            output[2 * i + 1] = odd * scale;
        }

        // retain tailing w0.len() samples
        let retain_from = input.len().saturating_sub(self.w0.len());
        self.w0.write(&input[retain_from..]);

        Ok(n_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
    use crate::utility::test_helpers::{validate_psd_signal, PsdRegion};
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    use num_complex::Complex32;

    #[test]
    #[autotest_annotate(autotest_resamp2_analysis)]
    fn test_resamp2_analysis() {
        let m = 5; // filter semi-length (actual length: 4*m+1)
        let n = 37; // number of input samples
        let as_ = 60.0; // stop-band attenuation [dB]
        let f0 = 0.0739; // low frequency signal
        let f1 = -0.1387; // high frequency signal (+pi)
        let tol = 1e-3; // error tolerance

        // allocate memory for data arrays
        let mut x = vec![Complex32::new(0.0, 0.0); 2 * n + 2 * m + 1]; // input signal (with delay)
        let mut y0 = vec![Complex32::new(0.0, 0.0); n]; // low-pass output
        let mut y1 = vec![Complex32::new(0.0, 0.0); n]; // high-pass output

        // generate the baseband signal
        for i in 0..(2 * n + 2 * m + 1) {
            x[i] = if i < 2 * n {
                Complex32::new(0.0, f0 * i as f32).exp()
                    + Complex32::new(0.0, (std::f32::consts::PI + f1) * i as f32).exp()
            } else {
                Complex32::new(0.0, 0.0)
            };
        }

        // create the half-band resampler, with a specified stopband attenuation level
        let mut q = HalfBandResampler::<Complex32>::new(m as usize, 0.0, as_).unwrap();

        // run half-band decimation
        let mut y_hat = [Complex32::new(0.0, 0.0); 2];
        for i in 0..n {
            q.analyzer_execute(&x[2 * i..2 * i + 2], &mut y_hat).unwrap();
            y0[i] = y_hat[0];
            y1[i] = y_hat[1];
        }

        // validate output
        for i in m..(n - m) {
            assert_abs_diff_eq!(y0[i + m].re, (2.0 * f0 * (i as f32 + 0.5)).cos(), epsilon = tol);
            assert_abs_diff_eq!(y0[i + m].im, (2.0 * f0 * (i as f32 + 0.5)).sin(), epsilon = tol);

            assert_abs_diff_eq!(y1[i + m].re, (2.0 * f1 * (i as f32 + 0.5)).cos(), epsilon = tol);
            assert_abs_diff_eq!(y1[i + m].im, (2.0 * f1 * (i as f32 + 0.5)).sin(), epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_resamp2_synthesis)]
    fn test_resamp2_synthesis() {
        let m = 5; // filter semi-length (actual length: 4*m+1)
        let n = 37; // number of input samples
        let as_ = 60.0; // stop-band attenuation [dB]
        let f0 = 0.0739; // low frequency signal
        let f1 = -0.1387; // high frequency signal (+pi)
        let tol = 3e-3; // error tolerance

        // allocate memory for data arrays
        let mut x0 = vec![Complex32::new(0.0, 0.0); n + 2 * m + 1]; // input signal (with delay)
        let mut x1 = vec![Complex32::new(0.0, 0.0); n + 2 * m + 1]; // input signal (with delay)
        let mut y = vec![Complex32::new(0.0, 0.0); 2 * n]; // synthesized output

        // generate the baseband signals
        for i in 0..(n + 2 * m + 1) {
            x0[i] = if i < 2 * n { Complex32::new(0.0, f0 * i as f32).exp() } else { Complex32::new(0.0, 0.0) };
            x1[i] = if i < 2 * n { Complex32::new(0.0, f1 * i as f32).exp() } else { Complex32::new(0.0, 0.0) };
        }

        // create the half-band resampler, with a specified stopband attenuation level
        let mut q = HalfBandResampler::<Complex32>::new(m as usize, 0.0, as_).unwrap();

        // run synthesis
        let mut x_hat = [Complex32::new(0.0, 0.0); 2];
        for i in 0..n {
            x_hat[0] = x0[i];
            x_hat[1] = x1[i];
            q.synthesizer_execute(&x_hat, &mut y[2 * i..2 * i + 2]).unwrap();
        }

        // validate output
        for i in m..(n - 2 * m) {
            assert_abs_diff_eq!(
                y[i + 2 * m].re,
                (0.5 * f0 * i as f32).cos() + ((std::f32::consts::PI + 0.5 * f1) * i as f32).cos(),
                epsilon = tol
            );
            assert_abs_diff_eq!(
                y[i + 2 * m].im,
                (0.5 * f0 * i as f32).sin() + ((std::f32::consts::PI + 0.5 * f1) * i as f32).sin(),
                epsilon = tol
            );
        }
    }

    fn testbench_resamp2_crcf_filter(m: usize, as_: f32) {
        // error tolerance [dB]
        let tol = 0.5f32;

        // create the half-band resampler
        let mut q = HalfBandResampler::<Complex32>::new(m, 0.0, as_).unwrap();

        // get impulse response
        let h_len = 4 * m + 1;
        let mut h_0 = vec![Complex32::new(0.0, 0.0); h_len]; // low-frequency response
        let mut h_1 = vec![Complex32::new(0.0, 0.0); h_len]; // high-frequency response

        for i in 0..h_len {
            let input = if i == 0 { Complex32::new(1.0, 0.0) } else { Complex32::new(0.0, 0.0) };
            let (y0, y1) = q.filter_execute(input).unwrap();
            h_0[i] = y0;
            h_1[i] = y1;
        }

        // compute expected transition band (extend slightly for relaxed constraints)
        let ft = filter::fir::design::estimate_req_filter_transition_bandwidth(as_, h_len).unwrap() * 1.1;

        // verify low-pass frequency response
        #[rustfmt::skip]
        let regions_h0 = vec![
            PsdRegion { fmin: -0.5,           fmax: -0.25 - ft/2.0, pmin: 0.0,  pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.25 + ft/2.0, fmax:  0.25 - ft/2.0, pmin: -1.0, pmax: 1.0,        test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.25 + ft/2.0, fmax:  0.5,           pmin: 0.0,  pmax: -as_ + tol, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_signal(&h_0, &regions_h0).unwrap());

        // verify high-pass frequency response
        #[rustfmt::skip]
        let regions_h1 = vec![
            PsdRegion { fmin: -0.5,           fmax: -0.25 - ft/2.0, pmin: -1.0, pmax: 1.0,        test_lo: true,  test_hi: true },
            PsdRegion { fmin: -0.25 + ft/2.0, fmax:  0.25 - ft/2.0, pmin: 0.0,  pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin:  0.25 + ft/2.0, fmax:  0.5,           pmin: -1.0, pmax: 1.0,        test_lo: true,  test_hi: true },
        ];

        assert!(validate_psd_signal(&h_1, &regions_h1).unwrap());
    }

    // test different configurations
    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_0)]
    fn test_resamp2_crcf_filter_0() {
        testbench_resamp2_crcf_filter(4, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_1)]
    fn test_resamp2_crcf_filter_1() {
        testbench_resamp2_crcf_filter(7, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_2)]
    fn test_resamp2_crcf_filter_2() {
        testbench_resamp2_crcf_filter(12, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_3)]
    fn test_resamp2_crcf_filter_3() {
        testbench_resamp2_crcf_filter(15, 80.0);
    }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_4)]
    fn test_resamp2_crcf_filter_4() {
        testbench_resamp2_crcf_filter(15, 100.0);
    }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_5)]
    fn test_resamp2_crcf_filter_5() {
        testbench_resamp2_crcf_filter(15, 120.0);
    }

    #[test]
    #[autotest_annotate(autotest_resamp2_config)]
    fn test_resamp2_config() {
        // check that object returns None for invalid configurations
        assert!(HalfBandResampler::<Complex32, f32>::new(0, 0.0, 60.0).is_err()); // m out of range
        assert!(HalfBandResampler::<Complex32, f32>::new(1, 0.0, 60.0).is_err()); // m out of range
        assert!(HalfBandResampler::<Complex32, f32>::new(2, 0.7, 60.0).is_err()); // f0 out of range
        assert!(HalfBandResampler::<Complex32, f32>::new(2, -0.7, 60.0).is_err()); // f0 out of range
        assert!(HalfBandResampler::<Complex32, f32>::new(2, 0.0, -1.0).is_err()); // as out of range

        // create proper object and test configurations
        let q = HalfBandResampler::<Complex32, f32>::new(4, 0.0, 60.0).unwrap();
        assert_eq!(q.delay(), 2 * 4 - 1);
        // q.print();

        // redesign filter with new length
        // nb there's no recreate
        // q = q.recreate(8, 0.0, 60.0);
        let q = HalfBandResampler::<Complex32, f32>::new(8, 0.0, 60.0).unwrap();
        assert_eq!(q.delay(), 2 * 8 - 1);

        // redesign filter with same length, but new stop-band suppression
        // q = q.recreate(8, 0.0, 80.0);
        let mut q = HalfBandResampler::<Complex32, f32>::new(8, 0.0, 80.0).unwrap();
        assert_eq!(q.delay(), 2 * 8 - 1);

        // test setting/getting properties
        q.set_scale(7.22);
        let scale = q.scale();
        assert_eq!(scale, 7.22);
    }

    // test copy method
    #[test]
    #[autotest_annotate(autotest_resamp2_copy)]
    fn test_resamp2_copy() {
        // create original half-band resampler
        let mut qa = HalfBandResampler::<Complex32>::new(12, 0.0, 60.0).unwrap();

        // run random samples through filter
        let num_samples = 80;
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let _ = qa.filter_execute(v);
        }

        // copy object
        let mut qb = qa.clone();

        // run random samples through both filters and compare
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let (ya0, ya1) = qa.filter_execute(v).unwrap();
            let (yb0, yb1) = qb.filter_execute(v).unwrap();

            assert_eq!(ya0, yb0);
            assert_eq!(ya1, yb1);
        }
    }

    #[test]
    fn test_resamp2_block_rejects_short_output_without_advancing() {
        let x = vec![Complex32::new(1.0, -0.5); 4];

        let mut q = HalfBandResampler::<Complex32, f32>::new(4, 0.0, 60.0).unwrap();
        assert!(q.filter_execute_block(&x[..2], &mut [Complex32::default(); 3]).is_err());

        let mut q = HalfBandResampler::<Complex32, f32>::new(4, 0.0, 60.0).unwrap();
        assert!(q.analyzer_execute_block(&x, &mut [Complex32::default(); 3]).is_err());

        let mut q = HalfBandResampler::<Complex32, f32>::new(4, 0.0, 60.0).unwrap();
        assert!(q.synthesizer_execute_block(&x, &mut [Complex32::default(); 3]).is_err());

        let mut q = HalfBandResampler::<Complex32, f32>::new(4, 0.0, 60.0).unwrap();
        assert!(q.decim_execute_block(&x, &mut [Complex32::default(); 1]).is_err());

        let mut q = HalfBandResampler::<Complex32, f32>::new(4, 0.0, 60.0).unwrap();
        let mut q_ref = q.clone();
        assert!(q.interp_execute_block(&x[..2], &mut [Complex32::default(); 3]).is_err());

        let mut y = [Complex32::default(); 8];
        let mut y_ref = [Complex32::default(); 8];
        q.interp_execute_block(&x, &mut y).unwrap();
        q_ref.interp_execute_block(&x, &mut y_ref).unwrap();
        assert_eq!(y, y_ref);
    }

    #[test]
    fn test_resamp2_decim_block_matches_crc() {
        for &m in &[2usize, 3, 5, 8] {
            let mut q_sample = HalfBandResampler::<Complex32, f32>::new(m, 0.0, 60.0).unwrap();
            q_sample.set_scale(0.73);
            let mut q_block = q_sample.clone();
            assert!(q_block.decim_phase.is_empty());

            let mut offset = 0usize;
            let mut max_n = 0usize;
            for n in [0, 1, m - 1, m, 2 * m - 1, 2 * m, 2 * m + 1, 37, 3] {
                let x: Vec<_> = (0..2 * n)
                    .map(|i| {
                        let t = (offset + i) as f32;
                        Complex32::new((0.17 * t).sin(), (0.31 * t).cos())
                    })
                    .collect();
                offset += x.len();
                max_n = max_n.max(n);

                let mut y_sample = vec![Complex32::default(); n];
                for i in 0..n {
                    y_sample[i] = q_sample.decim_execute(&x[2 * i..2 * i + 2]).unwrap();
                }

                let mut y_block = vec![Complex32::default(); n];
                assert_eq!(q_block.decim_execute_block(&x, &mut y_block).unwrap(), n);

                for (&actual, &expected) in y_block.iter().zip(&y_sample) {
                    assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-5);
                    assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-5);
                }
                assert_eq!(q_block.w0.read(), q_sample.w0.read(), "m={m}, n={n}, w0");
                assert_eq!(q_block.w1.read(), q_sample.w1.read(), "m={m}, n={n}, w1");
                assert_eq!(q_block.decim_phase.len(), max_n);
            }
        }
    }

    #[test]
    fn test_resamp2_decim_block_rejects_partial_pair() {
        let mut q = HalfBandResampler::<Complex32, f32>::new(4, 0.0, 60.0).unwrap();
        let mut q_ref = q.clone();
        assert!(q.decim_execute_block(&[Complex32::default(); 3], &mut [Complex32::default(); 2]).is_err());

        let input = [Complex32::new(1.0, -0.5), Complex32::new(0.25, 0.75)];
        let mut actual = [Complex32::default(); 1];
        let mut expected = [Complex32::default(); 1];
        q.decim_execute_block(&input, &mut actual).unwrap();
        q_ref.decim_execute_block(&input, &mut expected).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_resamp2_decim_block_matches_rrr() {
        for &m in &[2usize, 3, 5, 8, 16] {
            let mut q_sample = HalfBandResampler::<f32, f32>::new(m, 0.0, 60.0).unwrap();
            q_sample.set_scale(0.73);
            let mut q_block = q_sample.clone();

            let mut offset = 0usize;
            for n in [0, 1, m - 1, m, 2 * m - 1, 2 * m, 2 * m + 1, 73, 3] {
                let x: Vec<_> = (0..2 * n).map(|i| ((offset + i) as f32 * 0.17).sin()).collect();
                offset += x.len();

                let mut y_sample = vec![0.0; n];
                for i in 0..n {
                    y_sample[i] = q_sample.decim_execute(&x[2 * i..2 * i + 2]).unwrap();
                }

                let mut y_block = vec![0.0; n];
                assert_eq!(q_block.decim_execute_block(&x, &mut y_block).unwrap(), n);

                for (&actual, &expected) in y_block.iter().zip(&y_sample) {
                    assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
                }
                assert_eq!(q_block.w0.read(), q_sample.w0.read(), "m={m}, n={n}, w0");
                assert_eq!(q_block.w1.read(), q_sample.w1.read(), "m={m}, n={n}, w1");
            }
        }
    }

    #[test]
    fn test_resamp2_interp_block_matches_crc() {
        for &m in &[2usize, 3, 5, 8] {
            let mut q_sample = HalfBandResampler::<Complex32, f32>::new(m, 0.0, 60.0).unwrap();
            q_sample.set_scale(0.73);
            let mut q_block = q_sample.clone();

            let mut offset = 0usize;
            for n in [0, 1, m - 1, m, 2 * m - 1, 2 * m, 2 * m + 1, 37, 3] {
                let x: Vec<_> = (0..n)
                    .map(|i| {
                        let t = (offset + i) as f32;
                        Complex32::new((0.17 * t).sin(), (0.31 * t).cos())
                    })
                    .collect();
                offset += x.len();

                let mut y_sample = vec![Complex32::default(); 2 * n];
                for i in 0..n {
                    q_sample.interp_execute(x[i], &mut y_sample[2 * i..2 * i + 2]).unwrap();
                }

                let mut y_block = vec![Complex32::default(); 2 * n];
                assert_eq!(q_block.interp_execute_block(&x, &mut y_block).unwrap(), 2 * n);

                for (&actual, &expected) in y_block.iter().zip(&y_sample) {
                    assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-5);
                    assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-5);
                }
                assert_eq!(q_block.w0.read(), q_sample.w0.read(), "m={m}, n={n}, w0");
                assert_eq!(q_block.w1.read(), q_sample.w1.read(), "m={m}, n={n}, w1");
            }
        }
    }

    #[test]
    fn test_resamp2_interp_block_matches_rrr() {
        for &m in &[2usize, 3, 5, 8, 16] {
            let mut q_sample = HalfBandResampler::<f32, f32>::new(m, 0.0, 60.0).unwrap();
            q_sample.set_scale(0.73);
            let mut q_block = q_sample.clone();

            let mut offset = 0usize;
            for n in [0, 1, m - 1, m, 2 * m - 1, 2 * m, 2 * m + 1, 73, 3] {
                let x: Vec<_> = (0..n).map(|i| ((offset + i) as f32 * 0.17).sin()).collect();
                offset += x.len();

                let mut y_sample = vec![0.0; 2 * n];
                for i in 0..n {
                    q_sample.interp_execute(x[i], &mut y_sample[2 * i..2 * i + 2]).unwrap();
                }

                let mut y_block = vec![0.0; 2 * n];
                assert_eq!(q_block.interp_execute_block(&x, &mut y_block).unwrap(), 2 * n);

                for (&actual, &expected) in y_block.iter().zip(&y_sample) {
                    assert_abs_diff_eq!(actual, expected, epsilon = 1e-5);
                }
                assert_eq!(q_block.w0.read(), q_sample.w0.read(), "m={m}, n={n}, w0");
                assert_eq!(q_block.w1.read(), q_sample.w1.read(), "m={m}, n={n}, w1");
            }
        }
    }
}
