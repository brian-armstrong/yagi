// firpfbch2 : finite impulse response polyphase filterbank channelizer
// with output rate 2 Fs / M

use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use crate::filter;

use num_complex::{Complex32, ComplexFloat};

use super::ChannelizerType;

/// FIR polyphase filter bank whose channels run at twice the critical rate.
///
/// A channelizer is built as either an analyzer or a synthesizer, and `M`,
/// the number of channels, must be even. The analyzer turns each block of
/// `M/2` time samples into one value per channel, so every channel runs at
/// `2/M` of the input rate. The synthesizer turns one value per channel into
/// `M/2` time samples. Channels are in natural FFT-bin order, and channel `k`
/// is centered at `k/M` cycles per time-domain sample.
///
/// An analyzer followed by a synthesizer with the same Kaiser design
/// approximately reconstructs the input, delayed by
/// `2 * M * filter_semi_length - M/2 + 1` samples.
///
/// Filter history carries across calls, and [`Self::reset`] clears it. A new
/// channelizer starts with zero-valued history.
#[derive(Clone, Debug)]
#[doc(alias = "FirPfbChannelizer2")]
pub struct OversampledPolyphaseChannelizer<T> {
    channelizer_type: ChannelizerType,
    num_channels: usize,
    num_channels_half: usize,
    m: usize,

    // bank of dotprod objects
    dp: Vec<Vec<f32>>,

    // inverse FFT plan
    ifft: Fft<f32>,
    x: Vec<Complex32>,
    x_out: Vec<Complex32>,

    // common data structures shared between analysis and synthesis algorithms
    w0: Vec<Window<T>>,
    w1: Vec<Window<T>>,
    flag: bool,
}

impl<T> OversampledPolyphaseChannelizer<T>
where
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + Into<Complex32> + From<Complex32>,
    [f32]: DotProd<T, Output = T>,
{
    /// Creates a filter bank from prototype coefficients.
    ///
    /// `filter_semi_length` is the prototype span on each side of its center,
    /// in units of `num_channels` samples. The first
    /// `2 * num_channels * filter_semi_length` coefficients are copied in time
    /// order and dealt out across the branches. Extra coefficients are
    /// ignored. The coefficients are not normalized, so they set the analysis
    /// gain and channel width. For analysis, a tone at a channel center
    /// produces an output of about the sum of the coefficients divided by
    /// `num_channels`.
    ///
    /// # Errors
    ///
    /// Returns an error when `num_channels` is odd or less than two, or
    /// `filter_semi_length` is zero.
    ///
    /// # Panics
    ///
    /// Panics when `coefficients` is shorter than
    /// `2 * num_channels * filter_semi_length`.
    pub fn new(
        channelizer_type: ChannelizerType,
        num_channels: usize,
        filter_semi_length: usize,
        coefficients: &[f32],
    ) -> Result<Self> {
        if num_channels < 2 || !num_channels.is_multiple_of(2) {
            return Err(Error::Config("number of channels must be greater than 2 and even".into()));
        }
        if filter_semi_length < 1 {
            return Err(Error::Config("filter semi-length must be at least 1".into()));
        }

        let m = filter_semi_length;
        let h = coefficients;

        // set input parameters
        let num_channels_half = num_channels / 2;

        // generate bank of sub-sampled filters
        let mut dp = Vec::with_capacity(num_channels);
        let h_sub_len = 2 * m;
        for i in 0..num_channels {
            let mut h_sub = vec![0.0f32; h_sub_len];
            // sub-sample prototype filter, loading coefficients in reverse order
            for n in 0..h_sub_len {
                h_sub[h_sub_len - n - 1] = h[i + n * num_channels];
            }
            dp.push(h_sub);
        }

        // create FFT plan (inverse transform)
        let x = vec![Complex32::new(0.0, 0.0); num_channels];
        let x_out = vec![Complex32::new(0.0, 0.0); num_channels];
        let ifft = Fft::new(num_channels, Direction::Backward);

        // create buffer objects
        let mut w0 = Vec::with_capacity(num_channels);
        let mut w1 = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            w0.push(Window::new(h_sub_len)?);
            w1.push(Window::new(h_sub_len)?);
        }

        let mut q =
            Self { channelizer_type, num_channels, num_channels_half, m, dp, ifft, x, x_out, w0, w1, flag: false };

        q.reset();
        Ok(q)
    }

    /// Creates a filter bank with a Kaiser-windowed prototype.
    ///
    /// `filter_semi_length` is the prototype span on each side of its center,
    /// in units of `num_channels` samples. `stopband_attenuation` is in dB.
    /// The prototype is scaled so its coefficients sum to `num_channels`.
    ///
    /// The analyzer's cutoff is `1 / num_channels` cycles per time-domain
    /// sample, twice the synthesizer's, so each analysis channel reaches the
    /// centers of its neighbors. After startup, a unit tone at a channel center
    /// produces an analyzer output of about one in that channel and about
    /// `0.5` in each neighbor. A unit value held in one synthesizer channel
    /// produces a tone of amplitude about one.
    ///
    /// # Errors
    ///
    /// Returns an error when `num_channels` is odd or less than two,
    /// `filter_semi_length` is zero, or `stopband_attenuation` is zero or
    /// negative.
    pub fn new_kaiser(
        channelizer_type: ChannelizerType,
        num_channels: usize,
        filter_semi_length: usize,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        if num_channels < 2 || !num_channels.is_multiple_of(2) {
            return Err(Error::Config("number of channels must be greater than 2 and even".into()));
        }
        if filter_semi_length < 1 {
            return Err(Error::Config("filter semi-length must be at least 1".into()));
        }

        let m = filter_semi_length;
        let as_ = stopband_attenuation;

        // design prototype filter
        let h_len = 2 * num_channels * m + 1;

        // filter cut-off frequency (analyzer has twice the bandwidth of the synthesizer)
        let fc = match channelizer_type {
            ChannelizerType::Analyzer => 1.0 / num_channels as f32,
            ChannelizerType::Synthesizer => 0.5 / num_channels as f32,
        };

        // compute filter coefficients (floating point precision)
        let hf = filter::fir::design::fir_design_kaiser(h_len, fc, as_, 0.0)?;

        // normalize to unit average and scale by number of channels
        let hf_sum: f32 = hf.iter().sum();
        let h: Vec<f32> = hf.iter().map(|&x| x * num_channels as f32 / hf_sum).collect();

        // create filterbank channelizer object
        Self::new(channelizer_type, num_channels, m, &h)
    }

    /// Clears the filter history to zeros.
    ///
    /// The channelizer then behaves like a new one.
    pub fn reset(&mut self) {
        for i in 0..self.num_channels {
            self.w0[i].reset();
            self.w1[i].reset();
        }
        self.flag = false;
    }

    /// Returns whether this channelizer analyzes or synthesizes.
    pub fn channelizer_type(&self) -> ChannelizerType {
        self.channelizer_type
    }

    /// Returns the number of channels.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Returns the prototype semi-length, in units of
    /// [`Self::num_channels`] samples.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Analyzes `M/2` time samples into one value per channel.
    ///
    /// `M` is [`Self::num_channels`]. Reads the first `M/2` samples of `input`
    /// and writes the first `M` values of `output`. Extra input is ignored and
    /// extra output is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the channelizer is a synthesizer, `input` holds
    /// fewer than `M/2` samples, or `output` holds fewer than `M`. The
    /// channelizer and `output` are unchanged on error.
    pub fn execute_analyzer(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        if self.channelizer_type != ChannelizerType::Analyzer {
            return Err(Error::Config("cannot execute analyzer on synthesizer channelizer".into()));
        }
        if input.len() < self.num_channels_half {
            return Err(Error::Config("input length must be at least num_channels/2".into()));
        }
        if output.len() < self.num_channels {
            return Err(Error::Config("output length must be at least num_channels".into()));
        }

        // load buffers in blocks of num_channels/2 starting
        // in the middle of the filter bank and moving in the negative direction
        let base_index = if self.flag { self.num_channels } else { self.num_channels_half };
        for i in 0..self.num_channels_half {
            // push sample into buffer at filter index
            self.w0[base_index - i - 1].push(input[i]);
        }

        // execute filter outputs
        let offset = if self.flag { self.num_channels_half } else { 0 };
        for i in 0..self.num_channels {
            // read buffer at index
            let r = self.w0[i].read();

            // run dot product storing result in IFFT input buffer
            let result: T = self.dp[(offset + i) % self.num_channels].dotprod(r);
            self.x[i] = result.into();
        }

        // execute IFFT, store result in buffer 'x_out'
        self.ifft.run(&self.x, &mut self.x_out);

        // scale result by 1/num_channels (C transform)
        let scale = 1.0 / self.num_channels as f32;
        for (yi, &xo) in output[..self.num_channels].iter_mut().zip(&self.x_out[..self.num_channels]) {
            *yi = <T as From<Complex32>>::from(xo * scale);
        }

        // update flag
        self.flag = !self.flag;
        Ok(())
    }

    /// Synthesizes `M/2` time samples from one value per channel.
    ///
    /// `M` is [`Self::num_channels`]. Reads the first `M` values of `input` and
    /// writes the first `M/2` samples of `output`. Extra input is ignored and
    /// extra output is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the channelizer is an analyzer, `input` holds
    /// fewer than `M` values, or `output` holds fewer than `M/2`. The
    /// channelizer and `output` are unchanged on error.
    pub fn execute_synthesizer(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        if self.channelizer_type != ChannelizerType::Synthesizer {
            return Err(Error::Config("cannot execute synthesizer on analyzer channelizer".into()));
        }
        if input.len() < self.num_channels {
            return Err(Error::Config("input length must be at least num_channels".into()));
        }
        if output.len() < self.num_channels_half {
            return Err(Error::Config("output length must be at least num_channels/2".into()));
        }

        // copy input array to internal IFFT input buffer
        for (local, &input) in self.x[..self.num_channels].iter_mut().zip(&input[..self.num_channels]) {
            *local = input.into();
        }

        // execute IFFT, store result in buffer 'x_out'
        self.ifft.run(&self.x, &mut self.x_out);

        // scale result by 1/num_channels (C transform) and by num_channels/2
        let scale = 1.0 / self.num_channels as f32 * self.num_channels_half as f32;
        for i in 0..self.num_channels {
            self.x_out[i] *= scale;
        }

        // push samples into appropriate buffer
        let buffer = if !self.flag { &mut self.w1 } else { &mut self.w0 };
        for (buf, &xo) in buffer[..self.num_channels].iter_mut().zip(&self.x_out[..self.num_channels]) {
            buf.push(<T as From<Complex32>>::from(xo));
        }

        // compute filter outputs
        for i in 0..self.num_channels_half {
            // buffer index
            let b = if !self.flag { i } else { i + self.num_channels_half };

            // read buffer with index offset
            let r0 = self.w0[b].read();
            let r1 = self.w1[b].read();

            // swap buffer outputs on alternating runs
            let (p0, p1) = if self.flag { (r0, r1) } else { (r1, r0) };

            // run dot products
            let y0: T = self.dp[i].dotprod(p0);
            let y1: T = self.dp[i + self.num_channels_half].dotprod(p1);

            // save output
            output[i] = <T as From<Complex32>>::from(y0.into() + y1.into());
        }

        self.flag = !self.flag;
        Ok(())
    }

    /// Runs [`Self::execute_analyzer`] or [`Self::execute_synthesizer`],
    /// whichever matches [`Self::channelizer_type`].
    ///
    /// # Errors
    ///
    /// Returns an error when `input` or `output` is too short for the selected
    /// direction. The channelizer and `output` are unchanged on error.
    pub fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        match self.channelizer_type {
            ChannelizerType::Analyzer => self.execute_analyzer(input, output),
            ChannelizerType::Synthesizer => self.execute_synthesizer(input, output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    fn firpfbch2_crcf_runtest(num_channels: usize, m: usize, as_: f32) {
        let tol = 1e-3f32;

        // derived values
        let num_symbols = 8 * m;
        let num_samples = num_channels * num_symbols;

        // allocate arrays
        let mut x = vec![Complex32::new(0.0, 0.0); num_samples];
        let mut y = vec![Complex32::new(0.0, 0.0); num_samples];

        // generate pseudo-random sequence
        let mut s: u32 = 1;
        let p: u32 = 524287;
        let g: u32 = 1031;
        for i in 0..num_samples {
            s = (s.wrapping_mul(p)) % g;
            x[i] = Complex32::new(s as f32 / g as f32 - 0.5, 0.0);
        }

        // create filterbank objects from prototype
        let mut qa =
            OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, num_channels, m, as_)
                .unwrap();
        let mut qs = OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(
            ChannelizerType::Synthesizer,
            num_channels,
            m,
            as_,
        )
        .unwrap();

        // run channelizer
        let mut y_channels = vec![Complex32::new(0.0, 0.0); num_channels];
        let half = num_channels / 2;
        for i in (0..num_samples).step_by(half) {
            // run analysis filterbank
            qa.execute(&x[i..i + half], &mut y_channels).unwrap();

            // run synthesis filterbank
            qs.execute(&y_channels, &mut y[i..i + half]).unwrap();
        }

        // validate output
        let delay = 2 * num_channels * m - half + 1;
        let mut rmse = 0.0f32;
        for i in 0..num_samples {
            if i < delay {
                assert_abs_diff_eq!(y[i].re, 0.0, epsilon = tol);
                assert_abs_diff_eq!(y[i].im, 0.0, epsilon = tol);
            } else {
                assert_abs_diff_eq!(y[i].re, x[i - delay].re, epsilon = tol);
                assert_abs_diff_eq!(y[i].im, x[i - delay].im, epsilon = tol);
            }

            // compute rmse
            let expected = if i < delay { Complex32::new(0.0, 0.0) } else { x[i - delay] };
            let err = y[i] - expected;
            rmse += (err * err.conj()).re;
        }

        rmse = (rmse / num_samples as f32).sqrt();
        println!("firpfbch2: M={}, m={}, as={:.2} dB, rmse={:.4e}", num_channels, m, as_, rmse);
    }

    #[test]
    #[autotest_annotate(autotest_firpfbch2_crcf_n8)]
    fn test_firpfbch2_crcf_n8() {
        firpfbch2_crcf_runtest(8, 5, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_firpfbch2_crcf_n16)]
    fn test_firpfbch2_crcf_n16() {
        firpfbch2_crcf_runtest(16, 5, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_firpfbch2_crcf_n32)]
    fn test_firpfbch2_crcf_n32() {
        firpfbch2_crcf_runtest(32, 5, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_firpfbch2_crcf_n64)]
    fn test_firpfbch2_crcf_n64() {
        firpfbch2_crcf_runtest(64, 5, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_firpfbch2_crcf_copy)]
    fn test_firpfbch2_crcf_copy() {
        let num_channels = 72;
        let m = 12;
        let as_ = 80.0f32;
        let mut q_orig =
            OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, num_channels, m, as_)
                .unwrap();

        let half = num_channels / 2;
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); half];
        let mut buf_1_orig = vec![Complex32::new(0.0, 0.0); num_channels];
        let mut buf_1_copy = vec![Complex32::new(0.0, 0.0); num_channels];

        // start running input through filter
        let num_blocks = 32;
        for _ in 0..num_blocks {
            for j in 0..half {
                buf_0[j] = Complex32::new(crate::random::randnf(), crate::random::randnf());
            }
            q_orig.execute(&buf_0, &mut buf_1_orig).unwrap();
        }

        // copy object
        let mut q_copy = q_orig.clone();

        // continue running through both objects
        for _ in 0..num_blocks {
            for j in 0..half {
                buf_0[j] = Complex32::new(crate::random::randnf(), crate::random::randnf());
            }

            // run filters in parallel
            q_orig.execute(&buf_0, &mut buf_1_orig).unwrap();
            q_copy.execute(&buf_0, &mut buf_1_copy).unwrap();

            assert_eq!(buf_1_orig, buf_1_copy);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firpfbch2_crcf_config)]
    fn test_firpfbch2_crcf_config() {
        // check invalid function calls
        assert!(OversampledPolyphaseChannelizer::<Complex32>::new(ChannelizerType::Analyzer, 0, 12, &[]).is_err());
        assert!(OversampledPolyphaseChannelizer::<Complex32>::new(ChannelizerType::Analyzer, 17, 12, &[]).is_err());
        assert!(OversampledPolyphaseChannelizer::<Complex32>::new(ChannelizerType::Analyzer, 76, 0, &[]).is_err());

        assert!(
            OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 0, 12, 60.0).is_err()
        );
        assert!(
            OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 17, 12, 60.0).is_err()
        );
        assert!(
            OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 76, 0, 60.0).is_err()
        );

        // create proper object and test configurations
        let q =
            OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 76, 12, 60.0).unwrap();
        assert_eq!(q.channelizer_type(), ChannelizerType::Analyzer);
        assert_eq!(q.num_channels(), 76);
        assert_eq!(q.m(), 12);
    }

    #[test]
    fn test_firpfbch2_execute() {
        let channels = 8;
        let half = channels / 2;
        let sentinel = Complex32::new(-9.0, -9.0);

        let mut analyzer =
            OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, channels, 3, 60.0)
                .unwrap();
        let mut analyzer_ref = analyzer.clone();
        let input = [Complex32::new(1.0, -0.5); 4];
        assert!(analyzer.execute_analyzer(&input[..half - 1], &mut [Complex32::default(); 8]).is_err());
        assert!(analyzer.execute_analyzer(&input, &mut [Complex32::default(); 7]).is_err());
        let mut actual = [sentinel; 10];
        let mut expected = [Complex32::default(); 8];
        analyzer.execute_analyzer(&input, &mut actual).unwrap();
        analyzer_ref.execute_analyzer(&input, &mut expected).unwrap();
        assert_eq!(&actual[..channels], &expected);
        assert_eq!(&actual[channels..], &[sentinel; 2]);

        let mut synthesizer =
            OversampledPolyphaseChannelizer::<Complex32>::new_kaiser(ChannelizerType::Synthesizer, channels, 3, 60.0)
                .unwrap();
        let mut synthesizer_ref = synthesizer.clone();
        let input = [Complex32::new(1.0, -0.5); 8];
        assert!(synthesizer.execute_synthesizer(&input[..channels - 1], &mut [Complex32::default(); 4]).is_err());
        assert!(synthesizer.execute_synthesizer(&input, &mut [Complex32::default(); 3]).is_err());
        let mut actual = [sentinel; 6];
        let mut expected = [Complex32::default(); 4];
        synthesizer.execute_synthesizer(&input, &mut actual).unwrap();
        synthesizer_ref.execute_synthesizer(&input, &mut expected).unwrap();
        assert_eq!(&actual[..half], &expected);
        assert_eq!(&actual[half..], &[sentinel; 2]);
    }
}
