// firpfbch2 : finite impulse response polyphase filterbank channelizer
// with output rate 2 Fs / M

use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use crate::filter;

use num_complex::{Complex32, ComplexFloat};

use super::ChannelizerType;

/// Finite impulse response polyphase filterbank channelizer with output rate 2 Fs / M
#[derive(Clone, Debug)]
pub struct FirPfbChannelizer2<T> {
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

impl<T> FirPfbChannelizer2<T>
where
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + Into<Complex32> + From<Complex32>,
    [f32]: DotProd<T, Output = T>,
{
    /// Create firpfbch2 object
    ///
    /// # Arguments
    ///
    /// * `channelizer_type` - channelizer type (Analyzer or Synthesizer)
    /// * `num_channels` - number of channels (must be even)
    /// * `m` - prototype filter semi-length, length=2*M*m
    /// * `h` - prototype filter coefficient array
    pub fn new(channelizer_type: ChannelizerType, num_channels: usize, m: usize, h: &[f32]) -> Result<Self> {
        if num_channels < 2 || num_channels % 2 != 0 {
            return Err(Error::Config("number of channels must be greater than 2 and even".into()));
        }
        if m < 1 {
            return Err(Error::Config("filter semi-length must be at least 1".into()));
        }

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

        let mut q = Self { channelizer_type, num_channels, num_channels_half, m, dp, ifft, x, x_out, w0, w1, flag: false };

        q.reset();
        Ok(q)
    }

    /// Create firpfbch2 object using Kaiser window prototype
    ///
    /// # Arguments
    ///
    /// * `channelizer_type` - channelizer type (Analyzer or Synthesizer)
    /// * `num_channels` - number of channels (must be even)
    /// * `m` - prototype filter semi-length, length=2*M*m+1
    /// * `as_` - filter stop-band attenuation [dB]
    pub fn new_kaiser(channelizer_type: ChannelizerType, num_channels: usize, m: usize, as_: f32) -> Result<Self> {
        if num_channels < 2 || num_channels % 2 != 0 {
            return Err(Error::Config("number of channels must be greater than 2 and even".into()));
        }
        if m < 1 {
            return Err(Error::Config("filter semi-length must be at least 1".into()));
        }

        // design prototype filter
        let h_len = 2 * num_channels * m + 1;

        // filter cut-off frequency (analyzer has twice the bandwidth of the synthesizer)
        let fc = match channelizer_type {
            ChannelizerType::Analyzer => 1.0 / num_channels as f32,
            ChannelizerType::Synthesizer => 0.5 / num_channels as f32,
        };

        // compute filter coefficients (floating point precision)
        let hf = filter::fir_design_kaiser(h_len, fc, as_, 0.0)?;

        // normalize to unit average and scale by number of channels
        let hf_sum: f32 = hf.iter().sum();
        let h: Vec<f32> = hf.iter().map(|&x| x * num_channels as f32 / hf_sum).collect();

        // create filterbank channelizer object
        Self::new(channelizer_type, num_channels, m, &h)
    }

    /// Reset firpfbch2 object internals
    pub fn reset(&mut self) {
        for i in 0..self.num_channels {
            self.w0[i].reset();
            self.w1[i].reset();
        }
        self.flag = false;
    }

    /// Get channelizer type
    pub fn get_type(&self) -> ChannelizerType {
        self.channelizer_type
    }

    /// Get number of channels
    pub fn get_num_channels(&self) -> usize {
        self.num_channels
    }

    /// Get prototype filter semi-length
    pub fn get_m(&self) -> usize {
        self.m
    }

    /// Execute filterbank channelizer (analyzer)
    ///
    /// # Arguments
    ///
    /// * `x` - channelizer input, [size: num_channels/2 x 1]
    /// * `y` - channelizer output, [size: num_channels x 1]
    pub fn execute_analyzer(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if self.channelizer_type != ChannelizerType::Analyzer {
            return Err(Error::Config("cannot execute analyzer on synthesizer channelizer".into()));
        }

        // load buffers in blocks of num_channels/2 starting
        // in the middle of the filter bank and moving in the negative direction
        let base_index = if self.flag { self.num_channels } else { self.num_channels_half };
        for i in 0..self.num_channels_half {
            // push sample into buffer at filter index
            self.w0[base_index - i - 1].push(x[i]);
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
        for i in 0..self.num_channels {
            y[i] = <T as From<Complex32>>::from(self.x_out[i] * scale);
        }

        // update flag
        self.flag = !self.flag;
        Ok(())
    }

    /// Execute filterbank channelizer (synthesizer)
    ///
    /// # Arguments
    ///
    /// * `x` - channelizer input, [size: num_channels x 1]
    /// * `y` - channelizer output, [size: num_channels/2 x 1]
    pub fn execute_synthesizer(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if self.channelizer_type != ChannelizerType::Synthesizer {
            return Err(Error::Config("cannot execute synthesizer on analyzer channelizer".into()));
        }

        // copy input array to internal IFFT input buffer
        for i in 0..self.num_channels {
            self.x[i] = x[i].into();
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
        for i in 0..self.num_channels {
            buffer[i].push(<T as From<Complex32>>::from(self.x_out[i]));
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
            y[i] = <T as From<Complex32>>::from(y0.into() + y1.into());
        }

        self.flag = !self.flag;
        Ok(())
    }

    /// Execute filterbank channelizer
    /// ANALYZER: input: M/2, output: M
    /// SYNTHESIZER: input: M, output: M/2
    pub fn execute(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        match self.channelizer_type {
            ChannelizerType::Analyzer => self.execute_analyzer(x, y),
            ChannelizerType::Synthesizer => self.execute_synthesizer(x, y),
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
        let mut qa = FirPfbChannelizer2::<Complex32>::new_kaiser(ChannelizerType::Analyzer, num_channels, m, as_).unwrap();
        let mut qs = FirPfbChannelizer2::<Complex32>::new_kaiser(ChannelizerType::Synthesizer, num_channels, m, as_).unwrap();

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
        let mut q_orig = FirPfbChannelizer2::<Complex32>::new_kaiser(ChannelizerType::Analyzer, num_channels, m, as_).unwrap();

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
        assert!(FirPfbChannelizer2::<Complex32>::new(ChannelizerType::Analyzer, 0, 12, &[]).is_err());
        assert!(FirPfbChannelizer2::<Complex32>::new(ChannelizerType::Analyzer, 17, 12, &[]).is_err());
        assert!(FirPfbChannelizer2::<Complex32>::new(ChannelizerType::Analyzer, 76, 0, &[]).is_err());

        assert!(FirPfbChannelizer2::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 0, 12, 60.0).is_err());
        assert!(FirPfbChannelizer2::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 17, 12, 60.0).is_err());
        assert!(FirPfbChannelizer2::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 76, 0, 60.0).is_err());

        // create proper object and test configurations
        let q = FirPfbChannelizer2::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 76, 12, 60.0).unwrap();
        assert_eq!(q.get_type(), ChannelizerType::Analyzer);
        assert_eq!(q.get_num_channels(), 76);
        assert_eq!(q.get_m(), 12);
    }
}
