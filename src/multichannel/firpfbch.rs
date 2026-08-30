// firpfbch : finite impulse response polyphase filterbank channelizer

use crate::buffer::Window;
use crate::dotprod::{DotProd, DotProduct};
use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use crate::filter;

use num_complex::{Complex32, ComplexFloat};

/// Channelizer type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChannelizerType {
    /// Analysis channelizer (time to frequency)
    Analyzer,
    /// Synthesis channelizer (frequency to time)
    Synthesizer,
}

/// Finite impulse response polyphase filterbank channelizer
#[derive(Clone, Debug)]
pub struct FirPfbChannelizer<T, Coeff = f32> {
    channelizer_type: ChannelizerType,
    num_channels: usize,
    p: usize,

    // bank of dotprod objects and window buffers
    dp: Vec<DotProduct<T, Coeff>>,
    w: Vec<Window<T>>,
    filter_index: usize,

    // fft plan
    fft: Fft<f32>,
    x: Vec<Complex32>,
    x_out: Vec<Complex32>,
}

impl<T, Coeff> FirPfbChannelizer<T, Coeff>
where
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + Into<Complex32> + From<Complex32>,
    Coeff: Clone + Copy + From<f32>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Create FIR polyphase filterbank channelizer object
    ///
    /// # Arguments
    ///
    /// * `channelizer_type` - channelizer type (Analyzer or Synthesizer)
    /// * `num_channels` - number of channels
    /// * `p` - filter length (symbols)
    /// * `h` - filter coefficients, [size: num_channels * p x 1]
    pub fn new(channelizer_type: ChannelizerType, num_channels: usize, p: usize, h: &[Coeff]) -> Result<Self> {
        if num_channels == 0 {
            return Err(Error::Config("number of channels must be greater than 0".into()));
        }
        if p == 0 {
            return Err(Error::Config("filter size must be greater than 0".into()));
        }

        // create bank of filters
        let mut dp = Vec::with_capacity(num_channels);
        let mut w = Vec::with_capacity(num_channels);

        // copy filter coefficients
        let h_len = num_channels * p;
        let h_copy: Vec<Coeff> = h[..h_len].to_vec();

        // generate bank of sub-sampled filters
        let h_sub_len = p;
        for i in 0..num_channels {
            let mut h_sub = vec![0.0f32.into(); h_sub_len];
            // sub-sample prototype filter, loading coefficients in reverse order
            for n in 0..h_sub_len {
                h_sub[h_sub_len - n - 1] = h_copy[i + n * num_channels];
            }
            dp.push(DotProduct::new(&h_sub)?);
            w.push(Window::new(h_sub_len)?);
        }

        // allocate memory for buffers
        let x = vec![Complex32::new(0.0, 0.0); num_channels];
        let x_out = vec![Complex32::new(0.0, 0.0); num_channels];

        // create fft plan
        let fft = match channelizer_type {
            ChannelizerType::Analyzer => Fft::new(num_channels, Direction::Forward),
            ChannelizerType::Synthesizer => Fft::new(num_channels, Direction::Backward),
        };

        let mut q = Self { 
            channelizer_type,
            num_channels,
            p,
            dp,
            w,
            filter_index: 0,
            fft,
            x,
            x_out,
        };

        q.reset();
        Ok(q)
    }

    /// Create FIR polyphase filterbank channelizer object with
    /// prototype filter based on windowed Kaiser design
    ///
    /// # Arguments
    ///
    /// * `channelizer_type` - channelizer type (Analyzer or Synthesizer)
    /// * `num_channels` - number of channels
    /// * `m` - filter delay (symbols)
    /// * `as_` - stop-band attenuation [dB]
    pub fn new_kaiser(channelizer_type: ChannelizerType, num_channels: usize, m: usize, as_: f32) -> Result<Self> {
        if num_channels == 0 {
            return Err(Error::Config("number of channels must be greater than 0".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter size must be greater than 0".into()));
        }

        let as_ = as_.abs();

        // design filter
        let h_len = 2 * num_channels * m + 1;
        let fc = 0.5 / num_channels as f32;
        let hf = filter::fir_design_kaiser(h_len, fc, as_, 0.0)?;

        // convert to coefficient type
        let h: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();

        // create filterbank object
        let p = 2 * m;
        Self::new(channelizer_type, num_channels, p, &h)
    }

    /// Create FIR polyphase filterbank channelizer object with
    /// prototype root-Nyquist filter
    ///
    /// # Arguments
    ///
    /// * `channelizer_type` - channelizer type (Analyzer or Synthesizer)
    /// * `num_channels` - number of channels
    /// * `m` - filter delay (symbols)
    /// * `beta` - filter excess bandwidth factor, in [0,1]
    /// * `ftype` - filter prototype (rrcos, rkaiser, etc.)
    pub fn new_rnyquist(
        channelizer_type: ChannelizerType,
        num_channels: usize,
        m: usize,
        beta: f32,
        ftype: filter::FirFilterShape,
    ) -> Result<Self> {
        if num_channels == 0 {
            return Err(Error::Config("number of channels must be greater than 0".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter size must be greater than 0".into()));
        }

        // design filter based on requested prototype
        let h = filter::fir_design_prototype(ftype, num_channels, m, beta, 0.0)?;

        // copy coefficients to type-specific array, reversing order if
        // channelizer is an analyzer, matched filter: g(-t)
        let g_len = 2 * num_channels * m;
        let mut gc: Vec<Coeff> = vec![0.0f32.into(); g_len];
        match channelizer_type {
            ChannelizerType::Synthesizer => {
                for i in 0..g_len {
                    gc[i] = h[i].into();
                }
            }
            ChannelizerType::Analyzer => {
                for i in 0..g_len {
                    gc[i] = h[g_len - i - 1].into();
                }
            }
        }

        // create filterbank object
        let p = 2 * m;
        Self::new(channelizer_type, num_channels, p, &gc)
    }

    /// Reset filterbank object internals
    pub fn reset(&mut self) {
        for i in 0..self.num_channels {
            self.w[i].reset();
            self.x[i] = Complex32::new(0.0, 0.0);
            self.x_out[i] = Complex32::new(0.0, 0.0);
        }
        self.filter_index = self.num_channels - 1;
    }

    /// Get channelizer type
    pub fn get_type(&self) -> ChannelizerType {
        self.channelizer_type
    }

    /// Get number of channels
    pub fn get_num_channels(&self) -> usize {
        self.num_channels
    }

    /// Get filter semi-length
    pub fn get_m(&self) -> usize {
        self.p / 2
    }

    /// Execute filterbank as synthesizer on block of samples
    ///
    /// # Arguments
    ///
    /// * `x` - channelized input, [size: num_channels x 1]
    /// * `y` - output time series, [size: num_channels x 1]
    pub fn synthesizer_execute(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if self.channelizer_type != ChannelizerType::Synthesizer {
            return Err(Error::Config("cannot execute synthesizer on analyzer channelizer".into()));
        }

        // copy channelized symbols to transform input
        for i in 0..self.num_channels {
            self.x[i] = x[i].into();
        }

        // execute inverse DFT, store result in buffer 'x_out'
        self.fft.run(&self.x, &mut self.x_out);

        // push samples into filter bank and execute
        for i in 0..self.num_channels {
            self.w[i].push(<T as From<Complex32>>::from(self.x_out[i]));
            let r = self.w[i].read();
            y[i] = self.dp[i].execute(r);
        }

        Ok(())
    }

    /// Execute filterbank as analyzer on block of samples
    ///
    /// # Arguments
    ///
    /// * `x` - input time series, [size: num_channels x 1]
    /// * `y` - channelized output, [size: num_channels x 1]
    pub fn analyzer_execute(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if self.channelizer_type != ChannelizerType::Analyzer {
            return Err(Error::Config("cannot execute analyzer on synthesizer channelizer".into()));
        }

        // push samples into buffers
        for i in 0..self.num_channels {
            self.analyzer_push(x[i]);
        }

        // execute analysis filters on the given input starting
        // with filterbank at index zero
        self.analyzer_run(0, y)
    }

    /// Push single sample into analysis filterbank, updating index
    /// counter appropriately
    fn analyzer_push(&mut self, x: T) {
        // push sample into filter
        self.w[self.filter_index].push(x);

        // decrement filter index
        self.filter_index = (self.filter_index + self.num_channels - 1) % self.num_channels;
    }

    /// Run filterbank analyzer dot products, DFT
    fn analyzer_run(&mut self, k: usize, y: &mut [T]) -> Result<()> {
        // execute filter outputs, reversing order of output
        for i in 0..self.num_channels {
            // compute appropriate index
            let index = (i + k) % self.num_channels;

            // read buffer at specified index
            let r = self.w[index].read();

            // compute dot product
            let result: T = self.dp[i].execute(r);
            self.x[self.num_channels - i - 1] = result.into();
        }

        // execute DFT, store result in buffer 'x_out'
        self.fft.run(&self.x, &mut self.x_out);

        // move to output array
        for i in 0..self.num_channels {
            y[i] = <T as From<Complex32>>::from(self.x_out[i]);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    use crate::filter::FirFilter;
    use crate::sequence::MSequence;

    #[test]
    #[autotest_annotate(autotest_firpfbch_crcf_config)]
    fn test_firpfbch_crcf_config() {
        // check invalid function calls
        assert!(FirPfbChannelizer::<Complex32>::new(ChannelizerType::Analyzer, 0, 12, &[]).is_err());
        assert!(FirPfbChannelizer::<Complex32>::new(ChannelizerType::Analyzer, 76, 0, &[]).is_err());

        assert!(FirPfbChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 0, 12, 60.0).is_err());
        assert!(FirPfbChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 76, 0, 60.0).is_err());

        assert!(
            FirPfbChannelizer::<Complex32>::new_rnyquist(ChannelizerType::Analyzer, 0, 12, 0.2, filter::FirFilterShape::Arkaiser).is_err()
        );
        assert!(
            FirPfbChannelizer::<Complex32>::new_rnyquist(ChannelizerType::Analyzer, 76, 0, 0.2, filter::FirFilterShape::Arkaiser).is_err()
        );
        assert!(
            FirPfbChannelizer::<Complex32>::new_rnyquist(ChannelizerType::Analyzer, 76, 12, 77.0, filter::FirFilterShape::Arkaiser).is_err()
        ); // invalid filter excess bandwidth

        // create proper object and test configurations
        let q = FirPfbChannelizer::<Complex32>::new_kaiser(ChannelizerType::Analyzer, 76, 12, 60.0).unwrap();
        assert_eq!(q.get_type(), ChannelizerType::Analyzer);
        assert_eq!(q.get_num_channels(), 76);
        assert_eq!(q.get_m(), 12);
    }

    #[test]
    #[autotest_annotate(autotest_firpfbch_crcf_analysis)]
    fn test_firpfbch_crcf_analysis() {
        let tol = 1e-4f32;
        let num_channels = 4;
        let p = 5;
        let num_symbols = 40;

        // derived values
        let num_samples = num_channels * num_symbols;

        // generate filter coefficients using m-sequence
        let h_len = p * num_channels;
        let mut h = vec![0.0f32; h_len];
        let mut ms = MSequence::create_default(6).unwrap();
        for i in 0..h_len {
            h[i] = ms.generate_symbol(2) as f32 - 1.5;
        }

        // create filterbank object
        let mut q = FirPfbChannelizer::<Complex32, f32>::new(ChannelizerType::Analyzer, num_channels, p, &h).unwrap();

        // create filter object
        let mut f = FirFilter::<Complex32, f32>::new(&h).unwrap();

        // allocate memory for arrays
        let mut y = vec![Complex32::new(0.0, 0.0); num_samples];
        let mut y0 = vec![vec![Complex32::new(0.0, 0.0); num_channels]; num_symbols];
        let mut y1 = vec![vec![Complex32::new(0.0, 0.0); num_channels]; num_symbols];

        // generate input sequence (complex noise)
        let mut ms = MSequence::create_default(7).unwrap();
        for i in 0..num_samples {
            y[i] = Complex32::new(
                0.1 * std::f32::consts::FRAC_1_SQRT_2 * (ms.generate_symbol(2) as f32 - 1.5),
                0.1 * std::f32::consts::FRAC_1_SQRT_2 * (ms.generate_symbol(2) as f32 - 1.5),
            );
        }

        // run analysis filter bank
        for i in 0..num_symbols {
            q.analyzer_execute(&y[i * num_channels..(i + 1) * num_channels], &mut y0[i]).unwrap();
        }

        // run traditional down-converter (inefficient)
        for i in 0..num_channels {
            // reset filter
            f.reset();

            // set center frequency
            let dphi = 2.0 * std::f32::consts::PI * (i as f32) / (num_channels as f32);

            // reset symbol counter
            let mut n = 0;

            for j in 0..num_samples {
                // push down-converted sample into filter
                let expjwt = Complex32::new(0.0, -(j as f32) * dphi).exp();
                f.push(y[j] * expjwt);

                // compute output at the appropriate sample time
                if ((j + 1) % num_channels) == 0 {
                    y1[n][i] = f.execute();
                    n += 1;
                }
            }
            assert_eq!(n, num_symbols);
        }

        // compare results
        for i in 0..num_symbols {
            for j in 0..num_channels {
                assert_abs_diff_eq!(y0[i][j].re, y1[i][j].re, epsilon = tol);
                assert_abs_diff_eq!(y0[i][j].im, y1[i][j].im, epsilon = tol);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_firpfbch_crcf_synthesis)]
    fn test_firpfbch_crcf_synthesis() {
        let tol = 1e-4f32;
        let num_channels = 4;
        let p = 5;
        let num_symbols = 40;

        // derived values
        let num_samples = num_channels * num_symbols;

        // generate filter coefficients using m-sequence
        let h_len = p * num_channels;
        let mut h = vec![0.0f32; h_len];
        let mut ms = MSequence::create_default(6).unwrap();
        for i in 0..h_len {
            h[i] = ms.generate_symbol(2) as f32 - 1.5;
        }

        // create filter object
        let mut f = FirFilter::<Complex32, f32>::new(&h).unwrap();

        // create filterbank channelizer object
        let mut q = FirPfbChannelizer::<Complex32, f32>::new(ChannelizerType::Synthesizer, num_channels, p, &h).unwrap();

        let mut y_input = vec![vec![Complex32::new(0.0, 0.0); num_channels]; num_symbols];
        let mut y0 = vec![Complex32::new(0.0, 0.0); num_samples];
        let mut y1 = vec![Complex32::new(0.0, 0.0); num_samples];

        // generate input sequence (complex noise)
        let mut ms = MSequence::create_default(7).unwrap();
        for i in 0..num_symbols {
            for j in 0..num_channels {
                y_input[i][j] = Complex32::new(
                    0.1 * std::f32::consts::FRAC_1_SQRT_2 * (ms.generate_symbol(2) as f32 - 1.5),
                    0.1 * std::f32::consts::FRAC_1_SQRT_2 * (ms.generate_symbol(2) as f32 - 1.5),
                );
            }
        }

        // run synthesis filter bank
        for i in 0..num_symbols {
            q.synthesizer_execute(&y_input[i], &mut y0[i * num_channels..(i + 1) * num_channels]).unwrap();
        }

        // run traditional up-converter (inefficient)
        // clear output array
        for i in 0..num_samples {
            y1[i] = Complex32::new(0.0, 0.0);
        }

        for i in 0..num_channels {
            // reset filter
            f.reset();

            // set center frequency
            let dphi = 2.0 * std::f32::consts::PI * (i as f32) / (num_channels as f32);

            // reset input symbol counter
            let mut n = 0;

            for j in 0..num_samples {
                // interpolate sequence
                if (j % num_channels) == 0 {
                    f.push(y_input[n][i]);
                    n += 1;
                } else {
                    f.push(Complex32::new(0.0, 0.0));
                }
                let y_hat = f.execute();

                // accumulate up-converted sample
                let expjwt = Complex32::new(0.0, (j as f32) * dphi).exp();
                y1[j] += y_hat * expjwt;
            }
            assert_eq!(n, num_symbols);
        }

        // compare results
        for i in 0..num_samples {
            assert_abs_diff_eq!(y0[i].re, y1[i].re, epsilon = tol);
            assert_abs_diff_eq!(y0[i].im, y1[i].im, epsilon = tol);
        }
    }

    // Test that complex coefficients work (firpfbch_cccf equivalent)
    #[test]
    fn test_firpfbch_cccf_analysis() {
        let tol = 1e-4f32;
        let num_channels = 4;
        let p = 5;
        let num_symbols = 40;

        // derived values
        let num_samples = num_channels * num_symbols;

        // generate complex filter coefficients using m-sequence
        let h_len = p * num_channels;
        let mut h = vec![Complex32::new(0.0, 0.0); h_len];
        let mut ms = MSequence::create_default(6).unwrap();
        for i in 0..h_len {
            h[i] = Complex32::new(
                ms.generate_symbol(2) as f32 - 1.5,
                ms.generate_symbol(2) as f32 - 1.5,
            );
        }

        // create filterbank object with complex coefficients
        let mut q = FirPfbChannelizer::<Complex32, Complex32>::new(
            ChannelizerType::Analyzer, num_channels, p, &h
        ).unwrap();

        // create filter object
        let mut f = FirFilter::<Complex32, Complex32>::new(&h).unwrap();

        // allocate memory for arrays
        let mut y = vec![Complex32::new(0.0, 0.0); num_samples];
        let mut y0 = vec![vec![Complex32::new(0.0, 0.0); num_channels]; num_symbols];
        let mut y1 = vec![vec![Complex32::new(0.0, 0.0); num_channels]; num_symbols];

        // generate input sequence (complex noise)
        let mut ms = MSequence::create_default(7).unwrap();
        for i in 0..num_samples {
            y[i] = Complex32::new(
                0.1 * std::f32::consts::FRAC_1_SQRT_2 * (ms.generate_symbol(2) as f32 - 1.5),
                0.1 * std::f32::consts::FRAC_1_SQRT_2 * (ms.generate_symbol(2) as f32 - 1.5),
            );
        }

        // run analysis filter bank
        for i in 0..num_symbols {
            q.analyzer_execute(&y[i * num_channels..(i + 1) * num_channels], &mut y0[i]).unwrap();
        }

        // run traditional down-converter (inefficient)
        for i in 0..num_channels {
            // reset filter
            f.reset();

            // set center frequency
            let dphi = 2.0 * std::f32::consts::PI * (i as f32) / (num_channels as f32);

            // reset symbol counter
            let mut n = 0;

            for j in 0..num_samples {
                // push down-converted sample into filter
                let expjwt = Complex32::new(0.0, -(j as f32) * dphi).exp();
                f.push(y[j] * expjwt);

                // compute output at the appropriate sample time
                if ((j + 1) % num_channels) == 0 {
                    y1[n][i] = f.execute();
                    n += 1;
                }
            }
            assert_eq!(n, num_symbols);
        }

        // compare results
        for i in 0..num_symbols {
            for j in 0..num_channels {
                assert_abs_diff_eq!(y0[i][j].re, y1[i][j].re, epsilon = tol);
                assert_abs_diff_eq!(y0[i][j].im, y1[i][j].im, epsilon = tol);
            }
        }
    }
}
