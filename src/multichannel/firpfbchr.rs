// firpfbchr : finite impulse response polyphase filterbank channelizer
// with output rate Fs / P (rational rate)

use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use crate::filter;

use num_complex::{Complex32, ComplexFloat};

/// Finite impulse response polyphase filterbank channelizer with rational rate
/// (output rate Fs / P)
#[derive(Clone, Debug)]
pub struct FirPfbChannelizerR<T> {
    num_channels: usize,
    decim_rate: usize,
    m: usize,

    // bank of dotprod objects
    dp: Vec<Vec<f32>>,

    // inverse FFT plan
    ifft: Fft<f32>,
    x: Vec<Complex32>,
    x_out: Vec<Complex32>,

    // window buffer objects
    w: Vec<Window<T>>,
    base_index: usize,
}

impl<T> FirPfbChannelizerR<T>
where
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + Into<Complex32> + From<Complex32>,
    [f32]: DotProd<T, Output = T>,
{
    /// Create rational rate resampling channelizer (firpfbchr) object by
    /// specifying filter coefficients directly
    ///
    /// # Arguments
    ///
    /// * `num_channels` - number of output channels in channelizer
    /// * `decim_rate` - output decimation factor (output rate is 1/P the input)
    /// * `m` - prototype filter semi-length, length=2*M*m
    /// * `h` - prototype filter coefficient array, [size: 2*M*m x 1]
    pub fn new(num_channels: usize, decim_rate: usize, m: usize, h: &[f32]) -> Result<Self> {
        if num_channels < 2 {
            return Err(Error::Config("number of channels must be at least 2".into()));
        }
        if decim_rate < 1 {
            return Err(Error::Config("decimation rate must be at least 1".into()));
        }
        if m < 1 {
            return Err(Error::Config("filter semi-length must be at least 1".into()));
        }
        if h.is_empty() {
            return Err(Error::Config("filter coefficients cannot be null".into()));
        }

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
        let mut w = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            w.push(Window::new(h_sub_len)?);
        }

        let mut q = Self { num_channels, decim_rate, m, dp, ifft, x, x_out, w, base_index: 0 };

        q.reset();
        Ok(q)
    }

    /// Create rational rate resampling channelizer (firpfbchr) object by
    /// specifying filter design parameters for Kaiser prototype
    ///
    /// # Arguments
    ///
    /// * `num_channels` - number of output channels in channelizer
    /// * `decim_rate` - output decimation factor (output rate is 1/P the input)
    /// * `m` - prototype filter semi-length, length=2*M*m
    /// * `as_` - filter stop-band attenuation [dB]
    pub fn new_kaiser(num_channels: usize, decim_rate: usize, m: usize, as_: f32) -> Result<Self> {
        if num_channels < 2 {
            return Err(Error::Config("number of channels must be at least 2".into()));
        }
        if decim_rate < 1 {
            return Err(Error::Config("decimation rate must be at least 1".into()));
        }
        if m < 1 {
            return Err(Error::Config("filter semi-length must be at least 1".into()));
        }
        if as_ <= 0.0 {
            return Err(Error::Config("stop-band suppression out of range".into()));
        }

        // design prototype filter
        let h_len = 2 * num_channels * m + 1;

        // filter cut-off frequency
        let fc = 0.5 / decim_rate as f32;

        // compute filter coefficients (floating point precision)
        let hf = filter::fir_design_kaiser(h_len, fc, as_, 0.0)?;

        // normalize to unit average and scale by number of channels
        let hf_sum: f32 = hf.iter().sum();
        let scale = (decim_rate as f32).sqrt() * num_channels as f32 / hf_sum;
        let h: Vec<f32> = hf.iter().map(|&x| x * scale).collect();

        // create filterbank channelizer object
        Self::new(num_channels, decim_rate, m, &h)
    }

    /// Reset firpfbchr object internals
    pub fn reset(&mut self) {
        for i in 0..self.num_channels {
            self.w[i].reset();
        }
        self.base_index = self.num_channels - 1;
    }

    /// Get number of output channels to channelizer
    pub fn get_num_channels(&self) -> usize {
        self.num_channels
    }

    /// Get decimation rate
    pub fn get_decim_rate(&self) -> usize {
        self.decim_rate
    }

    /// Get semi-length to channelizer filter prototype
    pub fn get_m(&self) -> usize {
        self.m
    }

    /// Push samples into filter bank
    ///
    /// # Arguments
    ///
    /// * `x` - channelizer input, [size: decim_rate x 1]
    pub fn push(&mut self, x: &[T]) {
        // load buffers in blocks of P in the reverse direction
        for i in 0..self.decim_rate {
            // push sample into buffer at filter index
            self.w[self.base_index].push(x[i]);

            // decrement base index, wrapping around
            self.base_index = if self.base_index == 0 { self.num_channels - 1 } else { self.base_index - 1 };
        }
    }

    /// Execute filterbank channelizer
    ///
    /// # Arguments
    ///
    /// * `y` - channelizer output, [size: num_channels x 1]
    pub fn execute(&mut self, y: &mut [T]) {
        // execute filter outputs
        for i in 0..self.num_channels {
            // buffer index
            let buffer_index = (self.base_index + i + 1) % self.num_channels;

            // read buffer at index
            let r = self.w[buffer_index].read();

            // run dot products
            let result: T = self.dp[i].dotprod(r);
            self.x[buffer_index] = result.into();
        }

        // execute IFFT, store result in buffer 'x_out'
        self.ifft.run(&self.x, &mut self.x_out);

        // copy result to output, scale result by 1/num_channels (C transform)
        let g = 1.0 / self.num_channels as f32;
        for i in 0..self.num_channels {
            y[i] = <T as From<Complex32>>::from(self.x_out[i] * g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::spgram::Spgram;
    use crate::framing::MSource;
    use crate::modem::modem::ModulationScheme;
    use crate::utility::test_helpers::{validate_psd_spgramcf, PsdRegion};
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_firpfbchr_crcf_config)]
    fn test_firpfbchr_crcf_config() {
        // design prototype filter
        let h_len = 2 * 64 * 12 + 1;
        let h = filter::fir_design_kaiser(h_len, 0.1, 60.0, 0.0).unwrap();

        // check invalid function calls
        assert!(FirPfbChannelizerR::<Complex32>::new(0, 76, 12, &h).is_err()); // too few channels
        assert!(FirPfbChannelizerR::<Complex32>::new(64, 0, 12, &h).is_err()); // decimation rate too small
        assert!(FirPfbChannelizerR::<Complex32>::new(64, 76, 0, &h).is_err()); // filter delay too small
        assert!(FirPfbChannelizerR::<Complex32>::new(64, 76, 12, &[]).is_err()); // coefficients empty

        // kaiser
        assert!(FirPfbChannelizerR::<Complex32>::new_kaiser(0, 76, 12, 60.0).is_err()); // too few channels
        assert!(FirPfbChannelizerR::<Complex32>::new_kaiser(64, 0, 12, 60.0).is_err()); // decimation rate too small
        assert!(FirPfbChannelizerR::<Complex32>::new_kaiser(64, 76, 0, 60.0).is_err()); // filter delay too small
        assert!(FirPfbChannelizerR::<Complex32>::new_kaiser(64, 76, 12, -1.0).is_err()); // stop-band suppression out of range

        // create proper object and test configurations
        let q = FirPfbChannelizerR::<Complex32>::new_kaiser(64, 76, 12, 60.0).unwrap();
        assert_eq!(q.get_num_channels(), 64);
        assert_eq!(q.get_decim_rate(), 76);
        assert_eq!(q.get_m(), 12);
    }

    #[test]
    #[autotest_annotate(autotest_firpfbchr_crcf)]
    fn test_firpfbchr_crcf() {
        // options
        let m_channels = 16; // number of channels
        let p = 6; // output decimation rate
        let m = 12; // filter semi-length (symbols)
        let num_blocks = 1 << 16; // number of symbols
        let as_ = 60.0f32; // filter stop-band attenuation

        // create filterbank object
        let mut qa =
            FirPfbChannelizerR::<Complex32>::new_kaiser(m_channels, p, m, as_).unwrap();

        // create multi-signal source generator
        let mut gen = MSource::new_default().unwrap();

        // add signals (fc, bw, gain)
        gen.add_noise(0.0, 1.0, -60.0).unwrap(); // wide-band noise
        gen.add_noise(-0.30, 0.10, -20.0).unwrap(); // narrow-band noise
        gen.add_noise(0.08, 0.01, -30.0).unwrap(); // very narrow-band noise
        // modulated data
        gen.add_modem(
            0.1875,              // center frequency
            0.065,               // bandwidth (symbol rate)
            -20.0,               // gain
            ModulationScheme::Qpsk, // modulation scheme
            12,                  // filter semi-length
            0.3,                 // modem parameters
        )
        .unwrap();

        // create spectral periodograms
        let nfft = 2400;
        let mut p0 = Spgram::<Complex32>::default(nfft).unwrap();
        let mut c1 = Spgram::<Complex32>::default(nfft).unwrap();
        let mut c3 = Spgram::<Complex32>::default(nfft).unwrap();

        // run channelizer
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); p];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); m_channels];

        for _ in 0..num_blocks {
            // write samples to buffer
            gen.write_samples(&mut buf_0).unwrap();

            // run analysis filterbank
            qa.push(&buf_0);
            qa.execute(&mut buf_1);

            // push results through periodograms
            p0.write(&buf_0);
            c1.push(buf_1[1]);
            c3.push(buf_1[3]);
        }

        // verify results: full spectrum
        let regions_p0 = [
            // noise floor regions
            PsdRegion { fmin: -0.50, fmax: -0.37, pmin: -65.0, pmax: -55.0, test_lo: true, test_hi: true },
            // narrow-band noise at -0.30
            PsdRegion { fmin: -0.34, fmax: -0.26, pmin: -25.0, pmax: -15.0, test_lo: true, test_hi: true },
            // noise floor
            PsdRegion { fmin: -0.24, fmax: 0.05, pmin: -65.0, pmax: -55.0, test_lo: true, test_hi: true },
            // noise floor
            PsdRegion { fmin: 0.10, fmax: 0.13, pmin: -65.0, pmax: -55.0, test_lo: true, test_hi: true },
            // modulated signal at 0.1875
            PsdRegion { fmin: 0.16, fmax: 0.21, pmin: -25.0, pmax: -15.0, test_lo: true, test_hi: true },
            // noise floor
            PsdRegion { fmin: 0.25, fmax: 0.50, pmin: -65.0, pmax: -55.0, test_lo: true, test_hi: true },
        ];
        assert!(validate_psd_spgramcf(&p0, &regions_p0).unwrap());

        // verify results: channel 1 (sees very narrow-band noise at input 0.08)
        // Channel 1 center is ~0.0625, so signal at 0.08 appears at ~0.08-0.12 in channel
        let regions_c1 = [
            PsdRegion { fmin: -0.47, fmax: 0.05, pmin: -65.0, pmax: -55.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.08, fmax: 0.12, pmin: -35.0, pmax: -25.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.15, fmax: 0.47, pmin: -65.0, pmax: -55.0, test_lo: true, test_hi: true },
        ];
        assert!(validate_psd_spgramcf(&c1, &regions_c1).unwrap());

        // verify results: channel 3 (sees modem signal at input 0.1875)
        // Channel 3 center is ~0.1875, so modem signal appears near DC in channel
        let regions_c3 = [
            PsdRegion { fmin: -0.47, fmax: -0.28, pmin: -65.0, pmax: -55.0, test_lo: true, test_hi: true },
            // Modem signal around DC with bw ~0.065
            PsdRegion { fmin: -0.15, fmax: 0.15, pmin: -25.0, pmax: -15.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.28, fmax: 0.47, pmin: -65.0, pmax: -55.0, test_lo: true, test_hi: true },
        ];
        assert!(validate_psd_spgramcf(&c3, &regions_c3).unwrap());
    }

    #[test]
    fn test_firpfbchr_channel_mapping() {
        // Test that tones at known frequencies map to expected channels
        // Channel k responds to input frequencies around k/M
        let m_channels = 16;
        let p = 6;
        let m = 12;
        let as_ = 60.0f32;

        let mut qa = FirPfbChannelizerR::<Complex32>::new_kaiser(m_channels, p, m, as_).unwrap();

        let num_blocks = 10000;
        let mut channel_power = vec![0.0f32; m_channels];

        // Tone at -0.30 should map to channel ~11 (since -0.30 + 1.0 = 0.70, 0.70 * 16 ≈ 11)
        let tone_freq = -0.30f32;
        let mut phase = 0.0f32;
        let phase_inc = 2.0 * std::f32::consts::PI * tone_freq;

        let mut buf_in = vec![Complex32::new(0.0, 0.0); p];
        let mut buf_out = vec![Complex32::new(0.0, 0.0); m_channels];

        for _ in 0..num_blocks {
            for sample in buf_in.iter_mut() {
                *sample = Complex32::new(phase.cos(), phase.sin());
                phase += phase_inc;
            }
            qa.push(&buf_in);
            qa.execute(&mut buf_out);

            for (i, &s) in buf_out.iter().enumerate() {
                channel_power[i] += s.norm_sqr();
            }
        }

        let expected_channel = (tone_freq.rem_euclid(1.0) * m_channels as f32).round() as usize - 1;

        let max_power = channel_power.iter().cloned().fold(0.0f32, f32::max);
        let max_channel = channel_power.iter().position(|&p| p == max_power).unwrap();

        assert!(
            max_channel == expected_channel,
            "Tone at {} should map to channel {}, got {}",
            tone_freq,
            expected_channel,
            max_channel
        );
    }

    #[test]
    fn test_firpfbchr_channel_center_freqs() {
        // Verify channel center frequencies by sweeping tones
        let m_channels = 16;
        let p = 6;
        let m = 12;
        let as_ = 60.0f32;

        // Expected: channel k responds to freq k/16 for k=0..8, (k-16)/16 for k=8..16
        let expected = [
            (1, 0.0625),   // 1/16
            (3, 0.1875),   // 3/16
            (11, -0.3125), // -5/16
        ];

        for (test_channel, expected_freq) in expected {
            let mut best_freq = 0.0f32;
            let mut best_power = f32::NEG_INFINITY;

            for freq_idx in 0..32 {
                let tone_freq = (freq_idx as f32) / 32.0 - 0.5;

                let mut qa = FirPfbChannelizerR::<Complex32>::new_kaiser(m_channels, p, m, as_).unwrap();

                let num_blocks = 1000;
                let mut channel_power = 0.0f32;

                let mut phase = 0.0f32;
                let phase_inc = 2.0 * std::f32::consts::PI * tone_freq;

                let mut buf_in = vec![Complex32::new(0.0, 0.0); p];
                let mut buf_out = vec![Complex32::new(0.0, 0.0); m_channels];

                for _ in 0..num_blocks {
                    for sample in buf_in.iter_mut() {
                        *sample = Complex32::new(phase.cos(), phase.sin());
                        phase += phase_inc;
                    }
                    qa.push(&buf_in);
                    qa.execute(&mut buf_out);
                    channel_power += buf_out[test_channel].norm_sqr();
                }

                if channel_power > best_power {
                    best_power = channel_power;
                    best_freq = tone_freq;
                }
            }

            // Check that found frequency is close to expected
            assert!(
                (best_freq - expected_freq).abs() < 0.05,
                "Channel {} should respond to freq {}, got {}",
                test_channel,
                expected_freq,
                best_freq
            );
        }
    }
}
