use crate::error::{Error, Result};
use crate::fft::{Fft, Direction};
use crate::buffer::Window;
use crate::math::WindowType;
use crate::math::windows;
use num_complex::Complex32;
use num_traits::Zero;
use std::cmp::max;
use std::ops::Mul;

pub const SPGRAM_PSD_MIN: f32 = 1e-12;

#[derive(Debug, Clone)]
pub struct Spgram<T> {
    // options
    nfft: usize,
    wtype: WindowType,
    window_len: usize,
    delay: usize,
    alpha: f32,
    gamma: f32,
    accumulate: bool,

    buffer: Window<T>,
    buf_time: Vec<Complex32>,
    buf_freq: Vec<Complex32>,
    w: Vec<f32>,
    fft: Fft<f32>,

    // psd accumulation
    psd: Vec<f32>,
    sample_timer: usize,
    num_samples: u64,
    num_samples_total: u64,
    num_transforms: u64,
    num_transforms_total: u64,

    // parameters for display purposes only
    frequency: f32,
    sample_rate: f32,
}

impl<T: Copy + Default + From<f32> + Zero + Mul<Output = T> > Spgram<T> where Complex32: From<T> {
    /// create spgram object
    ///  _nfft       : FFT size
    ///  _wtype      : window type, e.g. LIQUID_WINDOW_HAMMING
    ///  _window_len : window length
    ///  _delay      : delay between transforms, _delay > 0
    pub fn new(nfft: usize, wtype: WindowType, window_len: usize, delay: usize) -> Result<Self> {
        // validate input
        if nfft < 2 {
            return Err(Error::Config("fft size must be at least 2".into()));
        }
        if window_len > nfft {
            return Err(Error::Config("window size cannot exceed fft size".into()));
        }
        if window_len == 0 {
            return Err(Error::Config("window size must be greater than zero".into()));
        }
        if wtype == WindowType::Kaiser && window_len % 2 != 0 {
            return Err(Error::Config("KBD window length must be even".into()));
        }
        if delay == 0 {
            return Err(Error::Config("delay must be greater than 0".into()));
        }

        let mut spgram = Spgram {
            nfft,
            wtype,
            window_len,
            delay,
            frequency: 0.0,
            sample_rate: -1.0,
            alpha: 0.0,
            gamma: 0.0,
            accumulate: false,
            buffer: Window::new(window_len).unwrap(),
            buf_time: vec![Complex32::new(0.0, 0.0); nfft],
            buf_freq: vec![Complex32::new(0.0, 0.0); nfft],
            psd: vec![0.0; nfft],
            fft: Fft::new(nfft, Direction::Forward),
            w: vec![0.0; window_len],
            sample_timer: delay,
            num_samples: 0,
            num_samples_total: 0,
            num_transforms: 0,
            num_transforms_total: 0,
        };

        spgram.set_alpha(-1.0)?;

        // create window
        let mut g = 0.0;
        let beta = 10.0;
        let zeta = 3.0;
        for i in 0..window_len {
            let w = match spgram.wtype {
                WindowType::Hamming => windows::hamming(i, window_len)?,
                WindowType::Hann => windows::hann(i, window_len)?,
                WindowType::BlackmanHarris => windows::blackman_harris(i, window_len)?,
                WindowType::BlackmanHarris7 => windows::blackman_harris7(i, window_len)?,
                WindowType::Kaiser => windows::kaiser(i, window_len, beta)?,
                WindowType::FlatTop => windows::flat_top(i, window_len)?,
                WindowType::Triangular => windows::triangular(i, window_len, window_len)?,
                WindowType::RcosTaper => windows::rcos_taper(i, window_len, window_len / 3)?,
                WindowType::Kbd => windows::kbd(i, window_len, zeta)?,
                _ => return Err(Error::Config("unknown window type".into())),
            };
            spgram.w[i] = w.into();
            g += w * w;
        }

        g = 1.0 / g.sqrt();

        // scale window and copy
        for i in 0..window_len {
            spgram.w[i] = g * spgram.w[i];
        }

        spgram.reset();

        Ok(spgram)
    }

    /// create default spgram object (Kaiser-Bessel window)
    pub fn default(nfft: usize) -> Result<Self> {
        if nfft < 2 {
            return Err(Error::Config("fft size must be at least 2".into()));
        }

        Self::new(nfft, WindowType::Kaiser, nfft / 2, nfft / 4)
    }

    /// clears the internal state of the spgram object, but not
    /// the internal buffer
    pub fn clear(&mut self) {
        for v in &mut self.buf_time {
            *v = Complex32::new(0.0, 0.0);
        }

        self.sample_timer = self.delay;
        self.num_transforms = 0;
        self.num_samples = 0;

        for v in &mut self.psd {
            *v = 0.0;
        }
    }

    /// reset the spgram object to its original state completely
    pub fn reset(&mut self) {
        self.clear();
        self.buffer.reset();
        self.num_samples_total = 0;
        self.num_transforms_total = 0;
    }

    /// set forgetting factor
    pub fn set_alpha(&mut self, _alpha: f32) -> Result<()> {
        if _alpha != -1.0 && (_alpha < 0.0 || _alpha > 1.0) {
            return Err(Error::Config("alpha must be in {-1,[0,1]}".into()));
        }

        self.accumulate = _alpha == -1.0;

        if self.accumulate {
            self.alpha = 1.0;
            self.gamma = 1.0;
        } else {
            self.alpha = _alpha;
            self.gamma = 1.0 - self.alpha;
        }

        Ok(())
    }

    /// set center frequency
    pub fn set_freq(&mut self, _freq: f32) {
        self.frequency = _freq;
    }

    /// set sample rate
    pub fn set_rate(&mut self, _rate: f32) -> Result<()> {
        if _rate <= 0.0 {
            return Err(Error::Config("sample rate must be greater than zero".into()));
        }
        self.sample_rate = _rate;
        Ok(())
    }

    /// get FFT size
    pub fn get_nfft(&self) -> usize {
        self.nfft
    }

    /// get window length
    pub fn get_window_len(&self) -> usize {
        self.window_len
    }

    /// get delay between transforms
    pub fn get_delay(&self) -> usize {
        self.delay
    }

    /// get window type used for spectral estimation
    pub fn get_wtype(&self) -> WindowType {
        self.wtype
    }

    /// get number of samples processed since reset
    pub fn get_num_samples(&self) -> u64 {
        self.num_samples
    }

    /// get number of samples processed since start
    pub fn get_num_samples_total(&self) -> u64 {
        self.num_samples_total
    }

    /// get number of transforms processed since reset
    pub fn get_num_transforms(&self) -> u64 {
        self.num_transforms
    }

    /// get number of transforms processed since start
    pub fn get_num_transforms_total(&self) -> u64 {
        self.num_transforms_total
    }

    /// get forgetting factor (filter bandwidth)
    pub fn get_alpha(&self) -> f32 {
        self.alpha
    }

    /// push a single sample into the spgram object
    pub fn push(&mut self, _x: T) {
        // push sample into internal buffer
        self.buffer.push(_x);

        // update counters
        self.num_samples += 1;
        self.num_samples_total += 1;
        self.sample_timer -= 1;

        // check if buffer is full
        if self.sample_timer == 0 {
            self.sample_timer = self.delay;
            self.step();
        }
    }

    /// write a block of samples to the spgram object
    pub fn write(&mut self, _x: &[T]) {
        for x in _x {
            self.push(*x);
        }
    }

    /// compute spectral periodogram output from current buffer contents
    fn step(&mut self) {
        // read buffer, copy to fft input array, applying window
        let rc = self.buffer.read();
        for i in 0..self.window_len {
            // first do the operation in T, then convert to Complex32
            self.buf_time[i] = (rc[i] * self.w[i].into()).into();
        }

        // execute fft on _q->buf_time and store result in _q->buf_freq
        self.fft.run(&mut self.buf_time, &mut self.buf_freq);

        // accumulate output
        let nfft = self.nfft;
        let alpha = self.alpha;
        let gamma = self.gamma;
        for i in 0..nfft {
            let mag_sq = (self.buf_freq[i] * self.buf_freq[i].conj()).re;
            if self.num_transforms == 0 {
                self.psd[i] = mag_sq;
            } else {
                self.psd[i] = gamma * self.psd[i] + alpha * mag_sq;
            }
        }

        // update counters
        self.num_transforms += 1;
        self.num_transforms_total += 1;
    }

    /// compute spectral periodogram output (fft-shifted values, linear)
    /// from current buffer contents
    pub fn get_psd_mag(&self) -> Vec<f32> {
        let mut psd_mag = vec![0.0; self.nfft];
        self.read_psd_mag(&mut psd_mag).unwrap();
        psd_mag
    }

    pub fn read_psd_mag(&self, psd_mag: &mut [f32]) -> Result<()> {
        if psd_mag.len() != self.nfft {
            return Err(Error::Value("psd_mag must be the same length as the fft size".into()));
        }

        let nfft_2 = self.nfft / 2; 
        let scale = if self.accumulate {
            1.0 / max(1, self.num_transforms) as f32
        } else {
            // 0.0
            // 1.0 / max(1, self.num_transforms) as f32
            1.0
        };
        for i in 0..self.nfft {
            let k = (i + nfft_2) % self.nfft;
            psd_mag[i] = self.psd[k].max(SPGRAM_PSD_MIN) * scale;
        }

        Ok(())
    }

    /// compute spectral periodogram output (fft-shifted values
    /// in dB) from current buffer contents
    pub fn get_psd(&self) -> Vec<f32> {
        let mut psd = vec![0.0; self.nfft];
        self.read_psd(&mut psd).unwrap();
        psd
    }

    pub fn read_psd(&self, psd: &mut [f32]) -> Result<()> {
        self.read_psd_mag(psd)?;
        for i in 0..self.nfft {
            psd[i] = 10.0 * psd[i].log10();
        }
        Ok(())
    }

    /// estimate spectrum on input signal
    pub fn estimate_psd(nfft: usize, x: &[T]) -> Result<Vec<f32>> {
        let mut q = Self::default(nfft)?;

        q.write(x);

        if q.num_transforms == 0 {
            q.step();
        }

        Ok(q.get_psd())            
    }
}

#[cfg(test)]
 mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;
    use crate::random::randnf;
    
    fn testbench_spgramcf_noise(nfft: usize, wlen: usize, delay: usize, wtype: WindowType, noise_floor: f32) {
        let num_samples = 2000 * nfft;  // number of samples to generate
        let nstd = 10f32.powf(noise_floor / 20.0); // noise std. dev.
        let tol = 0.5f32; // error tolerance [dB]
    
        // create spectral periodogram
        let mut q = if wlen == 0 || delay == 0 || wtype == WindowType::Unknown {
            Spgram::<Complex32>::default(nfft).unwrap()
        } else {
            Spgram::<Complex32>::new(nfft, wtype, wlen, delay).unwrap()
        };
    
        for _ in 0..num_samples {
            let noise = Complex32::new(randnf(), randnf()) * nstd * (0.5f32).sqrt();
            q.push(noise);
        }
    
        // verify number of samples processed
        assert_eq!(q.get_num_samples(), num_samples as u64);
        assert_eq!(q.get_num_samples_total(), num_samples as u64);
    
        // compute power spectral density output
        let psd = q.get_psd();
    
        // verify result
        for p in psd.iter() {
            assert_relative_eq!(*p, noise_floor, epsilon = tol);
        }
    }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_440)]
    fn test_spgramcf_noise_440() { testbench_spgramcf_noise(440, 0, 0, WindowType::Unknown, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_1024)]
    fn test_spgramcf_noise_1024() { testbench_spgramcf_noise(1024, 0, 0, WindowType::Unknown, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_1200)]
    fn test_spgramcf_noise_1200() { testbench_spgramcf_noise(1200, 0, 0, WindowType::Unknown, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_custom_0)]
    fn test_spgramcf_noise_custom_0() { testbench_spgramcf_noise(400, 400, 100, WindowType::Hamming, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_custom_1)]
    fn test_spgramcf_noise_custom_1() { testbench_spgramcf_noise(512, 200, 120, WindowType::Hamming, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_custom_2)]
    fn test_spgramcf_noise_custom_2() { testbench_spgramcf_noise(640, 100, 10, WindowType::Hamming, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_custom_3)]
    fn test_spgramcf_noise_custom_3() { testbench_spgramcf_noise(960, 83, 17, WindowType::Hamming, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_hamming)]
    fn test_spgramcf_noise_hamming() { testbench_spgramcf_noise(800, 0, 0, WindowType::Hamming, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_hann)]
    fn test_spgramcf_noise_hann() { testbench_spgramcf_noise(800, 0, 0, WindowType::Hann, -80.0); } 
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_blackmanharris)]
    fn test_spgramcf_noise_blackmanharris() { testbench_spgramcf_noise(800, 0, 0, WindowType::BlackmanHarris, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_blackmanharris7)]
    fn test_spgramcf_noise_blackmanharris7() { testbench_spgramcf_noise(800, 0, 0, WindowType::BlackmanHarris7, -80.0); }
    
    #[test] 
    #[autotest_annotate(autotest_spgramcf_noise_kaiser)]
    fn test_spgramcf_noise_kaiser() { testbench_spgramcf_noise(800, 0, 0, WindowType::Kaiser, -80.0); }
    
    #[test] 
    #[autotest_annotate(autotest_spgramcf_noise_flattop)]
    fn test_spgramcf_noise_flattop() { testbench_spgramcf_noise(800, 0, 0, WindowType::FlatTop, -80.0); }
    
    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_triangular)]
    fn test_spgramcf_noise_triangular() { testbench_spgramcf_noise(800, 0, 0, WindowType::Triangular, -80.0); }
    
    #[test] 
    #[autotest_annotate(autotest_spgramcf_noise_rcostaper)]
    fn test_spgramcf_noise_rcostaper() { testbench_spgramcf_noise(800, 0, 0, WindowType::RcosTaper, -80.0); }   

    #[test]
    #[autotest_annotate(autotest_spgramcf_noise_kbd)]
    fn test_spgramcf_noise_kbd() { testbench_spgramcf_noise(800, 0, 0, WindowType::Kbd, -80.0); }   

    fn testbench_spgramcf_signal(nfft: usize, wtype: WindowType, fc: f32, snr_db: f32) {
        use crate::filter::FirFilterShape;
        use crate::framing::symstreamr::SymStreamR;
        use crate::modem::modem::ModulationScheme;
        use crate::utility::test_helpers::{PsdRegion, validate_psd_spectrum};
        use crate::nco::{Osc, OscScheme};
        use std::f32::consts::{PI, FRAC_1_SQRT_2};

        let bw = 0.25f32; // signal bandwidth (relative)
        let m = 25;
        let beta = 0.2f32;
        let n0 = -80.0f32;
        let tol = 0.5f32;
    
        // create objects
        let mut q = Spgram::<Complex32>::new(nfft, wtype, nfft/2, nfft/4).unwrap();
        let mut gen = SymStreamR::new_linear(
            FirFilterShape::Kaiser,
            bw,
            m,
            beta,
            ModulationScheme::Qpsk
        ).unwrap();
        let mut mixer = Osc::new(OscScheme::Vco);
    
        // set parameters
        let nstd = 10f32.powf(n0 / 20.0); // noise std. dev.   
        gen.set_gain(10f32.powf((n0 + snr_db + 10.0 * bw.log10()) / 20.0));
        mixer.set_frequency(2.0 * PI * fc);
    
        // generate samples and push through spgram object
        let buf_len = 256;
        let mut num_samples = 0;
        let mut sample_buf = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut mixed_buf = vec![Complex32::new(0.0, 0.0); buf_len];
    
        while num_samples < 2000 * nfft {
            // generate block of samples
            gen.write_samples(&mut sample_buf).unwrap();

            // mix to desired frequency and add noise
            mixer.mix_block_up(&sample_buf, &mut mixed_buf).unwrap();
            for x in mixed_buf.iter_mut() {
                *x += Complex32::new(randnf(), randnf()) * nstd * FRAC_1_SQRT_2;
            }

            // run samples through the spgram object
            q.write(&mixed_buf);
            num_samples += buf_len;
        }
    
        // verify result
        let psd = q.get_psd();
        let sn = 10.0 * (10f32.powf((snr_db + n0) / 10.0) + 10f32.powf(n0 / 10.0)).log10();
    
        let regions = [
            PsdRegion { fmin: -0.5,          fmax: fc - 0.6 * bw, pmin: n0 - tol, pmax: n0 + tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: fc - 0.4 * bw, fmax: fc + 0.4 * bw, pmin: sn - tol, pmax: sn + tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: fc + 0.6 * bw, fmax: 0.5,           pmin: n0 - tol, pmax: n0 + tol, test_lo: true, test_hi: true },
        ];
    
        assert!(validate_psd_spectrum(&psd, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_spgramcf_signal_00)]
    fn test_spgramcf_signal_00() { testbench_spgramcf_signal(800, WindowType::Hamming, 0.0, 30.0); }

    #[test]
    #[autotest_annotate(autotest_spgramcf_signal_01)]
    fn test_spgramcf_signal_01() { testbench_spgramcf_signal(800, WindowType::Hamming, 0.2, 10.0); }

    #[test]
    #[autotest_annotate(autotest_spgramcf_signal_02)]
    fn test_spgramcf_signal_02() { testbench_spgramcf_signal(800, WindowType::Hann, 0.2, 10.0); }

    #[test]
    #[autotest_annotate(autotest_spgramcf_signal_03)]
    fn test_spgramcf_signal_03() { testbench_spgramcf_signal(400, WindowType::Kaiser, -0.3, 40.0); }

    #[test]
    #[autotest_annotate(autotest_spgramcf_signal_04)]
    fn test_spgramcf_signal_04() { testbench_spgramcf_signal(640, WindowType::Hamming, -0.2, 0.0); }

    #[test]
    #[autotest_annotate(autotest_spgramcf_signal_05)]
    fn test_spgramcf_signal_05() { testbench_spgramcf_signal(640, WindowType::Hamming, 0.1, -3.0); }

    #[test]
    #[autotest_annotate(autotest_spgramcf_counters)]
    fn test_spgramcf_counters() {
        // create spectral periodogram with specific parameters
        let nfft = 1200;
        let wlen = 400;
        let delay = 200;
        let wtype = WindowType::Hamming;
        let alpha = 0.0123456f32;
        let mut q = Spgram::<Complex32>::new(nfft, wtype, wlen, delay).unwrap();

        // check setting bandwidth
        assert!(q.set_alpha(0.1).is_ok());
        assert_relative_eq!(q.get_alpha(), 0.1, epsilon = 1e-6);
        assert!(q.set_alpha(-7.0).is_err());
        assert_relative_eq!(q.get_alpha(), 0.1, epsilon = 1e-6);
        assert!(q.set_alpha(alpha).is_ok());
        assert_relative_eq!(q.get_alpha(), alpha, epsilon = 1e-6);

        // check parameters
        assert_eq!(q.get_nfft(), nfft);
        assert_eq!(q.get_window_len(), wlen);
        assert_eq!(q.get_delay(), delay);
        assert_relative_eq!(q.get_alpha(), alpha, epsilon = 1e-6);

        let block_len = 1117;
        let num_blocks = 1123;
        let num_samples = block_len * num_blocks;
        let num_transforms = num_samples / delay;

        for _ in 0..num_samples {
            q.push(Complex32::new(randnf(), randnf()));
        }

        // verify number of samples and transforms processed
        assert_eq!(q.get_num_samples(), num_samples as u64);
        assert_eq!(q.get_num_samples_total(), num_samples as u64);
        assert_eq!(q.get_num_transforms(), num_transforms as u64);
        assert_eq!(q.get_num_transforms_total(), num_transforms as u64);

        // clear object and run in blocks
        q.clear();
        let mut block = vec![Complex32::new(0.0, 0.0); block_len];
        for x in &mut block {
            *x = Complex32::new(randnf(), randnf());
        }
        for _ in 0..num_blocks {
            q.write(&block);
        }

        // re-verify number of samples and transforms processed
        assert_eq!(q.get_num_samples(), num_samples as u64);
        assert_eq!(q.get_num_samples_total(), (num_samples * 2) as u64);
        assert_eq!(q.get_num_transforms(), num_transforms as u64);
        assert_eq!(q.get_num_transforms_total(), (num_transforms * 2) as u64);

        // reset object and ensure counters are zero
        q.reset();
        assert_eq!(q.get_num_samples(), 0);
        assert_eq!(q.get_num_samples_total(), 0);
        assert_eq!(q.get_num_transforms(), 0);
        assert_eq!(q.get_num_transforms_total(), 0);
    }

    #[test]
    #[autotest_annotate(autotest_spgramcf_invalid_config)]
    fn test_spgramcf_invalid_config() {
        // Test invalid configurations
        assert!(Spgram::<Complex32>::new(0, WindowType::Hamming, 100, 100).is_err()); // nfft too small
        assert!(Spgram::<Complex32>::new(1, WindowType::Hamming, 100, 100).is_err()); // nfft too small
        assert!(Spgram::<Complex32>::new(2, WindowType::Hamming, 100, 100).is_err()); // window length too large
        assert!(Spgram::<Complex32>::new(400, WindowType::Hamming, 0, 200).is_err()); // window length too small
        assert!(Spgram::<Complex32>::new(400, WindowType::Unknown, 0, 200).is_err()); // invalid window type
        // assert!(Spgram::<Complex32>::new(400, WindowType::NumFunctions, 200, 200).is_err()); // invalid window type (can't do in rust)
        assert!(Spgram::<Complex32>::new(400, WindowType::Kbd, 201, 200).is_err()); // KBD must be even
        assert!(Spgram::<Complex32>::new(400, WindowType::Hamming, 200, 0).is_err()); // delay too small

        assert!(Spgram::<Complex32>::default(0).is_err()); // nfft too small
        assert!(Spgram::<Complex32>::default(1).is_err()); // nfft too small

        let mut q = Spgram::<Complex32>::default(540).unwrap();
        assert!(q.set_rate(-10e6).is_err());
    }

    #[test]
    #[autotest_annotate(autotest_spgramcf_standalone)]
    fn test_spgramcf_standalone() {
        let nfft = 1200;
        let num_samples = 20 * nfft;
        let noise_floor = -20.0f32;
        let nstd = 10f32.powf(noise_floor / 20.0); // noise std. dev.

        let mut buf = vec![Complex32::new(0.0, 0.0); num_samples];
        for i in 0..num_samples {
            buf[i] = 0.1 + Complex32::new(randnf(), randnf()) * nstd * (0.5f32).sqrt();
        }

        let psd = Spgram::<Complex32>::estimate_psd(nfft, &buf).unwrap();

        // check mask
        for i in 0..nfft {
            let mask_lo = if i == nfft / 2 { 2.0f32 } else { noise_floor - 3.0f32 };
            let mask_hi = if i > nfft / 2 - 10 && i < nfft / 2 + 10 { 8.0f32 } else { noise_floor + 3.0f32 };
            assert!(psd[i] > mask_lo);
            assert!(psd[i] < mask_hi);
        }
    }

    #[test]
    #[autotest_annotate(autotest_spgramcf_short)]
    fn test_spgramcf_short() {
        let nfft = 1200;
        let num_samples = 200;
        let noise_floor = -20.0f32;
        let nstd = 10f32.powf(noise_floor / 20.0); // noise std. dev.

        let mut buf = vec![Complex32::new(0.0, 0.0); num_samples];
        for i in 0..num_samples {
            buf[i] = 1.0 + Complex32::new(randnf(), randnf()) * nstd * (0.5f32).sqrt();
        }

        let psd = Spgram::<Complex32>::estimate_psd(nfft, &buf).unwrap();

        // use a very loose upper mask as we have only computed a few hundred samples
        for i in 0..nfft {
            let f = (i as f32) / (nfft as f32) - 0.5f32;
            let mask_hi = if f.abs() < 0.2f32 { 15.0f32 - 30.0f32 * f.abs() / 0.2f32 } else { -15.0f32 };
            assert!(psd[i] < mask_hi);
        }
        // consider lower mask only for DC term
        let mask_lo = 0.0f32;
        let nfft_2 = nfft / 2;
        assert!(psd[nfft_2] > mask_lo);
    }
}