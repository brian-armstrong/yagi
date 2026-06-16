// Frame detector using FFT-based cross-correlation

use num_complex::Complex32;
use std::f32::consts::PI;

use crate::error::{Error, Result};
use crate::fft::{Fft, Direction};
use crate::filter::{FirInterpolationFilter, FirFilterShape};
use crate::math::nextpow2;
use crate::modem::gmskmod::GmskMod;
use crate::modem::cpfskmod::{Cpfskmod, CpfskFilterType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QdetectorState {
    Seek,
    Align,
}

/// Frame detector using FFT-based cross-correlation
#[derive(Clone)]
pub struct Qdetector {
    s_len: usize,                 // template (time) length
    s: Vec<Complex32>,            // template (time)
    s_conj_freq: Vec<Complex32>,  // template conjugate (freq), [size: nfft x 1]
    s2_sum: f32,                  // sum{ |s|^2 }

    buf_time_0: Vec<Complex32>,   // time-domain buffer (FFT input)
    buf_freq_0: Vec<Complex32>,   // frequency-domain buffer (FFT output)
    buf_freq_1: Vec<Complex32>,   // frequency-domain buffer (IFFT input)
    buf_time_1: Vec<Complex32>,   // time-domain buffer (IFFT output)
    nfft: usize,                  // fft size
    fft: Fft<f32>,                // FFT object
    ifft: Fft<f32>,               // IFFT object

    counter: usize,               // sample counter for determining when to compute FFTs
    threshold: f32,               // detection threshold
    dphi_max: f32,                // carrier offset search range (radians/sample)
    range: i32,                   // carrier offset search range (subcarriers)

    x2_sum_0: f32,                // sum{ |x|^2 } of first half of buffer
    x2_sum_1: f32,                // sum{ |x|^2 } of second half of buffer

    rxy: f32,                     // peak correlation output
    offset: i32,                  // FFT offset index for peak correlation (coarse carrier estimate)
    tau_hat: f32,                 // timing offset estimate
    gamma_hat: f32,               // signal level estimate (channel gain)
    dphi_hat: f32,                // carrier frequency offset estimate
    phi_hat: f32,                 // carrier phase offset estimate

    state: QdetectorState,        // execution state
    frame_detected: bool,         // frame detected flag
}

impl Qdetector {
    /// Create detector with generic sequence
    ///
    /// # Arguments
    ///
    /// * `s` - sample sequence
    pub fn new(s: &[Complex32]) -> Result<Self> {
        if s.is_empty() {
            return Err(Error::Config("sequence length cannot be zero".into()));
        }

        let s_len = s.len();

        // compute sum{ |s|^2 }
        let s2_sum: f32 = s.iter().map(|x| x.norm_sqr()).sum();

        // prepare transforms
        let nfft = 1 << nextpow2(2 * s_len as u32)?;

        let fft = Fft::new(nfft, Direction::Forward);
        let ifft = Fft::new(nfft, Direction::Backward);

        // create frequency-domain template
        let mut buf_time_0 = vec![Complex32::new(0.0, 0.0); nfft];
        let mut buf_freq_0 = vec![Complex32::new(0.0, 0.0); nfft];
        buf_time_0[..s_len].copy_from_slice(s);
        fft.run(&buf_time_0, &mut buf_freq_0);

        // store conjugate of S for cross-correlation
        let s_conj_freq: Vec<Complex32> = buf_freq_0.iter().map(|x| x.conj()).collect();

        // reset time buffer
        buf_time_0.fill(Complex32::new(0.0, 0.0));

        let mut q = Self {
            s_len,
            s: s.to_vec(),
            s_conj_freq,
            s2_sum,
            buf_time_0,
            buf_freq_0,
            buf_freq_1: vec![Complex32::new(0.0, 0.0); nfft],
            buf_time_1: vec![Complex32::new(0.0, 0.0); nfft],
            nfft,
            fft,
            ifft,
            counter: nfft / 2,
            threshold: 0.5,
            dphi_max: 0.3,
            range: 0,
            x2_sum_0: 0.0,
            x2_sum_1: 0.0,
            rxy: 0.0,
            offset: 0,
            tau_hat: 0.0,
            gamma_hat: 0.0,
            dphi_hat: 0.0,
            phi_hat: 0.0,
            state: QdetectorState::Seek,
            frame_detected: false,
        };

        q.set_range(0.3)?;
        Ok(q)
    }

    /// Create detector from sequence of symbols using internal linear interpolator
    ///
    /// # Arguments
    ///
    /// * `sequence` - symbol sequence
    /// * `ftype` - filter prototype
    /// * `k` - samples/symbol
    /// * `m` - filter delay
    /// * `beta` - excess bandwidth factor
    pub fn new_linear(
        sequence: &[Complex32],
        ftype: FirFilterShape,
        k: usize,
        m: usize,
        beta: f32,
    ) -> Result<Self> {
        if sequence.is_empty() {
            return Err(Error::Config("sequence length cannot be zero".into()));
        }
        if k < 2 || k > 80 {
            return Err(Error::Config("samples per symbol must be in [2,80]".into()));
        }
        if m < 1 || m > 100 {
            return Err(Error::Config("filter delay must be in [1,100]".into()));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(Error::Config("excess bandwidth factor must be in [0,1]".into()));
        }

        let sequence_len = sequence.len();
        let s_len = k * (sequence_len + 2 * m);
        let mut s = vec![Complex32::new(0.0, 0.0); s_len];

        let mut interp = FirInterpolationFilter::<Complex32, f32>::new_prototype(ftype, k, m, beta, 0.0)?;

        for i in 0..(sequence_len + 2 * m) {
            let sym = if i < sequence_len { sequence[i] } else { Complex32::new(0.0, 0.0) };
            interp.execute(sym, &mut s[k * i..k * (i + 1)])?;
        }

        Self::new(&s)
    }

    /// Create detector from sequence of symbols using GMSK modulation
    ///
    /// # Arguments
    ///
    /// * `sequence` - bit sequence
    /// * `k` - samples/symbol
    /// * `m` - filter delay
    /// * `beta` - excess bandwidth factor
    pub fn new_gmsk(
        sequence: &[u8],
        k: usize,
        m: usize,
        beta: f32,
    ) -> Result<Self> {
        if sequence.is_empty() {
            return Err(Error::Config("sequence length cannot be zero".into()));
        }
        if k < 2 || k > 80 {
            return Err(Error::Config("samples per symbol must be in [2,80]".into()));
        }
        if m < 1 || m > 100 {
            return Err(Error::Config("filter delay must be in [1,100]".into()));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(Error::Config("excess bandwidth factor must be in [0,1]".into()));
        }

        let sequence_len = sequence.len();
        let s_len = k * (sequence_len + 2 * m);
        let mut s = vec![Complex32::new(0.0, 0.0); s_len];

        let mut modulator = GmskMod::new(k, m, beta)?;

        for i in 0..(sequence_len + 2 * m) {
            let bit = if i < sequence_len { sequence[i] } else { 0 };
            modulator.modulate(bit, &mut s[k * i..k * (i + 1)])?;
        }

        Self::new(&s)
    }

    /// Create detector from sequence of CP-FSK symbols
    ///
    /// # Arguments
    ///
    /// * `sequence` - symbol sequence
    /// * `bps` - bits per symbol
    /// * `h` - modulation index
    /// * `k` - samples/symbol
    /// * `m` - filter delay
    /// * `beta` - filter bandwidth parameter
    /// * `filter_type` - filter type
    pub fn new_cpfsk(
        sequence: &[u8],
        bps: usize,
        h: f32,
        k: usize,
        m: usize,
        beta: f32,
        filter_type: CpfskFilterType,
    ) -> Result<Self> {
        if sequence.is_empty() {
            return Err(Error::Config("sequence length cannot be zero".into()));
        }
        if k < 2 || k > 80 {
            return Err(Error::Config("samples per symbol must be in [2,80]".into()));
        }
        if m < 1 || m > 100 {
            return Err(Error::Config("filter delay must be in [1,100]".into()));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(Error::Config("excess bandwidth factor must be in [0,1]".into()));
        }

        let sequence_len = sequence.len();
        let s_len = k * (sequence_len + 2 * m);
        let mut s = vec![Complex32::new(0.0, 0.0); s_len];

        let mut modulator = Cpfskmod::new(bps, h, k, m, beta, filter_type)?;

        for i in 0..(sequence_len + 2 * m) {
            let sym = if i < sequence_len { sequence[i] as usize } else { 0 };
            modulator.modulate(sym, &mut s[k * i..k * (i + 1)])?;
        }

        Self::new(&s)
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.counter = self.nfft / 2;
        self.x2_sum_0 = 0.0;
        self.x2_sum_1 = 0.0;
        self.state = QdetectorState::Seek;
        self.frame_detected = false;
        self.buf_time_0.fill(Complex32::new(0.0, 0.0));
    }

    /// Get detection threshold
    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }

    /// Set detection threshold (should be between 0 and 1, good starting point is 0.5)
    pub fn set_threshold(&mut self, threshold: f32) -> Result<()> {
        if threshold <= 0.0 || threshold > 2.0 {
            return Err(Error::Config(format!("threshold ({}) out of range", threshold)));
        }
        self.threshold = threshold;
        Ok(())
    }

    /// Get carrier offset search range
    pub fn get_range(&self) -> f32 {
        self.dphi_max
    }

    /// Set carrier offset search range
    pub fn set_range(&mut self, dphi_max: f32) -> Result<()> {
        if dphi_max < 0.0 || dphi_max > 0.5 {
            return Err(Error::Config(format!("carrier offset search range ({}) out of range", dphi_max)));
        }
        self.dphi_max = dphi_max;
        self.range = (dphi_max * self.nfft as f32 / (2.0 * PI)) as i32;
        if self.range < 0 {
            self.range = 0;
        }
        Ok(())
    }

    /// Get sequence length
    pub fn get_seq_len(&self) -> usize {
        self.s_len
    }

    /// Get pointer to sequence
    pub fn get_sequence(&self) -> &[Complex32] {
        &self.s
    }

    /// Get buffer length
    pub fn get_buf_len(&self) -> usize {
        self.nfft
    }

    /// Get correlator output
    pub fn get_rxy(&self) -> f32 {
        self.rxy
    }

    /// Get fractional timing offset estimate
    pub fn get_tau(&self) -> f32 {
        self.tau_hat
    }

    /// Get channel gain
    pub fn get_gamma(&self) -> f32 {
        self.gamma_hat
    }

    /// Get carrier frequency offset estimate
    pub fn get_dphi(&self) -> f32 {
        self.dphi_hat
    }

    /// Get carrier phase offset estimate
    pub fn get_phi(&self) -> f32 {
        self.phi_hat
    }

    /// Execute detector on single sample
    ///
    /// Returns Some(&[Complex32]) when frame is detected, pointing to the aligned buffer
    pub fn execute(&mut self, x: Complex32) -> Option<&[Complex32]> {
        match self.state {
            QdetectorState::Seek => self.execute_seek(x),
            QdetectorState::Align => self.execute_align(x),
        }

        if self.frame_detected {
            self.frame_detected = false;
            Some(&self.buf_time_1)
        } else {
            None
        }
    }

    /// Seek signal (initial detection)
    fn execute_seek(&mut self, x: Complex32) {
        // write sample to buffer and increment counter
        self.buf_time_0[self.counter] = x;
        self.counter += 1;

        // accumulate signal magnitude
        self.x2_sum_1 += x.norm_sqr();

        if self.counter < self.nfft {
            return;
        }

        // reset counter (last half of time buffer)
        self.counter = self.nfft / 2;

        // run forward transform
        self.fft.run(&self.buf_time_0, &mut self.buf_freq_0);

        // compute scaling factor
        let g0 = if self.x2_sum_0 == 0.0 {
            self.x2_sum_1.sqrt() * ((self.s_len as f32) / (self.nfft as f32 / 2.0)).sqrt()
        } else {
            (self.x2_sum_0 + self.x2_sum_1).sqrt() * ((self.s_len as f32) / (self.nfft as f32)).sqrt()
        };

        if g0 < 1e-10 {
            // copy last half to front
            let (first_half, second_half) = self.buf_time_0.split_at_mut(self.nfft / 2);
            first_half.copy_from_slice(second_half);

            // swap accumulated signal levels
            self.x2_sum_0 = self.x2_sum_1;
            self.x2_sum_1 = 0.0;
            return;
        }

        let g = 1.0 / (self.nfft as f32 * g0 * self.s2_sum.sqrt());

        // sweep over carrier frequency offset range
        let mut rxy_peak: f32 = 0.0;
        let mut rxy_index: usize = 0;
        let mut rxy_offset: i32 = 0;

        for offset in -self.range..=self.range {
            // cross-multiply, aligning appropriately
            for i in 0..self.nfft {
                let j = ((i as i32 + self.nfft as i32 - offset) as usize) % self.nfft;
                self.buf_freq_1[i] = self.buf_freq_0[i] * self.s_conj_freq[j];
            }

            // run inverse transform
            self.ifft.run(&self.buf_freq_1, &mut self.buf_time_1);

            // scale output appropriately
            for sample in self.buf_time_1.iter_mut() {
                *sample *= g;
            }

            // search for peak
            for (i, sample) in self.buf_time_1.iter().enumerate() {
                let rxy_abs = sample.norm();
                if rxy_abs > rxy_peak {
                    rxy_peak = rxy_abs;
                    rxy_index = i;
                    rxy_offset = offset;
                }
            }
        }

        if rxy_peak > self.threshold && rxy_index < self.nfft - self.s_len {
            // update state, reset counter, copy buffer appropriately
            self.state = QdetectorState::Align;
            self.offset = rxy_offset;
            self.rxy = rxy_peak;

            // copy last part of fft input buffer to front
            let remaining = self.nfft - rxy_index;
            for i in 0..remaining {
                self.buf_time_0[i] = self.buf_time_0[rxy_index + i];
            }
            self.counter = remaining;
            return;
        }

        // copy last half of fft input buffer to front
        let (first_half, second_half) = self.buf_time_0.split_at_mut(self.nfft / 2);
        first_half.copy_from_slice(second_half);

        // swap accumulated signal levels
        self.x2_sum_0 = self.x2_sum_1;
        self.x2_sum_1 = 0.0;
    }

    /// Align signal in time, compute offset estimates
    fn execute_align(&mut self, x: Complex32) {
        // write sample to buffer and increment counter
        self.buf_time_0[self.counter] = x;
        self.counter += 1;

        if self.counter < self.nfft {
            return;
        }

        // estimate timing offset
        self.fft.run(&self.buf_time_0, &mut self.buf_freq_0);

        // cross-multiply frequency-domain components
        for i in 0..self.nfft {
            let j = ((i as i32 + self.nfft as i32 - self.offset) as usize) % self.nfft;
            self.buf_freq_1[i] = self.buf_freq_0[i] * self.s_conj_freq[j];
        }

        self.ifft.run(&self.buf_freq_1, &mut self.buf_time_1);

        // time aligned to index 0
        // NOTE: taking the sqrt removes bias in the timing estimate
        let yneg = self.buf_time_1[self.nfft - 1].norm().sqrt();
        let y0 = self.buf_time_1[0].norm().sqrt();
        let ypos = self.buf_time_1[1].norm().sqrt();

        // compute timing offset estimate from quadratic polynomial fit
        let a = 0.5 * (ypos + yneg) - y0;
        let b = 0.5 * (ypos - yneg);
        let c = y0;
        self.tau_hat = -b / (2.0 * a);
        let g_hat = a * self.tau_hat * self.tau_hat + b * self.tau_hat + c;
        self.gamma_hat = g_hat * g_hat / (self.nfft as f32 * self.s2_sum);

        // copy buffer to preserve data integrity
        self.buf_time_1.copy_from_slice(&self.buf_time_0);

        // estimate carrier frequency offset
        for i in 0..self.nfft {
            self.buf_time_0[i] *= if i < self.s_len { self.s[i].conj() } else { Complex32::new(0.0, 0.0) };
        }
        self.fft.run(&self.buf_time_0, &mut self.buf_freq_0);

        // search for peak
        let mut v0: f32 = 0.0;
        let mut i0: usize = 0;
        for (i, sample) in self.buf_freq_0.iter().enumerate() {
            let v_abs = sample.norm();
            if v_abs > v0 {
                v0 = v_abs;
                i0 = i;
            }
        }

        // interpolate using quadratic polynomial for carrier frequency estimate
        let ineg = (i0 + self.nfft - 1) % self.nfft;
        let ipos = (i0 + 1) % self.nfft;
        let vneg = self.buf_freq_0[ineg].norm();
        let vpos = self.buf_freq_0[ipos].norm();
        let a = 0.5 * (vpos + vneg) - v0;
        let b = 0.5 * (vpos - vneg);
        let idx = -b / (2.0 * a);
        let index = i0 as f32 + idx;
        self.dphi_hat = if i0 > self.nfft / 2 {
            (index - self.nfft as f32) * 2.0 * PI / self.nfft as f32
        } else {
            index * 2.0 * PI / self.nfft as f32
        };

        // estimate carrier phase offset
        let mut metric = Complex32::new(0.0, 0.0);
        for i in 0..self.s_len {
            metric += self.buf_time_0[i] * Complex32::from_polar(1.0, -self.dphi_hat * i as f32);
        }
        self.phi_hat = metric.arg();

        // set flag
        self.frame_detected = true;

        // reset state
        // copy saved buffer state (last half of buf_time_1 to front half of buf_time_0)
        let half = self.nfft / 2;
        for i in 0..half {
            self.buf_time_0[i] = self.buf_time_1[half + i];
        }
        self.state = QdetectorState::Seek;
        self.x2_sum_0 = self.buf_time_0[..half].iter().map(|x| x.norm_sqr()).sum();
        self.x2_sum_1 = 0.0;
        self.counter = half;
    }
}

impl std::fmt::Debug for Qdetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Qdetector {{ seq={}, nfft={}, dphi_max={}, thresh={}, energy={} }}",
            self.s_len, self.nfft, self.dphi_max, self.threshold, self.s2_sum
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_abs_diff_eq;
    use rand::Rng;

    fn qdetector_runtest_linear(sequence_len: usize) {
        let k = 2;
        let m = 7;
        let beta = 0.3;
        let ftype = FirFilterShape::Arkaiser;

        // generate synchronization sequence (QPSK symbols)
        let mut rng = rand::thread_rng();
        let sequence: Vec<Complex32> = (0..sequence_len)
            .map(|_| {
                let re = if rng.gen::<bool>() { 1.0 } else { -1.0 } * std::f32::consts::FRAC_1_SQRT_2;
                let im = if rng.gen::<bool>() { 1.0 } else { -1.0 } * std::f32::consts::FRAC_1_SQRT_2;
                Complex32::new(re, im)
            })
            .collect();

        // create detector
        let q = Qdetector::new_linear(&sequence, ftype, k, m, beta).unwrap();
        qdetector_runtest(q);
    }

    fn qdetector_runtest_gmsk(sequence_len: usize) {
        let k = 2;
        let m = 7;
        let beta = 0.3;

        // generate synchronization sequence (bits)
        let mut rng = rand::thread_rng();
        let sequence: Vec<u8> = (0..sequence_len).map(|_| rng.gen::<u8>() & 0x01).collect();

        // create detector
        let q = Qdetector::new_gmsk(&sequence, k, m, beta).unwrap();
        qdetector_runtest(q);
    }

    fn qdetector_runtest(mut q: Qdetector) {
        let gamma = 1.0;  // channel gain
        let tau = 0.0;    // fractional sample timing offset
        let dphi = 0.0;   // carrier frequency offset
        let phi = 0.5;    // carrier phase offset

        let seq = q.get_sequence().to_vec();
        let sequence_len = q.get_seq_len();
        let num_samples = 8 * sequence_len;

        // generate received signal with channel impairments
        let buf_rx: Vec<Complex32> = (0..num_samples)
            .map(|i| {
                let sample = if i < sequence_len { seq[i] } else { Complex32::new(0.0, 0.0) };
                sample * gamma * Complex32::from_polar(1.0, dphi * i as f32 + phi)
            })
            .collect();

        // try to detect frame
        let mut tau_hat = 0.0f32;
        let mut gamma_hat = 0.0f32;
        let mut dphi_hat = 0.0f32;
        let mut phi_hat = 0.0f32;
        let mut frame_detected = false;

        for &sample in buf_rx.iter() {
            if frame_detected {
                break;
            }

            if q.execute(sample).is_some() {
                frame_detected = true;
                tau_hat = q.get_tau();
                gamma_hat = q.get_gamma();
                dphi_hat = q.get_dphi();
                phi_hat = q.get_phi();
                break;
            }
        }

        assert!(frame_detected, "frame not detected");

        // check signal level estimate
        assert_abs_diff_eq!(gamma_hat, gamma, epsilon = 0.05);

        // check timing offset estimate
        assert_abs_diff_eq!(tau_hat, tau, epsilon = 0.05);

        // check carrier frequency offset estimate
        assert_abs_diff_eq!(dphi_hat, dphi, epsilon = 0.01);

        // check carrier phase offset estimate
        assert_abs_diff_eq!(phi_hat, phi, epsilon = 0.1);
    }

    // linear tests
    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n64)]
    fn test_qdetector_linear_n64() { qdetector_runtest_linear(64); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n83)]
    fn test_qdetector_linear_n83() { qdetector_runtest_linear(83); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n128)]
    fn test_qdetector_linear_n128() { qdetector_runtest_linear(128); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n167)]
    fn test_qdetector_linear_n167() { qdetector_runtest_linear(167); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n256)]
    fn test_qdetector_linear_n256() { qdetector_runtest_linear(256); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n335)]
    fn test_qdetector_linear_n335() { qdetector_runtest_linear(335); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n512)]
    fn test_qdetector_linear_n512() { qdetector_runtest_linear(512); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n671)]
    fn test_qdetector_linear_n671() { qdetector_runtest_linear(671); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n1024)]
    fn test_qdetector_linear_n1024() { qdetector_runtest_linear(1024); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_linear_n1341)]
    fn test_qdetector_linear_n1341() { qdetector_runtest_linear(1341); }

    // gmsk tests
    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n64)]
    fn test_qdetector_gmsk_n64() { qdetector_runtest_gmsk(64); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n83)]
    fn test_qdetector_gmsk_n83() { qdetector_runtest_gmsk(83); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n128)]
    fn test_qdetector_gmsk_n128() { qdetector_runtest_gmsk(128); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n167)]
    fn test_qdetector_gmsk_n167() { qdetector_runtest_gmsk(167); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n256)]
    fn test_qdetector_gmsk_n256() { qdetector_runtest_gmsk(256); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n335)]
    fn test_qdetector_gmsk_n335() { qdetector_runtest_gmsk(335); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n512)]
    fn test_qdetector_gmsk_n512() { qdetector_runtest_gmsk(512); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n671)]
    fn test_qdetector_gmsk_n671() { qdetector_runtest_gmsk(671); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n1024)]
    fn test_qdetector_gmsk_n1024() { qdetector_runtest_gmsk(1024); }

    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_gmsk_n1341)]
    fn test_qdetector_gmsk_n1341() { qdetector_runtest_gmsk(1341); }

    // copy test
    #[test]
    #[autotest_annotate(autotest_qdetector_cccf_copy)]
    fn test_qdetector_copy() {
        let sequence_len = 64;
        let sequence: Vec<Complex32> = (0..sequence_len)
            .map(|i| {
                let r = i as f32 - 0.5 * sequence_len as f32;
                Complex32::from_polar(1.0, 0.02 * r * r)
            })
            .collect();

        // create initial detector
        let mut q0 = Qdetector::new(&sequence).unwrap();

        // run on random-ish samples
        for i in 0..347 {
            q0.execute(Complex32::from_polar(1.0, i as f32));
        }

        // create new object (copied)
        let mut q1 = q0.clone();

        // try to detect frame
        let mut frames_detected = 0;
        for i in 0..(sequence_len + 80) {
            let s = if i < sequence_len {
                sequence[i]
            } else {
                Complex32::from_polar(1.0, i as f32)
            };

            let v0 = q0.execute(s);
            let v1 = q1.execute(s);

            match (v0.is_some(), v1.is_some()) {
                (true, true) => {
                    frames_detected += 1;
                    assert_eq!(q0.get_tau(), q1.get_tau());
                    assert_eq!(q0.get_gamma(), q1.get_gamma());
                    assert_eq!(q0.get_dphi(), q1.get_dphi());
                    assert_eq!(q0.get_phi(), q1.get_phi());
                }
                (true, false) => panic!("frame detected on detector 0 but not detector 1"),
                (false, true) => panic!("frame detected on detector 1 but not detector 0"),
                (false, false) => {}
            }
        }

        assert_eq!(frames_detected, 1);
    }
}
