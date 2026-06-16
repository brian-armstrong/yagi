// msource : multi-signal source generator

use crate::error::{Error, Result};
use crate::framing::qsource::{QSource, QSourceCallback, QSourceConfig, QSourceType, SourceId};
use crate::modem::modem::ModulationScheme;
use crate::multichannel::{ChannelizerType, FirPfbChannelizer2};

use num_complex::Complex32;

/// Multi-signal source generator
///
/// This object manages multiple signal sources and combines them
/// using a polyphase filterbank channelizer for efficient synthesis.
#[derive(Clone, Debug)]
pub struct MSource {
    /// Array of signal sources
    sources: Vec<QSource>,
    /// ID counter for assigning unique IDs
    id_counter: SourceId,
    /// Channelizer size (number of channels)
    m_channels: usize,
    /// Channelizer filter semi-length
    m: usize,
    /// Channelizer filter stop-band suppression (dB)
    as_: f32,
    /// Synthesis channelizer
    ch: FirPfbChannelizer2<Complex32>,
    /// Frequency domain buffer
    buf_freq: Vec<Complex32>,
    /// Time domain buffer
    buf_time: Vec<Complex32>,
    /// Output buffer read index
    read_index: usize,
    /// Total number of samples generated
    num_samples: u64,
}

impl MSource {
    /// Create a new MSource object
    ///
    /// # Arguments
    ///
    /// * `m_channels` - number of channels in synthesis channelizer
    /// * `m` - channelizer filter semi-length
    /// * `as_` - channelizer filter stop-band suppression (dB)
    pub fn new(m_channels: usize, m: usize, as_: f32) -> Result<Self> {
        if m_channels < 2 {
            return Err(Error::Config(
                "number of subcarriers must be at least 2".into(),
            ));
        }
        if m_channels % 2 != 0 {
            return Err(Error::Config(
                "number of subcarriers must be even".into(),
            ));
        }
        if m == 0 {
            return Err(Error::Config(
                "filter semi-length must be greater than zero".into(),
            ));
        }

        let ch = FirPfbChannelizer2::new_kaiser(ChannelizerType::Synthesizer, m_channels, m, as_)?;

        let buf_freq = vec![Complex32::new(0.0, 0.0); m_channels];
        let buf_time = vec![Complex32::new(0.0, 0.0); m_channels / 2];

        Ok(Self {
            sources: Vec::new(),
            id_counter: 0,
            m_channels,
            m,
            as_,
            ch,
            buf_freq,
            buf_time,
            read_index: m_channels / 2, // indicate buffer is empty
            num_samples: 0,
        })
    }

    /// Create MSource with default parameters (M=1200, m=4, as=60dB)
    pub fn new_default() -> Result<Self> {
        Self::new(1200, 4, 60.0)
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.read_index = self.m_channels / 2;
    }

    /// Get number of samples generated so far
    pub fn get_num_samples(&self) -> u64 {
        self.num_samples
    }

    /// Get number of sources
    pub fn get_num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Add a source to the list and return its ID
    fn add_source(&mut self, mut source: QSource) -> SourceId {
        let id = self.id_counter;
        source.set_id(id);
        self.sources.push(source);
        self.id_counter += 1;
        id
    }

    /// Find source index by ID, returns None if not found
    fn find(&self, id: SourceId) -> Option<usize> {
        self.sources.iter().position(|s| s.get_id() == id)
    }

    /// Get mutable reference to source by ID
    fn get_source_mut(&mut self, id: SourceId) -> Result<&mut QSource> {
        match self.find(id) {
            Some(idx) => Ok(&mut self.sources[idx]),
            None => Err(Error::Range(format!("source with id {} not found", id))),
        }
    }

    /// Get reference to source by ID
    fn get_source(&self, id: SourceId) -> Result<&QSource> {
        match self.find(id) {
            Some(idx) => Ok(&self.sources[idx]),
            None => Err(Error::Range(format!("source with id {} not found", id))),
        }
    }

    /// Add tone source
    ///
    /// # Arguments
    ///
    /// * `fc` - center frequency [-0.5, 0.5]
    /// * `bw` - bandwidth [0, 1]
    /// * `gain` - signal gain (dB)
    ///
    /// # Returns
    ///
    /// ID of the added source
    pub fn add_tone(&mut self, fc: f32, bw: f32, gain: f32) -> Result<SourceId> {
        let source = QSource::new(self.m_channels, self.m, self.as_, fc, bw, gain, QSourceConfig::Tone)?;
        Ok(self.add_source(source))
    }

    /// Add chirp source
    ///
    /// # Arguments
    ///
    /// * `fc` - center frequency [-0.5, 0.5]
    /// * `bw` - bandwidth [0, 1]
    /// * `gain` - signal gain (dB)
    /// * `duration` - chirp duration in samples
    /// * `negate` - negate frequency direction
    /// * `single` - run single chirp or repeatedly
    ///
    /// # Returns
    ///
    /// ID of the added source
    pub fn add_chirp(
        &mut self,
        fc: f32,
        bw: f32,
        gain: f32,
        duration: f32,
        negate: bool,
        single: bool,
    ) -> Result<SourceId> {
        let config = QSourceConfig::Chirp { duration, negate, single };
        let source = QSource::new(self.m_channels, self.m, self.as_, fc, bw, gain, config)?;
        Ok(self.add_source(source))
    }

    /// Add noise source
    ///
    /// # Arguments
    ///
    /// * `fc` - center frequency [-0.5, 0.5]
    /// * `bw` - bandwidth [0, 1]
    /// * `gain` - signal gain (dB)
    ///
    /// # Returns
    ///
    /// ID of the added source
    pub fn add_noise(&mut self, fc: f32, bw: f32, gain: f32) -> Result<SourceId> {
        let source = QSource::new(self.m_channels, self.m, self.as_, fc, bw, gain, QSourceConfig::Noise)?;
        Ok(self.add_source(source))
    }

    /// Add linear modulation source
    ///
    /// # Arguments
    ///
    /// * `fc` - center frequency [-0.5, 0.5]
    /// * `bw` - bandwidth [0, 1] (doubled internally for 2 samples/symbol)
    /// * `gain` - signal gain (dB)
    /// * `ms` - modulation scheme
    /// * `m` - filter delay (symbols)
    /// * `beta` - filter excess bandwidth factor
    ///
    /// # Returns
    ///
    /// ID of the added source
    pub fn add_modem(
        &mut self,
        fc: f32,
        bw: f32,
        gain: f32,
        ms: ModulationScheme,
        m: usize,
        beta: f32,
    ) -> Result<SourceId> {
        // create object with double the bandwidth to account for 2 samples/symbol
        let config = QSourceConfig::Modem { scheme: ms, m, beta };
        let source = QSource::new(self.m_channels, self.m, self.as_, fc, 2.0 * bw, gain, config)?;
        Ok(self.add_source(source))
    }

    /// Add FSK modulation source
    ///
    /// # Arguments
    ///
    /// * `fc` - center frequency [-0.5, 0.5]
    /// * `bw` - bandwidth [0, 1] (doubled internally for k samples/symbol)
    /// * `gain` - signal gain (dB)
    /// * `m` - bits per symbol
    /// * `k` - samples per symbol
    ///
    /// # Returns
    ///
    /// ID of the added source
    pub fn add_fsk(
        &mut self,
        fc: f32,
        bw: f32,
        gain: f32,
        m: usize,
        k: usize,
    ) -> Result<SourceId> {
        // create object with double the bandwidth to account for k samples/symbol
        let config = QSourceConfig::Fsk { m, k };
        let source = QSource::new(self.m_channels, self.m, self.as_, fc, 2.0 * bw, gain, config)?;
        Ok(self.add_source(source))
    }

    /// Add GMSK modulation source
    ///
    /// # Arguments
    ///
    /// * `fc` - center frequency [-0.5, 0.5]
    /// * `bw` - bandwidth [0, 1] (doubled internally for 2 samples/symbol)
    /// * `gain` - signal gain (dB)
    /// * `m` - filter delay (symbols)
    /// * `bt` - bandwidth-time product
    ///
    /// # Returns
    ///
    /// ID of the added source
    pub fn add_gmsk(
        &mut self,
        fc: f32,
        bw: f32,
        gain: f32,
        m: usize,
        bt: f32,
    ) -> Result<SourceId> {
        // create object with double the bandwidth to account for 2 samples/symbol
        let config = QSourceConfig::Gmsk { m, bt };
        let source = QSource::new(self.m_channels, self.m, self.as_, fc, 2.0 * bw, gain, config)?;
        Ok(self.add_source(source))
    }

    /// Add user-defined signal generator
    ///
    /// # Arguments
    ///
    /// * `fc` - center frequency [-0.5, 0.5]
    /// * `bw` - bandwidth [0, 1]
    /// * `gain` - signal gain (dB)
    /// * `callback` - object implementing QSourceCallback trait
    ///
    /// # Returns
    ///
    /// ID of the added source
    pub fn add_user<C: QSourceCallback + 'static>(
        &mut self,
        fc: f32,
        bw: f32,
        gain: f32,
        callback: C,
    ) -> Result<SourceId> {
        let config = QSourceConfig::User(Box::new(callback));
        let source = QSource::new(self.m_channels, self.m, self.as_, fc, bw, gain, config)?;
        Ok(self.add_source(source))
    }

    /// Remove a source by ID
    pub fn remove(&mut self, id: SourceId) -> Result<()> {
        match self.find(id) {
            Some(idx) => {
                self.sources.remove(idx);
                Ok(())
            }
            None => Err(Error::Range(format!("source with id {} not found", id))),
        }
    }

    /// Enable a source by ID
    pub fn enable(&mut self, id: SourceId) -> Result<()> {
        self.get_source_mut(id)?.enable();
        Ok(())
    }

    /// Disable a source by ID
    pub fn disable(&mut self, id: SourceId) -> Result<()> {
        self.get_source_mut(id)?.disable();
        Ok(())
    }

    /// Set gain of a source by ID
    pub fn set_gain(&mut self, id: SourceId, gain_db: f32) -> Result<()> {
        self.get_source_mut(id)?.set_gain(gain_db);
        Ok(())
    }

    /// Get gain of a source by ID
    pub fn get_gain(&self, id: SourceId) -> Result<f32> {
        Ok(self.get_source(id)?.get_gain())
    }

    /// Set frequency of a source by ID
    pub fn set_frequency(&mut self, id: SourceId, fc: f32) -> Result<()> {
        self.get_source_mut(id)?.set_frequency(fc)
    }

    /// Get frequency of a source by ID
    pub fn get_frequency(&self, id: SourceId) -> Result<f32> {
        Ok(self.get_source(id)?.get_frequency())
    }

    /// Get number of samples generated by a specific source
    pub fn get_num_samples_source(&self, id: SourceId) -> Result<u64> {
        Ok(self.get_source(id)?.get_num_samples())
    }

    /// Get source type by ID
    pub fn get_source_type(&self, id: SourceId) -> Result<QSourceType> {
        Ok(self.get_source(id)?.get_type())
    }

    /// Generate samples internally
    fn generate(&mut self) -> Result<()> {
        // clear frequency buffer
        for v in self.buf_freq.iter_mut() {
            *v = Complex32::new(0.0, 0.0);
        }

        // add sources into main frequency buffer
        for source in self.sources.iter_mut() {
            source.generate_into(&mut self.buf_freq)?;
        }

        // run synthesis channelizer
        self.ch.execute_synthesizer(&self.buf_freq, &mut self.buf_time)?;

        // update state
        self.read_index = 0;
        self.num_samples += (self.m_channels / 2) as u64;
        Ok(())
    }

    /// Write samples to output buffer
    pub fn write_samples(&mut self, buf: &mut [Complex32]) -> Result<()> {
        let m2 = self.m_channels / 2;
        for sample in buf.iter_mut() {
            // generate more samples if needed
            if self.read_index >= m2 {
                self.generate()?;
            }

            *sample = self.buf_time[self.read_index];
            self.read_index += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::spgram::Spgram;
    use crate::utility::test_helpers::{validate_psd_spgramcf, PsdRegion};
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_msourcecf_config)]
    fn test_msourcecf_config() {
        // check invalid configurations
        assert!(MSource::new(0, 12, 60.0).is_err()); // too few subcarriers
        assert!(MSource::new(17, 12, 60.0).is_err()); // odd-numbered subcarriers
        assert!(MSource::new(64, 0, 60.0).is_err()); // filter semi-length too small

        // create proper object and test configurations
        let mut q = MSource::new(64, 12, 60.0).unwrap();

        // try to configure signals with invalid IDs
        assert!(q.remove(12345).is_err());
        assert!(q.enable(12345).is_err());
        assert!(q.disable(12345).is_err());
        assert!(q.set_gain(12345, 0.0).is_err());
        assert!(q.get_gain(12345).is_err());
        assert!(q.set_frequency(12345, 0.0).is_err());
        assert!(q.get_frequency(12345).is_err());

        // add signals and check setting values appropriately
        let id_tone = q.add_tone(-0.123456, 0.0, 20.0).unwrap();
        let id_gmsk = q.add_gmsk(0.220780, 0.05, 0.0, 4, 0.3).unwrap();

        // remove tone
        assert!(q.remove(id_tone).is_ok());
        assert!(q.set_gain(id_tone, 10.0).is_err());

        // disable GMSK signal
        assert!(q.disable(id_gmsk).is_ok());

        // assert buffer is zeros (only GMSK signal present and it's disabled)
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert_eq!(sum, 0.0);

        // enable GMSK signal
        assert!(q.enable(id_gmsk).is_ok());
        q.write_samples(&mut buf).unwrap();
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    #[autotest_annotate(autotest_msourcecf_accessor)]
    fn test_msourcecf_accessor() {
        // create object and add signals
        let mut q = MSource::new(240, 12, 60.0).unwrap();
        let id_tone = q.add_tone(-0.123456, 0.0, 20.0).unwrap();
        let id_noise = q.add_noise(0.220780, 0.10, 0.0).unwrap();

        // check center frequency of tone
        let fc = q.get_frequency(id_tone).unwrap();
        assert!((fc - (-0.123456)).abs() < 1e-5);

        // check center frequency of noise signal
        let fc = q.get_frequency(id_noise).unwrap();
        assert!((fc - 0.220780).abs() < 1e-5);

        // check gain of tone
        let gain = q.get_gain(id_tone).unwrap();
        assert!((gain - 20.0).abs() < 0.1);

        // check gain of noise signal
        let gain = q.get_gain(id_noise).unwrap();
        assert!(gain.abs() < 0.1);

        // remove tone
        q.remove(id_tone).unwrap();

        // disable noise signal
        q.disable(id_noise).unwrap();

        // set frequency of noise signal
        q.set_frequency(id_noise, 0.33333).unwrap();
        let fc = q.get_frequency(id_noise).unwrap();
        assert!((fc - 0.33333).abs() < 1e-5);

        // set gain of noise signal
        q.set_gain(id_noise, 30.0).unwrap();
        let gain = q.get_gain(id_noise).unwrap();
        assert!((gain - 30.0).abs() < 0.1);

        // assert buffer is zeros
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert_eq!(sum, 0.0);

        // enable noise signal and check power spectral density
        q.enable(id_noise).unwrap();

        let nfft = 2400;
        let mut spgram = Spgram::<Complex32>::default(nfft).unwrap();

        while q.get_num_samples() < 192000 {
            q.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        let regions = [
            // noise floor between signals
            PsdRegion { fmin: -0.500, fmax:  0.275, pmin: -80.0, pmax: -40.0, test_lo: false, test_hi: true },
            PsdRegion { fmin:  0.285, fmax:  0.375, pmin:  28.0, pmax:  32.0, test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.385, fmax:  0.500, pmin: -80.0, pmax: -40.0, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_spgramcf(&spgram, &regions).unwrap());
    }

    #[test]
    fn test_msourcecf_tone_basic() {
        let mut q = MSource::new_default().unwrap();

        // add a tone
        let id = q.add_tone(0.1, 0.0, 0.0).unwrap();
        assert_eq!(id, 0);

        // generate some samples
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();

        // verify samples are non-zero
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_msourcecf_chirp_basic() {
        let mut q = MSource::new_default().unwrap();

        // add a chirp
        let id = q.add_chirp(0.0, 0.2, 0.0, 1000.0, false, false).unwrap();
        assert_eq!(id, 0);

        // generate some samples
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();

        // verify samples are non-zero
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_msourcecf_modem_basic() {
        let mut q = MSource::new_default().unwrap();

        // add a linear modulation source
        let id = q.add_modem(0.0, 0.1, 0.0, ModulationScheme::Qpsk, 12, 0.3).unwrap();
        assert_eq!(id, 0);

        // generate some samples
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();

        // verify samples are non-zero
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_msourcecf_fsk_basic() {
        let mut q = MSource::new_default().unwrap();

        // add an FSK modulation source (2 bits/symbol, 4 samples/symbol)
        let id = q.add_fsk(0.0, 0.1, 0.0, 2, 4).unwrap();
        assert_eq!(id, 0);

        // generate some samples
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();

        // verify samples are non-zero
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_msourcecf_gmsk_basic() {
        let mut q = MSource::new_default().unwrap();

        // add a GMSK modulation source
        let id = q.add_gmsk(0.0, 0.1, 0.0, 3, 0.25).unwrap();
        assert_eq!(id, 0);

        // generate some samples
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();

        // verify samples are non-zero
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);
    }

    // Test user callback source
    #[derive(Clone)]
    struct TestConstantSource {
        value: Complex32,
    }

    impl QSourceCallback for TestConstantSource {
        fn generate(&mut self, output: &mut [Complex32]) -> crate::error::Result<()> {
            for sample in output.iter_mut() {
                *sample = self.value;
            }
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn QSourceCallback> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_msourcecf_user_basic() {
        let mut q = MSource::new_default().unwrap();

        // add a user-defined source
        let source = TestConstantSource {
            value: Complex32::new(1.0, 0.0),
        };
        let id = q.add_user(0.0, 0.1, 0.0, source).unwrap();
        assert_eq!(id, 0);

        // generate some samples
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();

        // verify samples are non-zero
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_msourcecf_multiple_sources() {
        let mut q = MSource::new(64, 12, 60.0).unwrap();

        // add multiple sources
        let id0 = q.add_noise(0.0, 1.0, -40.0).unwrap();
        let id1 = q.add_tone(-0.4, 0.0, 20.0).unwrap();
        let id2 = q.add_tone(-0.3, 0.0, 10.0).unwrap();

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(q.get_num_sources(), 3);

        // generate samples
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();

        // verify samples are non-zero
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);

        // remove a source
        q.remove(id1).unwrap();
        assert_eq!(q.get_num_sources(), 2);
    }

    #[test]
    fn test_msourcecf_spectrum() {
        // This test verifies msource produces signals at expected frequencies/powers
        let mut gen = MSource::new_default().unwrap();

        // Add signals matching the firpfbchr test
        gen.add_noise(0.0, 1.0, -60.0).unwrap();      // wide-band noise floor
        gen.add_noise(-0.30, 0.10, -20.0).unwrap();   // narrow-band noise at -0.30
        gen.add_noise(0.08, 0.01, -30.0).unwrap();    // very narrow-band noise at 0.08
        gen.add_modem(0.1875, 0.065, -20.0, ModulationScheme::Qpsk, 12, 0.3).unwrap();

        let nfft = 2400;
        let mut spgram = Spgram::<Complex32>::default(nfft).unwrap();

        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];

        // Generate samples
        for _ in 0..(1 << 16) {
            gen.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        let psd = spgram.get_psd();

        // Helper to check region
        let check_region = |fmin: f32, fmax: f32, label: &str| {
            let i_start = ((fmin + 0.5) * nfft as f32) as usize;
            let i_end = ((fmax + 0.5) * nfft as f32) as usize;
            let region: Vec<f32> = psd[i_start..i_end].to_vec();
            let min = region.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = region.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("{}: min={:.1} dB, max={:.1} dB", label, min, max);
            (min, max)
        };

        // Check regions
        let (floor_min, floor_max) = check_region(0.35, 0.45, "Noise floor [0.35, 0.45]");
        let (nb_min, nb_max) = check_region(-0.34, -0.26, "Narrow-band [-0.34, -0.26]");
        let (_vnb_min, vnb_max) = check_region(0.07, 0.09, "Very narrow [0.07, 0.09]");
        let (_mod_min, mod_max) = check_region(0.17, 0.20, "Modem [0.17, 0.20]");

        // Verify noise floor is around -60 dB
        assert!(floor_min > -65.0 && floor_max < -55.0,
            "Noise floor should be ~-60 dB, got [{}, {}]", floor_min, floor_max);

        // Verify narrow-band noise is around -20 dB
        assert!(nb_min > -25.0 && nb_max < -15.0,
            "Narrow-band noise should be ~-20 dB, got [{}, {}]", nb_min, nb_max);

        // Verify very narrow-band noise peak is around -30 dB
        // (region includes noise floor at edges due to narrow bandwidth)
        assert!(vnb_max > -35.0 && vnb_max < -25.0,
            "Very narrow noise peak should be ~-30 dB, got max={}", vnb_max);

        // Verify modem signal is present and above noise floor
        // Note: actual power differs from gain setting due to modulation/filtering
        assert!(mod_max > floor_max + 10.0,
            "Modem signal should be above noise floor, got max={} vs floor={}", mod_max, floor_max);
    }

    #[test]
    #[autotest_annotate(autotest_msourcecf_tone)]
    fn test_msourcecf_tone() {
        let nfft = 2400;
        let num_samples = 192000;

        let mut spgram = Spgram::<Complex32>::default(nfft).unwrap();

        let mut gen = MSource::new_default().unwrap();
        // add signals (fc, bw, gain)
        gen.add_noise(0.0, 1.0, -40.0).unwrap();   // wide-band noise
        gen.add_tone(-0.4, 0.0, 20.0).unwrap();    // tone
        gen.add_tone(-0.3, 0.0, 10.0).unwrap();    // tone
        gen.add_tone(-0.2, 0.0, 0.0).unwrap();     // tone
        gen.add_tone(-0.1, 0.0, -10.0).unwrap();   // tone
        gen.add_tone(0.0, 0.0, -20.0).unwrap();    // tone
        gen.add_tone(0.1, 0.0, -30.0).unwrap();    // tone

        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        while gen.get_num_samples() < num_samples as u64 {
            gen.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        let regions = [
            // noise floor between signals
            PsdRegion { fmin: -0.500, fmax: -0.405, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.395, fmax: -0.305, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.295, fmax: -0.205, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.195, fmax: -0.105, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.095, fmax: -0.005, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.005, fmax:  0.095, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.105, fmax:  0.500, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            // tones
            PsdRegion { fmin: -0.401, fmax: -0.399, pmin:  10.0, pmax:  22.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.301, fmax: -0.299, pmin:   0.0, pmax:  12.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.201, fmax: -0.199, pmin: -10.0, pmax:   2.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.101, fmax: -0.099, pmin: -20.0, pmax:  -8.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.001, fmax:  0.001, pmin: -30.0, pmax: -18.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.099, fmax:  0.101, pmin: -40.0, pmax: -28.0, test_lo: true, test_hi: true },
        ];
        assert!(validate_psd_spgramcf(&spgram, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_msourcecf_chirp)]
    fn test_msourcecf_chirp() {
        let nfft = 2400;
        let num_samples: u64 = 192000;

        let mut spgram = Spgram::<Complex32>::default(nfft).unwrap();

        let mut gen = MSource::new_default().unwrap();
        // add signals (fc, bw, gain, duration, negate, single)
        gen.add_noise(0.0, 1.0, -40.0).unwrap();   // wide-band noise
        gen.add_chirp(0.0, 0.60, 20.0, (num_samples as f32) * 0.9, false, true).unwrap();

        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        while gen.get_num_samples() < num_samples {
            gen.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        let regions = [
            // noise floor outside chirp bandwidth
            PsdRegion { fmin: -0.500, fmax: -0.305, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            // chirp signal bandwidth
            PsdRegion { fmin: -0.295, fmax:  0.295, pmin:  15.0, pmax:  22.0, test_lo: true, test_hi: true },
            // noise floor outside chirp bandwidth
            PsdRegion { fmin:  0.305, fmax:  0.500, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
        ];
        assert!(validate_psd_spgramcf(&spgram, &regions).unwrap());
    }

    // User callback for aggregate test: generates pulse train (1 every 8 samples)
    // which creates evenly spaced tones in frequency domain
    #[derive(Clone)]
    struct PulseTrainSource {
        counter: usize,
    }

    impl QSourceCallback for PulseTrainSource {
        fn generate(&mut self, output: &mut [Complex32]) -> crate::error::Result<()> {
            for sample in output.iter_mut() {
                *sample = if self.counter == 0 {
                    Complex32::new(1.0, 0.0)
                } else {
                    Complex32::new(0.0, 0.0)
                };
                self.counter = (self.counter + 1) % 8;
            }
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn QSourceCallback> {
            Box::new(self.clone())
        }
    }

    #[test]
    #[autotest_annotate(autotest_msourcecf_aggregate)]
    fn test_msourcecf_aggregate() {
        // msource parameters
        let ms = ModulationScheme::Qpsk;    // linear modulation scheme
        let m = 12;                          // modulation filter semi-length
        let beta = 0.30;                     // modulation filter excess bandwidth factor
        let bt = 0.35;                       // GMSK filter bandwidth-time factor

        // spectral periodogram options
        let nfft = 2400;
        let num_samples: u64 = 192000;

        let mut spgram = Spgram::<Complex32>::default(nfft).unwrap();

        let buf_len = 1024;
        let mut buf = vec![Complex32::new(0.0, 0.0); buf_len];

        // create multi-signal source generator
        let mut gen = MSource::new_default().unwrap();

        // add signals     (fc,    bw,    gain, {options})
        gen.add_noise(0.00, 1.00, -40.0).unwrap();                      // wide-band noise
        gen.add_tone(-0.45, 0.00, 20.0).unwrap();                       // tone
        gen.add_fsk(-0.33, 0.05, -10.0, 3, 16).unwrap();                // FSK
        gen.add_gmsk(-0.20, 0.05, 0.0, m, bt).unwrap();                 // modulated data (GMSK)
        gen.add_noise(-0.05, 0.10, 0.0).unwrap();                       // narrow-band noise
        gen.add_chirp(0.07, 0.07, 20.0, 8000.0, false, false).unwrap(); // chirp
        gen.add_modem(0.20, 0.10, 0.0, ms, m, beta).unwrap();           // modulated data (linear)
        gen.add_user(0.40, 0.15, -10.0, PulseTrainSource { counter: 0 }).unwrap(); // tones

        while gen.get_num_samples() < num_samples {
            gen.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        let regions = [
            // noise floor between signals
            PsdRegion { fmin: -0.500, fmax: -0.455, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.445, fmax: -0.385, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.275, fmax: -0.260, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.140, fmax: -0.110, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.005, fmax:  0.030, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.110, fmax:  0.130, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.270, fmax:  0.320, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            // space between tones
            PsdRegion { fmin:  0.328, fmax:  0.338, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.348, fmax:  0.358, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.368, fmax:  0.378, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.386, fmax:  0.396, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.405, fmax:  0.415, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.424, fmax:  0.434, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.442, fmax:  0.452, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            // end
            PsdRegion { fmin:  0.461, fmax:  0.500, pmin: -43.0, pmax: -37.0, test_lo: true, test_hi: true },
            // signals
            PsdRegion { fmin: -0.451, fmax: -0.449, pmin:  10.0, pmax:  22.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.355, fmax: -0.305, pmin: -15.0, pmax:   0.0, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.220, fmax: -0.180, pmin: -18.0, pmax:   6.5, test_lo: true, test_hi: true },
            PsdRegion { fmin: -0.095, fmax: -0.005, pmin:  -5.0, pmax:   2.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.037, fmax:  0.102, pmin:  18.0, pmax:  22.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.160, fmax:  0.240, pmin:  -5.0, pmax:   2.0, test_lo: true, test_hi: true },
            // tones
            PsdRegion { fmin:  0.3245, fmax:  0.3255, pmin: -20.0, pmax:  0.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.3432, fmax:  0.3442, pmin: -20.0, pmax:  0.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.3620, fmax:  0.3630, pmin: -20.0, pmax:  0.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.3810, fmax:  0.3820, pmin: -20.0, pmax:  0.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.3995, fmax:  0.4005, pmin: -20.0, pmax:  0.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.4182, fmax:  0.4192, pmin: -20.0, pmax:  0.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.4370, fmax:  0.4380, pmin: -20.0, pmax:  0.0, test_lo: true, test_hi: true },
            PsdRegion { fmin:  0.4555, fmax:  0.4565, pmin: -20.0, pmax:  0.0, test_lo: true, test_hi: true },
        ];
        assert!(validate_psd_spgramcf(&spgram, &regions).unwrap());
    }
}
