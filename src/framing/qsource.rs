// qsource : generic single signal source generator

use crate::error::{Error, Result};

/// Source identifier type
pub type SourceId = i32;
use crate::filter::resamp::Resamp;
use crate::framing::symstream::SymStream;
use crate::modem::fskmod::Fskmod;
use crate::modem::gmskmod::GmskMod;
use crate::multichannel::{ChannelizerType, FirPfbChannelizer2};
use crate::nco::{Osc, OscScheme};
use crate::random::randnf;

use num_complex::Complex32;
use std::f32::consts::PI;
use std::sync::Arc;

// Note: The resamp field is created but unused; liquid-dsp has a "TODO: push through resampler"
// comment in generate() as well. The current implementation works without it.

/// User-defined signal source callback trait
///
/// Implement this trait to create custom signal generators for use with QSource.
/// The callback is invoked to fill a buffer with generated samples.
///
/// # Example
///
/// ```ignore
/// struct MySineSource {
///     phase: f32,
///     freq: f32,
/// }
///
/// impl QSourceCallback for MySineSource {
///     fn generate(&mut self, output: &mut [Complex32]) -> Result<()> {
///         for sample in output.iter_mut() {
///             *sample = Complex32::new(self.phase.cos(), self.phase.sin());
///             self.phase += self.freq;
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait QSourceCallback: Send + Sync {
    /// Generate samples into the output buffer
    ///
    /// # Arguments
    ///
    /// * `output` - buffer to fill with generated samples
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or an error if generation fails
    fn generate(&mut self, output: &mut [Complex32]) -> Result<()>;

    /// Clone the callback into a boxed trait object
    fn clone_box(&self) -> Box<dyn QSourceCallback>;
}

impl Clone for Box<dyn QSourceCallback> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl std::fmt::Debug for dyn QSourceCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QSourceCallback")
    }
}

/// Signal source type (for querying)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QSourceType {
    User,
    Tone,
    Chirp,
    Noise,
    Modem,
    Fsk,
    Gmsk,
}

/// Signal source configuration
#[derive(Clone)]
pub enum QSourceConfig {
    /// Simple tone (CW)
    Tone,
    /// Linear frequency chirp
    Chirp {
        /// Duration in samples
        duration: f32,
        /// Negate frequency direction
        negate: bool,
        /// Single chirp (disables source after completion) vs repeated
        single: bool,
    },
    /// Gaussian noise
    Noise,
    /// Linear modulation (PSK/QAM)
    Modem {
        /// Modulation scheme
        scheme: crate::modem::modem::ModulationScheme,
        /// Filter delay (symbols)
        m: usize,
        /// Filter excess bandwidth factor
        beta: f32,
    },
    /// FSK modulation
    Fsk {
        /// Bits per symbol
        m: usize,
        /// Samples per symbol
        k: usize,
    },
    /// GMSK modulation
    Gmsk {
        /// Filter delay (symbols)
        m: usize,
        /// Bandwidth-time product
        bt: f32,
    },
    /// User-defined callback
    User(Box<dyn QSourceCallback>),
}

impl std::fmt::Debug for QSourceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QSourceConfig::Tone => write!(f, "Tone"),
            QSourceConfig::Chirp { duration, negate, single } => {
                f.debug_struct("Chirp")
                    .field("duration", duration)
                    .field("negate", negate)
                    .field("single", single)
                    .finish()
            }
            QSourceConfig::Noise => write!(f, "Noise"),
            QSourceConfig::Modem { scheme, m, beta } => {
                f.debug_struct("Modem")
                    .field("scheme", scheme)
                    .field("m", m)
                    .field("beta", beta)
                    .finish()
            }
            QSourceConfig::Fsk { m, k } => {
                f.debug_struct("Fsk")
                    .field("m", m)
                    .field("k", k)
                    .finish()
            }
            QSourceConfig::Gmsk { m, bt } => {
                f.debug_struct("Gmsk")
                    .field("m", m)
                    .field("bt", bt)
                    .finish()
            }
            QSourceConfig::User(_) => write!(f, "User"),
        }
    }
}

/// Chirp state
#[derive(Clone, Debug)]
struct ChirpState {
    nco: Osc,
    df: f32,
    negate: bool,
    single: bool,
    num: u64,
    timer: u64,
}

/// Modem state
#[derive(Clone, Debug)]
struct ModemState {
    symstream: SymStream,
}

/// FSK state
#[derive(Clone, Debug)]
struct FskState {
    modulator: Fskmod,
    buf: Vec<Complex32>,
    mask: usize,
    index: usize,
}

/// GMSK state
#[derive(Clone, Debug)]
struct GmskState {
    modulator: GmskMod,
    buf: [Complex32; 2],
    index: usize,
}

/// User callback state
#[derive(Clone)]
struct UserState {
    callback: Arc<std::sync::Mutex<Box<dyn QSourceCallback>>>,
}

impl std::fmt::Debug for UserState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserState").finish()
    }
}

/// Internal source state
#[derive(Clone, Debug)]
enum SourceState {
    User(UserState),
    Tone,
    Chirp(ChirpState),
    Noise,
    Modem(ModemState),
    Fsk(FskState),
    Gmsk(GmskState),
}

/// Generic single signal source generator
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct QSource {
    id: SourceId,
    m_channels: usize,
    p_channels: usize,
    m: usize,
    as_: f32,
    fc: f32,
    bw: f32,
    index: usize,
    resamp: Resamp<Complex32>,
    mixer: Osc,
    gain: f32,
    gain_ch: f32,
    buf: Vec<Complex32>,
    buf_time: Vec<Complex32>,
    buf_freq: Vec<Complex32>,
    ch: FirPfbChannelizer2<Complex32>,
    enabled: bool,
    num_samples: u64,
    source: SourceState,
}

impl QSource {
    /// Create a new QSource object
    ///
    /// # Arguments
    ///
    /// * `m_channels` - number of channels in parent object's synthesis channelizer
    /// * `m` - channelizer filter semi-length
    /// * `as_` - channelizer filter stop-band suppression (dB)
    /// * `fc` - signal normalized center frequency [-0.5, 0.5]
    /// * `bw` - signal normalized bandwidth [0, 1]
    /// * `gain` - signal gain (dB)
    /// * `config` - source type configuration
    pub fn new(
        m_channels: usize,
        m: usize,
        as_: f32,
        fc: f32,
        bw: f32,
        gain: f32,
        config: QSourceConfig,
    ) -> Result<Self> {
        if m_channels < 2 || (m_channels % 2) != 0 {
            return Err(Error::Config(
                "invalid channelizer size; must be even and greater than 1".into(),
            ));
        }
        if m == 0 {
            return Err(Error::Config(
                "invalid channelizer filter semi-length; must be greater than 0".into(),
            ));
        }
        if fc < -0.5 || fc > 0.5 {
            return Err(Error::Config(
                "invalid frequency offset; must be in [-0.5, 0.5]".into(),
            ));
        }
        if bw < 0.0 || bw > 1.0 {
            return Err(Error::Config(
                "invalid bandwidth; must be in [0, 1]".into(),
            ));
        }

        // set channelizer values appropriately
        let p_channels = (2.0 * (0.5 * bw * m_channels as f32).ceil()) as usize;
        let p_channels = p_channels.max(2);

        // create resampler to correct for rate offset
        let rate = if bw == 0.0 {
            1.0
        } else {
            bw * (m_channels as f32) / (p_channels as f32)
        };
        let resamp = Resamp::new(rate, 12, 0.45, as_, 64)?;

        // create mixer for frequency offset correction
        let mixer = Osc::new(OscScheme::Vco);

        // create buffers
        let buf_len = 64;
        let buf = vec![Complex32::new(0.0, 0.0); buf_len];
        let buf_time = vec![Complex32::new(0.0, 0.0); p_channels / 2];
        let buf_freq = vec![Complex32::new(0.0, 0.0); p_channels];

        // create analysis channelizer
        let ch = FirPfbChannelizer2::new_kaiser(ChannelizerType::Analyzer, p_channels, m, as_)?;

        // channelizer gain correction
        let gain_ch = ((p_channels as f32) / (m_channels as f32)).sqrt();

        // Initialize source state from config
        let source = match config {
            QSourceConfig::Tone => SourceState::Tone,
            QSourceConfig::Chirp { duration, negate, single } => {
                let mut nco = Osc::new(OscScheme::Vco);
                let num = (duration * bw).round() as u64;
                let df = 2.0 * PI / (num as f32) * (if negate { -1.0 } else { 1.0 });
                nco.set_frequency(if negate { PI } else { -PI });
                SourceState::Chirp(ChirpState {
                    nco,
                    df,
                    negate,
                    single,
                    num,
                    timer: num,
                })
            }
            QSourceConfig::Noise => SourceState::Noise,
            QSourceConfig::Modem { scheme, m: filter_m, beta } => {
                let symstream = SymStream::new_linear(
                    crate::filter::FirFilterShape::Arkaiser,
                    2,  // k = 2 samples per symbol (fixed)
                    filter_m,
                    beta,
                    scheme,
                )?;
                SourceState::Modem(ModemState { symstream })
            }
            QSourceConfig::Fsk { m: bits_per_sym, k } => {
                let modulator = Fskmod::new(bits_per_sym, k, 0.25)?;
                let mask = (1 << bits_per_sym) - 1;
                SourceState::Fsk(FskState {
                    modulator,
                    buf: vec![Complex32::new(0.0, 0.0); k],
                    mask,
                    index: 0,
                })
            }
            QSourceConfig::Gmsk { m: filter_m, bt } => {
                let modulator = GmskMod::new(2, filter_m, bt)?;
                SourceState::Gmsk(GmskState {
                    modulator,
                    buf: [Complex32::new(0.0, 0.0); 2],
                    index: 0,
                })
            }
            QSourceConfig::User(callback) => {
                SourceState::User(UserState {
                    callback: Arc::new(std::sync::Mutex::new(callback)),
                })
            }
        };

        let mut q = Self {
            id: -1,
            m_channels,
            p_channels,
            m,
            as_,
            fc,
            bw,
            index: 0,
            resamp,
            mixer,
            gain: 10.0f32.powf(gain / 20.0),
            gain_ch,
            buf,
            buf_time,
            buf_freq,
            ch,
            enabled: true,
            num_samples: 0,
            source,
        };

        q.set_frequency(fc)?;
        q.reset();
        Ok(q)
    }

    /// Reset object internals
    pub fn reset(&mut self) {
        // placeholder for future functionality
    }

    /// Set internal object identifier
    pub fn set_id(&mut self, id: SourceId) {
        self.id = id;
    }

    /// Get internal object identifier
    pub fn get_id(&self) -> SourceId {
        self.id
    }

    /// Enable source generation
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable source generation
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if source is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get number of samples generated by the object so far
    pub fn get_num_samples(&self) -> u64 {
        self.num_samples
    }

    /// Set signal gain in dB
    pub fn set_gain(&mut self, gain_db: f32) {
        self.gain = 10.0f32.powf(gain_db / 20.0);
    }

    /// Get signal gain in dB
    pub fn get_gain(&self) -> f32 {
        20.0 * self.gain.log10()
    }

    /// Get center frequency of signal applied by channelizer alignment
    fn get_frequency_index(&self) -> f32 {
        let s = if self.index < self.m_channels / 2 {
            0.0
        } else {
            -1.0
        };
        (self.index as f32) / (self.m_channels as f32) + s
    }

    /// Set signal center frequency
    pub fn set_frequency(&mut self, fc: f32) -> Result<()> {
        if fc < -0.5 || fc > 0.5 {
            return Err(Error::Config(
                "invalid frequency offset; must be in [-0.5, 0.5]".into(),
            ));
        }

        self.fc = fc;

        // set channelizer index appropriately
        self.index = ((if fc < 0.0 { fc + 1.0 } else { fc }) * self.m_channels as f32).round()
            as usize
            % self.m_channels;

        // compute frequency applied by channelizer alignment
        let fc_index = self.get_frequency_index();

        // compute residual frequency needed by mixer
        let fc_mixer = fc - fc_index;

        // apply mixer frequency (in radians), scaled by resampling ratio
        self.mixer.set_frequency(
            2.0 * PI * fc_mixer * (self.m_channels as f32) / (self.p_channels as f32),
        );

        Ok(())
    }

    /// Get signal center frequency
    pub fn get_frequency(&self) -> f32 {
        let fc_index = self.get_frequency_index();
        let fc_mixer = self.mixer.get_frequency() * (self.p_channels as f32)
            / (2.0 * PI * self.m_channels as f32);
        fc_index + fc_mixer
    }

    /// Get source type
    pub fn get_type(&self) -> QSourceType {
        match &self.source {
            SourceState::User(_) => QSourceType::User,
            SourceState::Tone => QSourceType::Tone,
            SourceState::Chirp(_) => QSourceType::Chirp,
            SourceState::Noise => QSourceType::Noise,
            SourceState::Modem(_) => QSourceType::Modem,
            SourceState::Fsk(_) => QSourceType::Fsk,
            SourceState::Gmsk(_) => QSourceType::Gmsk,
        }
    }

    /// Get number of channels
    pub fn get_m(&self) -> usize {
        self.m_channels
    }

    /// Get number of analysis channels
    pub fn get_p(&self) -> usize {
        self.p_channels
    }

    /// Generate a single sample
    pub fn generate(&mut self) -> Result<Complex32> {
        let sample = match &mut self.source {
            SourceState::User(ref user) => {
                let mut buf = [Complex32::new(0.0, 0.0); 1];
                user.callback.lock().unwrap().generate(&mut buf)?;
                buf[0]
            }
            SourceState::Tone => Complex32::new(1.0, 0.0),
            SourceState::Chirp(ref mut chirp) => {
                let sample = chirp.nco.cexp();
                chirp.nco.adjust_frequency(chirp.df);
                chirp.nco.step();
                chirp.timer -= 1;
                if chirp.timer == 0 {
                    chirp.timer = chirp.num;
                    if chirp.single {
                        self.enabled = false;
                    }
                    chirp
                        .nco
                        .set_frequency(if chirp.negate { PI } else { -PI });
                }
                sample
            }
            SourceState::Noise => {
                Complex32::new(randnf(), randnf()) * std::f32::consts::FRAC_1_SQRT_2
            }
            SourceState::Modem(ref mut modem) => {
                let mut buf = [Complex32::new(0.0, 0.0); 1];
                modem.symstream.write_samples(&mut buf)?;
                buf[0] * std::f32::consts::FRAC_1_SQRT_2
            }
            SourceState::Fsk(ref mut fsk) => {
                // modulate new symbol when index is 0
                if fsk.index == 0 {
                    let sym = (crate::random::randf() * (fsk.mask + 1) as f32) as usize & fsk.mask;
                    fsk.modulator.modulate(sym, &mut fsk.buf)?;
                }
                let sample = fsk.buf[fsk.index];
                fsk.index = (fsk.index + 1) % fsk.buf.len();
                sample
            }
            SourceState::Gmsk(ref mut gmsk) => {
                // modulate new symbol when index is 0
                if gmsk.index == 0 {
                    let sym = if crate::random::randf() > 0.5 { 1u8 } else { 0u8 };
                    gmsk.modulator.modulate(sym, &mut gmsk.buf)?;
                }
                let sample = gmsk.buf[gmsk.index] * std::f32::consts::FRAC_1_SQRT_2;
                gmsk.index = (gmsk.index + 1) & 1; // reset index every 2 samples
                sample
            }
        };

        let sample = if !self.enabled {
            Complex32::new(0.0, 0.0)
        } else {
            sample
        };

        // mix sample up
        let mixed = self.mixer.mix_up(sample);
        self.mixer.step();

        Ok(mixed)
    }

    /// Generate a block of samples and write into parent channelizer buffer
    pub fn generate_into(&mut self, buf: &mut [Complex32]) -> Result<()> {
        let p2 = self.p_channels / 2;

        // fill input buffer for channelizer
        for i in 0..p2 {
            self.buf_time[i] = self.generate()?;
        }

        // run analysis channelizer
        self.ch.execute_analyzer(&self.buf_time, &mut self.buf_freq)?;

        // aggregate gain
        let g = self.gain * self.gain_ch;

        // copy upper frequency band (base index = self.index)
        let base_index = self.index;
        for i in 0..p2 {
            buf[(base_index + i) % self.m_channels] += self.buf_freq[i] * g;
        }

        // copy lower frequency band
        let mut base_index = self.index;
        while base_index <= p2 {
            base_index += self.m_channels;
        }
        base_index -= p2;
        for i in 0..p2 {
            buf[(base_index + i) % self.m_channels] += self.buf_freq[i + p2] * g;
        }

        self.num_samples += p2 as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_qsourcecf_config)]
    fn test_qsourcecf_config() {
        // check invalid function calls
        assert!(QSource::new(0, 12, 60.0, 0.0, 0.2, 10.0, QSourceConfig::Tone).is_err()); // too few subcarriers
        assert!(QSource::new(17, 12, 60.0, 0.0, 0.2, 10.0, QSourceConfig::Tone).is_err()); // odd-numbered subcarriers
        assert!(QSource::new(64, 0, 60.0, 0.0, 0.2, 10.0, QSourceConfig::Tone).is_err()); // filter semi-length too small
        assert!(QSource::new(64, 12, 60.0, 0.6, 0.2, 10.0, QSourceConfig::Tone).is_err()); // center frequency out of range
        assert!(QSource::new(64, 12, 60.0, -0.6, 0.2, 10.0, QSourceConfig::Tone).is_err()); // center frequency out of range
        assert!(QSource::new(64, 12, 60.0, 0.0, -0.1, 10.0, QSourceConfig::Tone).is_err()); // bandwidth out of range
        assert!(QSource::new(64, 12, 60.0, 0.0, 1.1, 10.0, QSourceConfig::Tone).is_err()); // bandwidth out of range
    }

    #[test]
    fn test_qsourcecf_tone() {
        let mut q = QSource::new(64, 12, 60.0, 0.0, 0.2, 0.0, QSourceConfig::Tone).unwrap();

        // generate some samples
        for _ in 0..100 {
            let sample = q.generate().unwrap();
            // tone should have magnitude ~1
            assert!((sample.norm() - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_qsourcecf_noise() {
        let mut q = QSource::new(64, 12, 60.0, 0.0, 0.2, 0.0, QSourceConfig::Noise).unwrap();

        // generate samples and check they're not all zero
        let mut sum = 0.0f32;
        for _ in 0..1000 {
            let sample = q.generate().unwrap();
            sum += sample.norm_sqr();
        }
        // average power should be around 1
        let avg_power = sum / 1000.0;
        assert!(avg_power > 0.5 && avg_power < 2.0);
    }

    #[test]
    fn test_qsourcecf_chirp() {
        let config = QSourceConfig::Chirp { duration: 100.0, negate: false, single: false };
        let mut q = QSource::new(64, 12, 60.0, 0.0, 0.2, 0.0, config).unwrap();

        // generate some samples
        for _ in 0..100 {
            let sample = q.generate().unwrap();
            // chirp should have magnitude ~1
            assert!((sample.norm() - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_qsourcecf_enable_disable() {
        let mut q = QSource::new(64, 12, 60.0, 0.0, 0.2, 0.0, QSourceConfig::Tone).unwrap();

        assert!(q.is_enabled());

        q.disable();
        assert!(!q.is_enabled());

        // disabled source should produce zero
        let sample = q.generate().unwrap();
        assert_eq!(sample, Complex32::new(0.0, 0.0));

        q.enable();
        assert!(q.is_enabled());

        // enabled source should produce non-zero
        let sample = q.generate().unwrap();
        assert!(sample.norm() > 0.0);
    }

    #[test]
    fn test_qsourcecf_gmsk() {
        let config = QSourceConfig::Gmsk { m: 3, bt: 0.25 };
        let mut q = QSource::new(64, 12, 60.0, 0.0, 0.2, 0.0, config).unwrap();

        assert_eq!(q.get_type(), QSourceType::Gmsk);

        // generate samples and check they're on unit circle (scaled by 1/sqrt(2))
        for _ in 0..100 {
            let sample = q.generate().unwrap();
            // GMSK output is unit circle scaled by 1/sqrt(2) ≈ 0.707
            let expected_mag = std::f32::consts::FRAC_1_SQRT_2;
            assert!(
                (sample.norm() - expected_mag).abs() < 0.1,
                "expected magnitude ~{}, got {}",
                expected_mag,
                sample.norm()
            );
        }
    }

    #[test]
    fn test_qsourcecf_fsk() {
        let config = QSourceConfig::Fsk { m: 2, k: 4 }; // 2 bits/symbol, 4 samples/symbol
        let mut q = QSource::new(64, 12, 60.0, 0.0, 0.2, 0.0, config).unwrap();

        assert_eq!(q.get_type(), QSourceType::Fsk);

        // generate samples and check they're on unit circle
        for _ in 0..100 {
            let sample = q.generate().unwrap();
            // FSK output is on unit circle (no scaling)
            assert!(
                (sample.norm() - 1.0).abs() < 0.1,
                "expected magnitude ~1.0, got {}",
                sample.norm()
            );
        }
    }

    // Test user callback source
    #[derive(Clone)]
    struct TestSineSource {
        phase: f32,
        freq: f32,
    }

    impl QSourceCallback for TestSineSource {
        fn generate(&mut self, output: &mut [Complex32]) -> Result<()> {
            for sample in output.iter_mut() {
                *sample = Complex32::new(self.phase.cos(), self.phase.sin());
                self.phase += self.freq;
            }
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn QSourceCallback> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_qsourcecf_user() {
        let source = TestSineSource {
            phase: 0.0,
            freq: 0.1,
        };
        let config = QSourceConfig::User(Box::new(source));
        let mut q = QSource::new(64, 12, 60.0, 0.0, 0.2, 0.0, config).unwrap();

        assert_eq!(q.get_type(), QSourceType::User);

        // generate samples and check they're on unit circle
        for _ in 0..100 {
            let sample = q.generate().unwrap();
            assert!(
                (sample.norm() - 1.0).abs() < 0.1,
                "expected magnitude ~1.0, got {}",
                sample.norm()
            );
        }
    }
}
