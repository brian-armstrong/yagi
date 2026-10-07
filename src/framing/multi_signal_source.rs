// msource : multi-signal source generator

use super::signal_source::{SignalSource, SignalSourceCallback, SignalSourceConfig, SignalSourceType, SourceId};
use crate::error::{Error, Result};
use crate::modem::ModulationScheme;
use crate::multichannel::{ChannelizerType, OversampledPolyphaseChannelizer};

use num_complex::Complex32;

/// Generates a test signal made of many synthetic sources.
///
/// Each source is a tone, chirp, noise band, modulated signal, or user
/// callback with its own center frequency, bandwidth, and gain. A synthesis
/// channelizer with `num_channels` channels sums them into one complex
/// stream. See [`SignalSource`] for how a single source is generated.
///
/// Adding a source returns an identifier for the later calls. Identifiers
/// count up from zero and are not reused after [`Self::remove`].
///
/// A source's gain sets its level per unit of bandwidth, not its total power.
/// With gain `G` dB and a bandwidth `bw` that rounds up only a little, the
/// output power of a source is about:
///
/// - tone, chirp, noise, modem, GMSK: `G + 10*log10(bw)` dB
/// - FSK: `G + 10*log10(bw) + 3` dB
/// - user callback: `G + 10*log10(bw)` dB plus the callback's sample power in dB
///
/// Bandwidths are rounded up to a nonzero multiple of `2 / num_channels`, so
/// a tone with `bw` of zero has power `G + 10*log10(2 / num_channels)` dB.
/// With the default 1200 channels, the rounding is small for most
/// bandwidths. With few channels, a source can be noticeably wider than
/// requested and off center.
///
/// Output is generated in blocks of `num_channels / 2` samples, which
/// [`Self::write_samples`] hands out as needed. A change to a source takes
/// effect at the next block.
#[derive(Clone, Debug)]
#[doc(alias = "MSource")]
pub struct MultiSignalSource {
    /// Array of signal sources
    sources: Vec<SignalSource>,
    /// ID counter for assigning unique IDs
    id_counter: SourceId,
    /// Channelizer size (number of channels)
    m_channels: usize,
    /// Channelizer filter semi-length
    m: usize,
    /// Channelizer filter stop-band suppression (dB)
    as_: f32,
    /// Synthesis channelizer
    ch: OversampledPolyphaseChannelizer<Complex32>,
    /// Frequency domain buffer
    buf_freq: Vec<Complex32>,
    /// Time domain buffer
    buf_time: Vec<Complex32>,
    /// Output buffer read index
    read_index: usize,
    /// Total number of samples generated
    num_samples: u64,
}

impl MultiSignalSource {
    /// Creates a generator with no sources.
    ///
    /// `num_channels` sets the frequency resolution, with a bandwidth step of
    /// `2 / num_channels`. `filter_semi_length` and `stopband_attenuation`,
    /// in dB, configure the channelizer filters, including those of every
    /// source.
    ///
    /// # Errors
    ///
    /// Returns an error when `num_channels` is odd or less than two,
    /// `filter_semi_length` is zero, or the channelizer design rejects
    /// `stopband_attenuation`.
    pub fn new(num_channels: usize, filter_semi_length: usize, stopband_attenuation: f32) -> Result<Self> {
        if num_channels < 2 {
            return Err(Error::Config("number of subcarriers must be at least 2".into()));
        }
        if !num_channels.is_multiple_of(2) {
            return Err(Error::Config("number of subcarriers must be even".into()));
        }
        if filter_semi_length == 0 {
            return Err(Error::Config("filter semi-length must be greater than zero".into()));
        }

        let ch = OversampledPolyphaseChannelizer::new_kaiser(
            ChannelizerType::Synthesizer,
            num_channels,
            filter_semi_length,
            stopband_attenuation,
        )?;

        let buf_freq = vec![Complex32::new(0.0, 0.0); num_channels];
        let buf_time = vec![Complex32::new(0.0, 0.0); num_channels / 2];

        Ok(Self {
            sources: Vec::new(),
            id_counter: 0,
            m_channels: num_channels,
            m: filter_semi_length,
            as_: stopband_attenuation,
            ch,
            buf_freq,
            buf_time,
            read_index: num_channels / 2, // indicate buffer is empty
            num_samples: 0,
        })
    }

    /// Creates a generator with 1200 channels, a filter semi-length of 4, and
    /// 60 dB stopband attenuation.
    ///
    /// This never returns an error.
    pub fn new_default() -> Result<Self> {
        Self::new(1200, 4, 60.0)
    }

    /// Discards the rest of the current output block.
    ///
    /// The next output starts a new block. The sources and
    /// [`Self::num_samples`] are unchanged.
    pub fn reset(&mut self) {
        self.read_index = self.m_channels / 2;
    }

    /// Returns the number of output samples generated so far.
    ///
    /// Output is generated a block at a time, so this can exceed the number
    /// that [`Self::write_samples`] has handed out.
    pub fn num_samples(&self) -> u64 {
        self.num_samples
    }

    /// Returns the number of sources, enabled or not.
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Add a source to the list and return its ID
    fn add_source(&mut self, mut source: SignalSource) -> SourceId {
        let id = self.id_counter;
        source.set_id(id);
        self.sources.push(source);
        self.id_counter += 1;
        id
    }

    /// Find source index by ID, returns None if not found
    fn find(&self, id: SourceId) -> Option<usize> {
        self.sources.iter().position(|s| s.id() == id)
    }

    /// Get mutable reference to source by ID
    fn source_mut(&mut self, id: SourceId) -> Result<&mut SignalSource> {
        match self.find(id) {
            Some(idx) => Ok(&mut self.sources[idx]),
            None => Err(Error::Range(format!("source with id {} not found", id))),
        }
    }

    /// Get reference to source by ID
    fn source(&self, id: SourceId) -> Result<&SignalSource> {
        match self.find(id) {
            Some(idx) => Ok(&self.sources[idx]),
            None => Err(Error::Range(format!("source with id {} not found", id))),
        }
    }

    /// Adds a tone and returns its identifier.
    ///
    /// `center_frequency` is in cycles per sample, in `[-0.5, 0.5]`.
    /// `bandwidth` is in `[0, 1]` and, for a tone, sets only the power.
    /// `gain_db` is in dB. The power is about `gain_db + 10*log10(w)` dB,
    /// where `w` is `bandwidth` rounded up as described on the type.
    ///
    /// # Errors
    ///
    /// Returns an error when `center_frequency` or `bandwidth` is outside its range.
    pub fn add_tone(&mut self, center_frequency: f32, bandwidth: f32, gain_db: f32) -> Result<SourceId> {
        let source = SignalSource::new(
            self.m_channels,
            self.m,
            self.as_,
            center_frequency,
            bandwidth,
            gain_db,
            SignalSourceConfig::Tone,
        )?;
        Ok(self.add_source(source))
    }

    /// Adds a linear frequency sweep and returns its identifier.
    ///
    /// The sweep covers `bandwidth` around `center_frequency`, both in cycles
    /// per sample, from low to high frequency. `gain_db` is in dB. `duration`
    /// is the approximate sweep period in samples, and
    /// `duration * bandwidth` must be at least 0.5. `negate`
    /// sweeps from high to low instead. `single` disables the source after
    /// one sweep.
    ///
    /// # Errors
    ///
    /// Returns an error when `center_frequency` is outside `[-0.5, 0.5]` or
    /// `bandwidth` is outside `[0, 1]`.
    ///
    /// # Panics
    ///
    /// A `duration * bandwidth` below 0.5 is accepted, but generating output
    /// then panics when overflow checks are enabled. Without overflow checks, such
    /// a chirp does not sweep.
    pub fn add_chirp(
        &mut self,
        center_frequency: f32,
        bandwidth: f32,
        gain_db: f32,
        duration: f32,
        negate: bool,
        single: bool,
    ) -> Result<SourceId> {
        let config = SignalSourceConfig::Chirp { duration, negate, single };
        let source =
            SignalSource::new(self.m_channels, self.m, self.as_, center_frequency, bandwidth, gain_db, config)?;
        Ok(self.add_source(source))
    }

    /// Adds a band of white Gaussian noise and returns its identifier.
    ///
    /// The band is `bandwidth` wide around `center_frequency`, both in cycles
    /// per sample. `gain_db` is in dB, and the power is about
    /// `gain_db + 10*log10(bandwidth)` dB.
    ///
    /// # Errors
    ///
    /// Returns an error when `center_frequency` is outside `[-0.5, 0.5]` or
    /// `bandwidth` is outside `[0, 1]`.
    pub fn add_noise(&mut self, center_frequency: f32, bandwidth: f32, gain_db: f32) -> Result<SourceId> {
        let source = SignalSource::new(
            self.m_channels,
            self.m,
            self.as_,
            center_frequency,
            bandwidth,
            gain_db,
            SignalSourceConfig::Noise,
        )?;
        Ok(self.add_source(source))
    }

    /// Adds a signal of random linearly modulated symbols and returns its
    /// identifier.
    ///
    /// # Arguments
    ///
    /// - `center_frequency`: center frequency in cycles per sample, in `[-0.5, 0.5]`.
    /// - `symbol_rate`: symbols per sample, in `[0, 0.5]`. The signal
    ///   occupies about `symbol_rate * (1 + rolloff)`.
    /// - `gain_db`: in dB. The power is about `gain_db + 10*log10(symbol_rate)` dB.
    /// - `filter_semi_length`: pulse-shaping filter semi-length in symbols. Must be nonzero.
    /// - `rolloff`: rolloff factor, in `[0, 1]`.
    ///
    /// The source runs at two samples per symbol, so its bandwidth is
    /// `2 * symbol_rate`.
    ///
    /// # Errors
    ///
    /// Returns an error when a parameter is outside its range,
    /// `modulation_scheme` has no built-in constellation, or the pulse shape
    /// rejects `filter_semi_length` and `rolloff`.
    pub fn add_modem(
        &mut self,
        center_frequency: f32,
        symbol_rate: f32,
        gain_db: f32,
        modulation_scheme: ModulationScheme,
        filter_semi_length: usize,
        rolloff: f32,
    ) -> Result<SourceId> {
        // create object with double the bandwidth to account for 2 samples/symbol
        let config = SignalSourceConfig::Modem { scheme: modulation_scheme, m: filter_semi_length, beta: rolloff };
        let source =
            SignalSource::new(self.m_channels, self.m, self.as_, center_frequency, 2.0 * symbol_rate, gain_db, config)?;
        Ok(self.add_source(source))
    }

    /// Adds a signal of random FSK symbols and returns its identifier.
    ///
    /// # Arguments
    ///
    /// - `center_frequency`: center frequency in cycles per sample, in `[-0.5, 0.5]`.
    /// - `half_bandwidth`: half the source bandwidth, in cycles per sample, in `[0, 0.5]`.
    /// - `gain_db`: in dB. The power is about `gain_db + 10*log10(half_bandwidth) + 3` dB,
    ///   3 dB more than a modem or GMSK source with the same `half_bandwidth`.
    /// - `bits_per_symbol`: bits per symbol.
    /// - `samples_per_symbol`: source samples per symbol, so the symbol rate
    ///   is `2 * half_bandwidth / samples_per_symbol` symbols per sample.
    ///
    /// # Errors
    ///
    /// Returns an error when `center_frequency` or `half_bandwidth` is outside
    /// its range, or the FSK modulator rejects `bits_per_symbol` and
    /// `samples_per_symbol`.
    pub fn add_fsk(
        &mut self,
        center_frequency: f32,
        half_bandwidth: f32,
        gain_db: f32,
        bits_per_symbol: usize,
        samples_per_symbol: usize,
    ) -> Result<SourceId> {
        // create object with double the bandwidth to account for k samples/symbol
        let config = SignalSourceConfig::Fsk { m: bits_per_symbol, k: samples_per_symbol };
        let source = SignalSource::new(
            self.m_channels,
            self.m,
            self.as_,
            center_frequency,
            2.0 * half_bandwidth,
            gain_db,
            config,
        )?;
        Ok(self.add_source(source))
    }

    /// Adds a signal of random GMSK bits and returns its identifier.
    ///
    /// # Arguments
    ///
    /// - `center_frequency`: center frequency in cycles per sample, in `[-0.5, 0.5]`.
    /// - `bit_rate`: bits per sample, in `[0, 0.5]`.
    /// - `gain_db`: in dB. The power is about `gain_db + 10*log10(bit_rate)` dB.
    /// - `filter_semi_length`: filter semi-length in bits. Must be nonzero.
    /// - `bandwidth_time_product`: bandwidth-time product, in `(0, 1)`.
    ///
    /// The source runs at two samples per bit, so its bandwidth is `2 * bit_rate`.
    ///
    /// # Errors
    ///
    /// Returns an error when a parameter is outside its range.
    pub fn add_gmsk(
        &mut self,
        center_frequency: f32,
        bit_rate: f32,
        gain_db: f32,
        filter_semi_length: usize,
        bandwidth_time_product: f32,
    ) -> Result<SourceId> {
        // create object with double the bandwidth to account for 2 samples/symbol
        let config = SignalSourceConfig::Gmsk { m: filter_semi_length, bt: bandwidth_time_product };
        let source =
            SignalSource::new(self.m_channels, self.m, self.as_, center_frequency, 2.0 * bit_rate, gain_db, config)?;
        Ok(self.add_source(source))
    }

    /// Adds a source whose samples come from `callback` and returns its
    /// identifier.
    ///
    /// The callback runs at the source's own rate, `bandwidth` times the output rate
    /// after rounding, and its full band is placed `bandwidth` wide around `center_frequency`.
    /// `center_frequency` and `bandwidth` are in cycles per sample, in `[-0.5, 0.5]` and `[0, 1]`.
    /// `gain_db` is in dB. The power is about `gain_db + 10*log10(bandwidth)` dB plus the
    /// callback's sample power in dB.
    ///
    /// # Errors
    ///
    /// Returns an error when `center_frequency` or `bandwidth` is outside its range.
    pub fn add_user<C: SignalSourceCallback + 'static>(
        &mut self,
        center_frequency: f32,
        bandwidth: f32,
        gain_db: f32,
        callback: C,
    ) -> Result<SourceId> {
        let config = SignalSourceConfig::User(Box::new(callback));
        let source =
            SignalSource::new(self.m_channels, self.m, self.as_, center_frequency, bandwidth, gain_db, config)?;
        Ok(self.add_source(source))
    }

    /// Removes the source identified by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id`, or `gain_db` is
    /// nonfinite or overflows the linear gain.
    pub fn remove(&mut self, id: SourceId) -> Result<()> {
        match self.find(id) {
            Some(idx) => {
                self.sources.remove(idx);
                Ok(())
            }
            None => Err(Error::Range(format!("source with id {} not found", id))),
        }
    }

    /// Enables the source identified by `id`.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id`.
    pub fn enable(&mut self, id: SourceId) -> Result<()> {
        self.source_mut(id)?.enable();
        Ok(())
    }

    /// Disables the source identified by `id`, so it contributes nothing.
    ///
    /// The source keeps advancing while disabled.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id`.
    pub fn disable(&mut self, id: SourceId) -> Result<()> {
        self.source_mut(id)?.disable();
        Ok(())
    }

    /// Sets the gain of a source in dB.
    ///
    /// See the type documentation for how the gain relates to power.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id`.
    pub fn set_gain(&mut self, id: SourceId, gain_db: f32) -> Result<()> {
        self.source_mut(id)?.set_gain(gain_db)?;
        Ok(())
    }

    /// Returns the gain of a source in dB.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id`.
    pub fn gain(&self, id: SourceId) -> Result<f32> {
        Ok(self.source(id)?.gain())
    }

    /// Sets the center frequency of a source in cycles per sample.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id` or `frequency` is
    /// outside `[-0.5, 0.5]`.
    pub fn set_frequency(&mut self, id: SourceId, frequency: f32) -> Result<()> {
        self.source_mut(id)?.set_frequency(frequency)
    }

    /// Returns the center frequency of a source in cycles per sample.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id`.
    pub fn frequency(&self, id: SourceId) -> Result<f32> {
        Ok(self.source(id)?.frequency())
    }

    /// Returns the number of samples a source has generated, at the source's
    /// own rate.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id`.
    pub fn num_samples_source(&self, id: SourceId) -> Result<u64> {
        Ok(self.source(id)?.num_samples())
    }

    /// Returns the kind of waveform a source generates.
    ///
    /// # Errors
    ///
    /// Returns an error when no source has identifier `id`.
    pub fn source_type(&self, id: SourceId) -> Result<SignalSourceType> {
        Ok(self.source(id)?.source_type())
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

    /// Fills `output` with the next samples.
    ///
    /// The output continues across calls, so any block size works.
    ///
    /// # Errors
    ///
    /// Returns an error when a user callback returns one. The samples of `output`
    /// before the failed block are already written.
    ///
    /// # Panics
    ///
    /// Panics under the conditions given on [`SignalSource::generate`].
    pub fn write_samples(&mut self, output: &mut [Complex32]) -> Result<()> {
        let m2 = self.m_channels / 2;
        for sample in output.iter_mut() {
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
    use crate::fft::SpectralPeriodogram;
    use crate::utility::test_helpers::{validate_psd_spgramcf, PsdRegion};
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_msourcecf_config)]
    fn test_msourcecf_config() {
        // check invalid configurations
        assert!(MultiSignalSource::new(0, 12, 60.0).is_err()); // too few subcarriers
        assert!(MultiSignalSource::new(17, 12, 60.0).is_err()); // odd-numbered subcarriers
        assert!(MultiSignalSource::new(64, 0, 60.0).is_err()); // filter semi-length too small

        // create proper object and test configurations
        let mut q = MultiSignalSource::new(64, 12, 60.0).unwrap();

        // try to configure signals with invalid IDs
        assert!(q.remove(12345).is_err());
        assert!(q.enable(12345).is_err());
        assert!(q.disable(12345).is_err());
        assert!(q.set_gain(12345, 0.0).is_err());
        assert!(q.gain(12345).is_err());
        assert!(q.set_frequency(12345, 0.0).is_err());
        assert!(q.frequency(12345).is_err());

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
        let mut q = MultiSignalSource::new(240, 12, 60.0).unwrap();
        let id_tone = q.add_tone(-0.123456, 0.0, 20.0).unwrap();
        let id_noise = q.add_noise(0.220780, 0.10, 0.0).unwrap();

        // check center frequency of tone
        let fc = q.frequency(id_tone).unwrap();
        assert!((fc - (-0.123456)).abs() < 1e-5);

        // check center frequency of noise signal
        let fc = q.frequency(id_noise).unwrap();
        assert!((fc - 0.220780).abs() < 1e-5);

        // check gain of tone
        let gain = q.gain(id_tone).unwrap();
        assert!((gain - 20.0).abs() < 0.1);

        // check gain of noise signal
        let gain = q.gain(id_noise).unwrap();
        assert!(gain.abs() < 0.1);

        // remove tone
        q.remove(id_tone).unwrap();

        // disable noise signal
        q.disable(id_noise).unwrap();

        // set frequency of noise signal
        q.set_frequency(id_noise, 0.33333).unwrap();
        let fc = q.frequency(id_noise).unwrap();
        assert!((fc - 0.33333).abs() < 1e-5);

        // set gain of noise signal
        q.set_gain(id_noise, 30.0).unwrap();
        let gain = q.gain(id_noise).unwrap();
        assert!((gain - 30.0).abs() < 0.1);

        // assert buffer is zeros
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert_eq!(sum, 0.0);

        // enable noise signal and check power spectral density
        q.enable(id_noise).unwrap();

        let nfft = 2400;
        let mut spgram = SpectralPeriodogram::<Complex32>::from_nfft(nfft).unwrap();

        while q.num_samples() < 192000 {
            q.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        #[rustfmt::skip]
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
        let mut q = MultiSignalSource::new_default().unwrap();

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
        let mut q = MultiSignalSource::new_default().unwrap();

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
        let mut q = MultiSignalSource::new_default().unwrap();

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
        let mut q = MultiSignalSource::new_default().unwrap();

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
        let mut q = MultiSignalSource::new_default().unwrap();

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

    impl SignalSourceCallback for TestConstantSource {
        fn generate(&mut self, output: &mut [Complex32]) -> crate::error::Result<()> {
            for sample in output.iter_mut() {
                *sample = self.value;
            }
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn SignalSourceCallback> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_msourcecf_user_basic() {
        let mut q = MultiSignalSource::new_default().unwrap();

        // add a user-defined source
        let source = TestConstantSource { value: Complex32::new(1.0, 0.0) };
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
        let mut q = MultiSignalSource::new(64, 12, 60.0).unwrap();

        // add multiple sources
        let id0 = q.add_noise(0.0, 1.0, -40.0).unwrap();
        let id1 = q.add_tone(-0.4, 0.0, 20.0).unwrap();
        let id2 = q.add_tone(-0.3, 0.0, 10.0).unwrap();

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(q.num_sources(), 3);

        // generate samples
        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        q.write_samples(&mut buf).unwrap();

        // verify samples are non-zero
        let sum: f32 = buf.iter().map(|x| x.norm_sqr()).sum();
        assert!(sum > 0.0);

        // remove a source
        q.remove(id1).unwrap();
        assert_eq!(q.num_sources(), 2);
    }

    #[test]
    fn test_msourcecf_spectrum() {
        // This test verifies msource produces signals at expected frequencies/powers
        let mut gen = MultiSignalSource::new_default().unwrap();

        // Add signals matching the firpfbchr test
        gen.add_noise(0.0, 1.0, -60.0).unwrap(); // wide-band noise floor
        gen.add_noise(-0.30, 0.10, -20.0).unwrap(); // narrow-band noise at -0.30
        gen.add_noise(0.08, 0.01, -30.0).unwrap(); // very narrow-band noise at 0.08
        gen.add_modem(0.1875, 0.065, -20.0, ModulationScheme::Qpsk, 12, 0.3).unwrap();

        let nfft = 2400;
        let mut spgram = SpectralPeriodogram::<Complex32>::from_nfft(nfft).unwrap();

        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];

        // Generate samples
        for _ in 0..(1 << 12) {
            gen.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        let psd = spgram.psd();

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
        assert!(
            floor_min > -65.0 && floor_max < -55.0,
            "Noise floor should be ~-60 dB, got [{}, {}]",
            floor_min,
            floor_max
        );

        // Verify narrow-band noise is around -20 dB
        assert!(
            nb_min > -25.0 && nb_max < -15.0,
            "Narrow-band noise should be ~-20 dB, got [{}, {}]",
            nb_min,
            nb_max
        );

        // Verify very narrow-band noise peak is around -30 dB
        // (region includes noise floor at edges due to narrow bandwidth)
        assert!(vnb_max > -35.0 && vnb_max < -25.0, "Very narrow noise peak should be ~-30 dB, got max={}", vnb_max);

        // Verify modem signal is present and above noise floor
        // Note: actual power differs from gain setting due to modulation/filtering
        assert!(
            mod_max > floor_max + 10.0,
            "Modem signal should be above noise floor, got max={} vs floor={}",
            mod_max,
            floor_max
        );
    }

    #[test]
    #[autotest_annotate(autotest_msourcecf_tone)]
    fn test_msourcecf_tone() {
        let nfft = 2400;
        let num_samples = 192000;

        let mut spgram = SpectralPeriodogram::<Complex32>::from_nfft(nfft).unwrap();

        let mut gen = MultiSignalSource::new_default().unwrap();
        // add signals (fc, bw, gain)
        gen.add_noise(0.0, 1.0, -40.0).unwrap(); // wide-band noise
        gen.add_tone(-0.4, 0.0, 20.0).unwrap(); // tone
        gen.add_tone(-0.3, 0.0, 10.0).unwrap(); // tone
        gen.add_tone(-0.2, 0.0, 0.0).unwrap(); // tone
        gen.add_tone(-0.1, 0.0, -10.0).unwrap(); // tone
        gen.add_tone(0.0, 0.0, -20.0).unwrap(); // tone
        gen.add_tone(0.1, 0.0, -30.0).unwrap(); // tone

        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        while gen.num_samples() < num_samples as u64 {
            gen.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        #[rustfmt::skip]
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

        let mut spgram = SpectralPeriodogram::<Complex32>::from_nfft(nfft).unwrap();

        let mut gen = MultiSignalSource::new_default().unwrap();
        // add signals (fc, bw, gain, duration, negate, single)
        gen.add_noise(0.0, 1.0, -40.0).unwrap(); // wide-band noise
        gen.add_chirp(0.0, 0.60, 20.0, (num_samples as f32) * 0.9, false, true).unwrap();

        let mut buf = vec![Complex32::new(0.0, 0.0); 1024];
        while gen.num_samples() < num_samples {
            gen.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        #[rustfmt::skip]
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

    impl SignalSourceCallback for PulseTrainSource {
        fn generate(&mut self, output: &mut [Complex32]) -> crate::error::Result<()> {
            for sample in output.iter_mut() {
                *sample = if self.counter == 0 { Complex32::new(1.0, 0.0) } else { Complex32::new(0.0, 0.0) };
                self.counter = (self.counter + 1) % 8;
            }
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn SignalSourceCallback> {
            Box::new(self.clone())
        }
    }

    #[test]
    #[autotest_annotate(autotest_msourcecf_aggregate)]
    fn test_msourcecf_aggregate() {
        // msource parameters
        let ms = ModulationScheme::Qpsk; // linear modulation scheme
        let m = 12; // modulation filter semi-length
        let beta = 0.30; // modulation filter excess bandwidth factor
        let bt = 0.35; // GMSK filter bandwidth-time factor

        // spectral periodogram options
        let nfft = 2400;
        let num_samples: u64 = 192000;

        let mut spgram = SpectralPeriodogram::<Complex32>::from_nfft(nfft).unwrap();

        let buf_len = 1024;
        let mut buf = vec![Complex32::new(0.0, 0.0); buf_len];

        // create multi-signal source generator
        let mut gen = MultiSignalSource::new_default().unwrap();

        // add signals     (fc,    bw,    gain, {options})
        gen.add_noise(0.00, 1.00, -40.0).unwrap(); // wide-band noise
        gen.add_tone(-0.45, 0.00, 20.0).unwrap(); // tone
        gen.add_fsk(-0.33, 0.05, -10.0, 3, 16).unwrap(); // FSK
        gen.add_gmsk(-0.20, 0.05, 0.0, m, bt).unwrap(); // modulated data (GMSK)
        gen.add_noise(-0.05, 0.10, 0.0).unwrap(); // narrow-band noise
        gen.add_chirp(0.07, 0.07, 20.0, 8000.0, false, false).unwrap(); // chirp
        gen.add_modem(0.20, 0.10, 0.0, ms, m, beta).unwrap(); // modulated data (linear)
        gen.add_user(0.40, 0.15, -10.0, PulseTrainSource { counter: 0 }).unwrap(); // tones

        while gen.num_samples() < num_samples {
            gen.write_samples(&mut buf).unwrap();
            spgram.write(&buf);
        }

        #[rustfmt::skip]
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

    #[test]
    #[autotest_annotate(autotest_msourcecf_copy)]
    fn test_msourcecf_copy() {
        let tol = 1.5;
        let ms = ModulationScheme::Qpsk;
        let m = 12;
        let beta = 0.30;
        let bt = 0.35;
        let nfft = 1200;
        let num_samples = 720_000;

        let mut periodogram_orig = SpectralPeriodogram::from_nfft(nfft).unwrap();
        let mut periodogram_copy = SpectralPeriodogram::from_nfft(nfft).unwrap();

        let mut gen_orig = MultiSignalSource::new_default().unwrap();
        gen_orig.add_noise(0.00, 1.00, -40.0).unwrap(); // wide-band noise
        gen_orig.add_tone(-0.45, 0.00, 20.0).unwrap(); // tone
        gen_orig.add_fsk(-0.33, 0.05, -10.0, 3, 16).unwrap(); // FSK
        gen_orig.add_gmsk(-0.20, 0.05, 0.0, m, bt).unwrap(); // GMSK
        gen_orig.add_noise(-0.05, 0.10, 0.0).unwrap(); // narrow-band noise
        gen_orig.add_chirp(0.07, 0.07, 20.0, 8000.0, false, false).unwrap(); // chirp
        gen_orig.add_modem(0.20, 0.10, 0.0, ms, m, beta).unwrap(); // linear modem

        // clone the configured generator
        let mut gen_copy = gen_orig.clone();

        let buf_len = 1024;
        let mut buf_orig = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_copy = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut total = 0;
        while total < num_samples {
            gen_orig.write_samples(&mut buf_orig).unwrap();
            gen_copy.write_samples(&mut buf_copy).unwrap();
            periodogram_orig.write(&buf_orig);
            periodogram_copy.write(&buf_copy);
            total += buf_len;
        }

        let psd_orig = periodogram_orig.psd();
        let psd_copy = periodogram_copy.psd();
        assert_eq!(psd_orig.len(), psd_copy.len());

        // the two spectra agree within tolerance across all bins
        let mut max_err = 0.0f32;
        for (a, b) in psd_orig.iter().zip(psd_copy.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < tol, "clone PSD diverged from original by {max_err} dB");
    }
}
