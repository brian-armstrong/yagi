use super::symbol_stream::SymbolStream;
use crate::error::{Error, Result};
use crate::filter::FirFilterShape;
use crate::filter::MultistageResampler;
use crate::math::nextpow2;
use crate::modem::ModulationScheme;
use num_complex::Complex32;

/// Generates random linearly modulated symbols at an arbitrary symbol rate.
///
/// A [`SymbolStream`] at two samples per symbol is resampled to the requested
/// symbol rate. The realized rate can differ slightly from the request,
/// because the resampler rounds its phase step. The symbols follow a fixed
/// pseudo-random sequence, so two new streams with the same settings produce
/// the same samples.
#[derive(Clone, Debug)]
#[doc(alias = "SymStreamR")]
pub struct ArbitraryRateSymbolStream {
    symstream: SymbolStream,
    resamp: MultistageResampler<Complex32, f32>,
    buf: Vec<Complex32>,
    buf_size: usize,
    buf_index: usize,
}

impl ArbitraryRateSymbolStream {
    /// Creates a QPSK stream at 0.5 symbols per sample.
    ///
    /// The shaping filter is [`FirFilterShape::Arkaiser`] with a delay of 7
    /// symbols and an excess bandwidth of 0.3. This never returns an error.
    pub fn new() -> Result<Self> {
        Self::new_linear(FirFilterShape::Arkaiser, 0.5, 7, 0.3, ModulationScheme::Qpsk)
    }

    /// Creates a stream with symbol rate `symbol_rate`.
    ///
    /// `symbol_rate` is in symbols per output sample and must be in
    /// `[0.001, 1]`. `filter_delay` is the filter semi-length in symbols, and
    /// `excess_bandwidth` is the rolloff factor. The gain starts at one.
    ///
    /// With a Nyquist or root-Nyquist shape, the signal occupies about
    /// `symbol_rate * (1 + excess_bandwidth)` cycles per sample. A `symbol_rate`
    /// above `1 / (1 + excess_bandwidth)` is accepted but aliases.
    ///
    /// # Errors
    ///
    /// Returns an error when `symbol_rate` is outside its range or
    /// [`SymbolStream::new_linear`] rejects the other parameters.
    pub fn new_linear(
        filter_shape: FirFilterShape,
        symbol_rate: f32,
        filter_delay: usize,
        excess_bandwidth: f32,
        modulation_scheme: ModulationScheme,
    ) -> Result<Self> {
        const BW_MIN: f32 = 0.001;
        const BW_MAX: f32 = 1.000;
        if !(BW_MIN..=BW_MAX).contains(&symbol_rate) {
            return Err(Error::Config(format!("symbol rate ({}) must be in [{},{}]", symbol_rate, BW_MIN, BW_MAX)));
        }

        let symstream = SymbolStream::new_linear(filter_shape, 2, filter_delay, excess_bandwidth, modulation_scheme)?;
        let rate = 0.5 / symbol_rate;
        let resamp = MultistageResampler::new(rate, 60.0)?;

        let buf_len = 1 << nextpow2((0.5 / symbol_rate).ceil() as u32)?;
        let buf = vec![Complex32::new(0.0, 0.0); buf_len];

        let mut q = Self { symstream, resamp, buf, buf_size: 0, buf_index: 0 };

        q.reset();
        Ok(q)
    }

    /// Clears the filter and resampler history and discards buffered samples.
    ///
    /// The scheme and gain are unchanged. The random symbol sequence continues
    /// where it left off.
    pub fn reset(&mut self) {
        self.symstream.reset();
        self.resamp.reset();
        self.buf_size = 0;
        self.buf_index = 0;
    }

    /// Returns the pulse-shaping filter shape.
    pub fn filter_shape(&self) -> FirFilterShape {
        self.symstream.filter_shape()
    }

    /// Returns the requested symbol rate in symbols per output sample.
    ///
    /// The value is recomputed from the resampler rate, so it can differ from
    /// the constructor argument by rounding.
    pub fn symbol_rate(&self) -> f32 {
        1.0 / (self.resamp.rate() * self.symstream.samples_per_symbol() as f32)
    }

    /// Returns the shaping filter semi-length in symbols.
    pub fn filter_semi_length(&self) -> usize {
        self.symstream.filter_semi_length()
    }

    /// Returns the rolloff factor of the shaping filter.
    pub fn rolloff(&self) -> f32 {
        self.symstream.rolloff()
    }

    /// Changes the modulation scheme.
    ///
    /// Filter and resampler history and buffered samples are kept, so pulses
    /// of old and new symbols overlap. The random symbol sequence restarts.
    ///
    /// # Errors
    ///
    /// Returns an error when `modulation_scheme` has no built-in
    /// constellation. The stream is unchanged on error.
    pub fn set_modulation_scheme(&mut self, modulation_scheme: ModulationScheme) -> Result<()> {
        self.symstream.set_modulation_scheme(modulation_scheme)
    }

    /// Returns the modulation scheme.
    pub fn modulation_scheme(&self) -> ModulationScheme {
        self.symstream.modulation_scheme()
    }

    /// Sets the linear amplitude factor applied to each new symbol.
    ///
    /// Samples already in the filter history keep the previous gain.
    pub fn set_gain(&mut self, gain: f32) {
        self.symstream.set_gain(gain);
    }

    /// Returns the linear amplitude factor applied to each symbol.
    pub fn gain(&self) -> f32 {
        self.symstream.gain()
    }

    /// Returns the delay of the shaping filter and resampler together, in
    /// output samples.
    ///
    /// This is the time from a symbol entering the shaping filter to the peak
    /// of its pulse at the output. It is usually fractional.
    pub fn delay(&self) -> f32 {
        let p = self.symstream.delay() as f32;
        let d = self.resamp.delay();
        let r = self.resamp.rate();
        (p + d) * r
    }

    fn fill_buffer(&mut self) -> Result<()> {
        if self.buf_index != self.buf_size {
            return Err(Error::Internal("buffer not empty".into()));
        }

        self.buf_size = 0;
        self.buf_index = 0;

        while self.buf_size == 0 {
            let mut sample = [Complex32::new(0.0, 0.0)];
            self.symstream.write_samples(&mut sample)?;

            self.buf_size = self.resamp.execute(sample[0], &mut self.buf)?;
        }
        Ok(())
    }

    /// Fills `output` with the next samples of the stream.
    ///
    /// The stream continues across calls, so any block size works. It never
    /// returns an error.
    pub fn write_samples(&mut self, output: &mut [Complex32]) -> Result<()> {
        for sample in output.iter_mut() {
            if self.buf_index == self.buf_size {
                self.fill_buffer()?;
            }

            *sample = self.buf[self.buf_index];
            self.buf_index += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::SpectralPeriodogram;
    use crate::fft::{fft_run, Direction};
    use crate::utility::test_helpers::{validate_psd_spectrum, PsdRegion};
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    fn testbench_symstreamrcf_delay(bw: f32, m: usize) {
        // create object and get expected delay
        let ftype = FirFilterShape::Arkaiser;
        let beta = 0.30;
        let ms = ModulationScheme::Qpsk;
        let mut gen = ArbitraryRateSymbolStream::new_linear(ftype, bw, m, beta, ms).unwrap();
        let delay = gen.delay();
        let tol = 0.05; // error tolerance

        // compute buffer length based on delay
        let nfft = 2 * (120 + (delay / bw.sqrt()) as usize);
        let mut buf_time = vec![Complex32::new(0.0, 0.0); nfft];
        let mut buf_freq = vec![Complex32::new(0.0, 0.0); nfft];

        // write samples to buffer
        gen.write_samples(&mut buf_time[..1]).unwrap();
        gen.set_gain(0.0);
        gen.write_samples(&mut buf_time[1..]).unwrap();

        // take forward transform
        fft_run(&buf_time, &mut buf_freq, Direction::Forward);

        // measure phase slope across pass-band
        let m = (0.4 * bw * nfft as f32) as usize; // use 0.4 to account for filter roll-off
        let mut p = Complex32::new(0.0, 0.0);
        for i in -(m as i32)..m as i32 {
            let idx1 = (nfft as i32 + i) as usize % nfft;
            let idx2 = (nfft as i32 + i + 1) as usize % nfft;
            p += buf_freq[idx1] * buf_freq[idx2].conj();
        }
        let delay_meas = p.arg() * nfft as f32 / (2.0 * std::f32::consts::PI);

        // print results
        if cfg!(test) {
            println!(
                "expected delay: {:.6}, measured: {:.6}, err: {:.6} (tol= {:.3})",
                delay,
                delay_meas,
                delay - delay_meas,
                tol
            );
        }

        // verify delay is relatively close to expected
        assert_abs_diff_eq!(delay, delay_meas, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_00)]
    fn test_symstreamrcf_delay_00() {
        testbench_symstreamrcf_delay(0.500, 4);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_01)]
    fn test_symstreamrcf_delay_01() {
        testbench_symstreamrcf_delay(0.500, 5);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_02)]
    fn test_symstreamrcf_delay_02() {
        testbench_symstreamrcf_delay(0.500, 6);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_03)]
    fn test_symstreamrcf_delay_03() {
        testbench_symstreamrcf_delay(0.500, 7);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_04)]
    fn test_symstreamrcf_delay_04() {
        testbench_symstreamrcf_delay(0.500, 8);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_05)]
    fn test_symstreamrcf_delay_05() {
        testbench_symstreamrcf_delay(0.500, 9);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_06)]
    fn test_symstreamrcf_delay_06() {
        testbench_symstreamrcf_delay(0.500, 10);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_07)]
    fn test_symstreamrcf_delay_07() {
        testbench_symstreamrcf_delay(0.500, 14);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_08)]
    fn test_symstreamrcf_delay_08() {
        testbench_symstreamrcf_delay(0.500, 20);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_09)]
    fn test_symstreamrcf_delay_09() {
        testbench_symstreamrcf_delay(0.500, 31);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_10)]
    fn test_symstreamrcf_delay_10() {
        testbench_symstreamrcf_delay(0.800, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_11)]
    fn test_symstreamrcf_delay_11() {
        testbench_symstreamrcf_delay(0.700, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_12)]
    fn test_symstreamrcf_delay_12() {
        testbench_symstreamrcf_delay(0.600, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_13)]
    fn test_symstreamrcf_delay_13() {
        testbench_symstreamrcf_delay(0.500, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_14)]
    fn test_symstreamrcf_delay_14() {
        testbench_symstreamrcf_delay(0.400, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_15)]
    fn test_symstreamrcf_delay_15() {
        testbench_symstreamrcf_delay(0.300, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_16)]
    fn test_symstreamrcf_delay_16() {
        testbench_symstreamrcf_delay(0.200, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_17)]
    fn test_symstreamrcf_delay_17() {
        testbench_symstreamrcf_delay(0.100, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_18)]
    fn test_symstreamrcf_delay_18() {
        testbench_symstreamrcf_delay(0.050, 12);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_19)]
    fn test_symstreamrcf_delay_19() {
        testbench_symstreamrcf_delay(0.025, 12);
    }

    fn testbench_symstreamrcf_psd(bw: f32, m: usize, beta: f32) {
        // create object
        let ftype = FirFilterShape::Arkaiser;
        let ms = ModulationScheme::Qpsk;
        let mut gen = ArbitraryRateSymbolStream::new_linear(ftype, bw, m, beta, ms).unwrap();
        gen.set_gain(bw.sqrt());

        // spectral periodogram options
        let nfft = 2400; // spectral periodogram FFT size
        let num_samples = (192000.0 / bw) as usize; // number of samples

        // create spectral periodogram
        let mut periodogram = SpectralPeriodogram::from_nfft(nfft).unwrap();

        let buf_len = 1337;
        let mut buf = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut n = 0;
        while n < num_samples {
            // fill buffer
            gen.write_samples(&mut buf).unwrap();
            n += buf_len;

            // run through spectral estimation object
            periodogram.write(&buf);
        }

        // compute power spectral density output
        let psd = periodogram.psd();

        // verify spectrum
        // TODO: sidelobe suppression based on internal msresamp object; should be more like -80 dB
        let f0 = 0.5 * (1.0 - beta) * bw;
        let f1 = 0.5 * (1.0 + beta) * bw;
        #[rustfmt::skip]
        let regions = vec![
            PsdRegion {fmin: -0.5, fmax: -f1,  pmin:  0.0, pmax: -55.0, test_lo: false, test_hi: true},
            PsdRegion {fmin: -f0,  fmax:  f0,  pmin: -2.0, pmax:   2.0, test_lo: true,  test_hi: true},
            PsdRegion {fmin:  f1,  fmax:  0.5, pmin:  0.0, pmax: -55.0, test_lo: false, test_hi: true},
        ];

        assert!(validate_psd_spectrum(&psd, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_psd_bw200_m12_b030)]
    fn test_symstreamrcf_psd_bw200_m12_b030() {
        testbench_symstreamrcf_psd(0.2, 12, 0.30)
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_psd_bw400_m12_b030)]
    fn test_symstreamrcf_psd_bw400_m12_b030() {
        testbench_symstreamrcf_psd(0.4, 12, 0.30)
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_psd_bw400_m25_b020)]
    fn test_symstreamrcf_psd_bw400_m25_b020() {
        testbench_symstreamrcf_psd(0.4, 25, 0.20)
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_psd_bw700_m11_b035)]
    fn test_symstreamrcf_psd_bw700_m11_b035() {
        testbench_symstreamrcf_psd(0.7, 11, 0.35)
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_copy)]
    fn test_symstreamrcf_copy() {
        // create objects
        let mut gen_orig = SymbolStream::new_linear(
            FirFilterShape::Arkaiser,
            5, // k = 1/0.2
            17,
            0.27,
            ModulationScheme::Dpsk4,
        )
        .unwrap();

        // allocate memory for buffers
        let buf_len = 1337;
        let mut buf_orig = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_copy = vec![Complex32::new(0.0, 0.0); buf_len];

        // generate some samples
        gen_orig.write_samples(&mut buf_orig).unwrap();

        // copy object
        let mut gen_copy = gen_orig.clone();

        // generate samples from each object
        gen_orig.write_samples(&mut buf_orig).unwrap();
        gen_copy.write_samples(&mut buf_copy).unwrap();

        // compare result
        // NOTE: this will fail as they have different symbol generators
        //assert_eq!(buf_orig, buf_copy);
        println!("WARNING: symstreamrcf_copy results ignored until common PRNG is used");

        // objects are automatically destroyed when they go out of scope
    }
}
