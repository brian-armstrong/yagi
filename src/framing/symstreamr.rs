use crate::error::{Error, Result};
use crate::filter::msresamp::MsResamp;
use crate::framing::symstream::SymStream;
use crate::filter::FirFilterShape;
use crate::modem::modem::ModulationScheme;
use crate::math::nextpow2;
use num_complex::Complex32;

#[derive(Clone, Debug)]
pub struct SymStreamR {
    symstream: SymStream,
    resamp: MsResamp<Complex32, f32>,
    buf: Vec<Complex32>,
    buf_size: usize,
    buf_index: usize,
}

impl SymStreamR {
    pub fn new() -> Result<Self> {
        Self::new_linear(FirFilterShape::Arkaiser, 0.5, 7, 0.3, ModulationScheme::Qpsk)
    }

    pub fn new_linear(
        ftype: FirFilterShape,
        bw: f32,
        m: usize,
        beta: f32,
        ms: ModulationScheme,
    ) -> Result<Self> {
        const BW_MIN: f32 = 0.001;
        const BW_MAX: f32 = 1.000;
        if !(BW_MIN..=BW_MAX).contains(&bw) {
            return Err(Error::Config(format!("symbol bandwidth ({}) must be in [{},{}]", bw, BW_MIN, BW_MAX)));
        }

        let symstream = SymStream::new_linear(ftype, 2, m, beta, ms)?;
        let rate = 0.5 / bw;
        let resamp = MsResamp::new(rate, 60.0)?;

        let buf_len = 1 << nextpow2((0.5 / bw).ceil() as u32)?;
        let buf = vec![Complex32::new(0.0, 0.0); buf_len];

        let mut q = Self {
            symstream,
            resamp,
            buf,
            buf_size: 0,
            buf_index: 0,
        };

        q.reset();
        Ok(q)
    }

    pub fn reset(&mut self) {
        self.symstream.reset();
        self.resamp.reset();
        self.buf_size = 0;
        self.buf_index = 0;
    }

    pub fn get_ftype(&self) -> FirFilterShape {
        self.symstream.get_ftype()
    }

    pub fn get_bw(&self) -> f32 {
        1.0 / (self.resamp.get_rate() * self.symstream.get_k() as f32)
    }

    pub fn get_m(&self) -> usize {
        self.symstream.get_m()
    }

    pub fn get_beta(&self) -> f32 {
        self.symstream.get_beta()
    }

    pub fn set_scheme(&mut self, ms: ModulationScheme) -> Result<()> {
        self.symstream.set_scheme(ms)
    }

    pub fn get_scheme(&self) -> ModulationScheme {
        self.symstream.get_scheme()
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.symstream.set_gain(gain);
    }

    pub fn get_gain(&self) -> f32 {
        self.symstream.get_gain()
    }

    pub fn get_delay(&self) -> f32 {
        let p = self.symstream.get_delay() as f32;
        let d = self.resamp.get_delay();
        let r = self.resamp.get_rate();
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

            self.buf_size = self.resamp.execute(&sample, &mut self.buf)?;
        }
        Ok(())
    }

    pub fn write_samples(&mut self, buf: &mut [Complex32]) -> Result<()> {
        for sample in buf.iter_mut() {
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
    use test_macro::autotest_annotate;
    use crate::fft::{fft_run, Direction};
    use crate::fft::spgram::Spgram;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_spectrum};
    use approx::assert_abs_diff_eq;

    fn testbench_symstreamrcf_delay(bw: f32, m: usize) {
        // create object and get expected delay
        let ftype = FirFilterShape::Arkaiser;
        let beta = 0.30;
        let ms = ModulationScheme::Qpsk;
        let mut gen = SymStreamR::new_linear(ftype, bw, m, beta, ms).unwrap();
        let delay = gen.get_delay();
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
            println!("expected delay: {:.6}, measured: {:.6}, err: {:.6} (tol= {:.3})",
                     delay, delay_meas, delay - delay_meas, tol);
        }

        // verify delay is relatively close to expected
        assert_abs_diff_eq!(delay, delay_meas, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_00)]
    fn test_symstreamrcf_delay_00() { testbench_symstreamrcf_delay(0.500, 4); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_01)]
    fn test_symstreamrcf_delay_01() { testbench_symstreamrcf_delay(0.500, 5); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_02)]
    fn test_symstreamrcf_delay_02() { testbench_symstreamrcf_delay(0.500, 6); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_03)]
    fn test_symstreamrcf_delay_03() { testbench_symstreamrcf_delay(0.500, 7); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_04)]
    fn test_symstreamrcf_delay_04() { testbench_symstreamrcf_delay(0.500, 8); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_05)]
    fn test_symstreamrcf_delay_05() { testbench_symstreamrcf_delay(0.500, 9); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_06)]
    fn test_symstreamrcf_delay_06() { testbench_symstreamrcf_delay(0.500, 10); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_07)]
    fn test_symstreamrcf_delay_07() { testbench_symstreamrcf_delay(0.500, 14); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_08)]
    fn test_symstreamrcf_delay_08() { testbench_symstreamrcf_delay(0.500, 20); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_09)]
    fn test_symstreamrcf_delay_09() { testbench_symstreamrcf_delay(0.500, 31); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_10)]
    fn test_symstreamrcf_delay_10() { testbench_symstreamrcf_delay(0.800, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_11)]
    fn test_symstreamrcf_delay_11() { testbench_symstreamrcf_delay(0.700, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_12)]
    fn test_symstreamrcf_delay_12() { testbench_symstreamrcf_delay(0.600, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_13)]
    fn test_symstreamrcf_delay_13() { testbench_symstreamrcf_delay(0.500, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_14)]
    fn test_symstreamrcf_delay_14() { testbench_symstreamrcf_delay(0.400, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_15)]
    fn test_symstreamrcf_delay_15() { testbench_symstreamrcf_delay(0.300, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_16)]
    fn test_symstreamrcf_delay_16() { testbench_symstreamrcf_delay(0.200, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_17)]
    fn test_symstreamrcf_delay_17() { testbench_symstreamrcf_delay(0.100, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_18)]
    fn test_symstreamrcf_delay_18() { testbench_symstreamrcf_delay(0.050, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamrcf_delay_19)]
    fn test_symstreamrcf_delay_19() { testbench_symstreamrcf_delay(0.025, 12); }

    fn testbench_symstreamrcf_psd(bw: f32, m: usize, beta: f32) {
        // create object
        let ftype = FirFilterShape::Arkaiser;
        let ms = ModulationScheme::Qpsk;
        let mut gen = SymStreamR::new_linear(ftype, bw, m, beta, ms).unwrap();
        gen.set_gain(bw.sqrt());

        // spectral periodogram options
        let nfft = 2400; // spectral periodogram FFT size
        let num_samples = (192000.0 / bw) as usize; // number of samples

        // create spectral periodogram
        let mut periodogram = Spgram::default(nfft).unwrap();

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
        let psd = periodogram.get_psd();

        // verify spectrum
        // TODO: sidelobe suppression based on internal msresamp object; should be more like -80 dB
        let f0 = 0.5 * (1.0 - beta) * bw;
        let f1 = 0.5 * (1.0 + beta) * bw;
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
        let mut gen_orig = SymStream::new_linear(
            FirFilterShape::Arkaiser,
            5, // k = 1/0.2
            17,
            0.27,
            ModulationScheme::Dpsk4
        ).unwrap();

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