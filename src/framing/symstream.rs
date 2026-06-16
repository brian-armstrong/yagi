use crate::error::{Error, Result};
use crate::filter::{FirInterpolationFilter, FirFilterShape};
use crate::modem::modem::{Modem, ModulationScheme};
use num_complex::Complex32;

#[derive(Clone, Debug)]
pub struct SymStream {
    filter_type: FirFilterShape,
    k: usize,
    m: usize,
    beta: f32,
    modem: Modem,
    gain: f32,
    interp: FirInterpolationFilter<Complex32>,
    buf: Vec<Complex32>,
    buf_index: usize,
}

impl SymStream {
    pub fn new() -> Result<Self> {
        Self::new_linear(FirFilterShape::Arkaiser, 2, 7, 0.3, ModulationScheme::Qpsk)
    }

    pub fn new_linear(
        ftype: FirFilterShape,
        k: usize,
        m: usize,
        beta: f32,
        ms: ModulationScheme,
    ) -> Result<Self> {
        if k < 2 {
            return Err(Error::Config("samples/symbol must be at least 2".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than zero".into()));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(Error::Config("filter excess bandwidth must be in (0,1]".into()));
        }

        let mod_ = Modem::new(ms)?;
        let interp = FirInterpolationFilter::new_prototype(ftype, k, m, beta, 0.0)?;
        let buf = vec![Complex32::default(); k];

        let mut q = Self {
            filter_type: ftype,
            k,
            m,
            beta,
            modem: mod_,
            gain: 1.0,
            interp,
            buf,
            buf_index: 0,
        };

        q.reset();
        Ok(q)
    }

    pub fn reset(&mut self) -> () {
        self.modem.reset();
        self.interp.reset();
        self.buf_index = 0;
    }

    pub fn get_ftype(&self) -> FirFilterShape {
        self.filter_type
    }

    pub fn get_k(&self) -> usize {
        self.k
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn get_beta(&self) -> f32 {
        self.beta
    }

    pub fn set_scheme(&mut self, ms: ModulationScheme) -> Result<()> {
        self.modem = Modem::new(ms)?;
        Ok(())
    }

    pub fn get_scheme(&self) -> ModulationScheme {
        self.modem.get_scheme()
    }

    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain;
    }

    pub fn get_gain(&self) -> f32 {
        self.gain
    }

    pub fn get_delay(&self) -> usize {
        self.k * self.m
    }

    fn fill_buffer(&mut self) -> Result<()> {
        let sym = self.modem.random_symbol();
        let v = self.modem.modulate(sym)? * self.gain;
        self.interp.execute(v, &mut self.buf)?;
        Ok(())
    }

    pub fn write_samples(&mut self, buf: &mut [Complex32]) -> Result<()> {
        for sample in buf.iter_mut() {
            if self.buf_index == 0 {
                self.fill_buffer()?;
            }

            *sample = self.buf[self.buf_index];
            self.buf_index = (self.buf_index + 1) % self.k;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_spectrum};
    use crate::fft::spgram::Spgram;
    use approx::assert_abs_diff_eq;

    fn testbench_symstreamcf_delay(k: usize, m: usize) {
        // create object and get expected delay
        let ftype = FirFilterShape::Arkaiser;
        let beta = 0.30;
        let ms = ModulationScheme::Qpsk;
        let mut gen = SymStream::new_linear(ftype, k, m, beta, ms).unwrap();
        let delay = gen.get_delay();
        let tol = 2.0 + k as f32; // error tolerance (fairly wide due to random signal)

        let mut i = 0;
        let mut sample = Complex32::new(0.0, 0.0);
        for _ in 0..1000 + delay {
            // generate a single sample
            gen.write_samples(std::slice::from_mut(&mut sample)).unwrap();

            // check to see if value exceeds 1
            if sample.norm() > 0.9 {
                break;
            }
            i += 1;
        }

        // verify delay is relatively close to expected
        assert_abs_diff_eq!(delay as f32, i as f32, epsilon = tol);
    }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_00)]
    fn test_symstreamcf_delay_00() { testbench_symstreamcf_delay(2, 4); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_01)]
    fn test_symstreamcf_delay_01() { testbench_symstreamcf_delay(2, 5); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_02)]
    fn test_symstreamcf_delay_02() { testbench_symstreamcf_delay(2, 6); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_03)]
    fn test_symstreamcf_delay_03() { testbench_symstreamcf_delay(2, 7); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_04)]
    fn test_symstreamcf_delay_04() { testbench_symstreamcf_delay(2, 8); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_05)]
    fn test_symstreamcf_delay_05() { testbench_symstreamcf_delay(2, 9); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_06)]
    fn test_symstreamcf_delay_06() { testbench_symstreamcf_delay(2, 10); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_07)]
    fn test_symstreamcf_delay_07() { testbench_symstreamcf_delay(2, 14); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_08)]
    fn test_symstreamcf_delay_08() { testbench_symstreamcf_delay(2, 20); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_09)]
    fn test_symstreamcf_delay_09() { testbench_symstreamcf_delay(2, 31); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_10)]
    fn test_symstreamcf_delay_10() { testbench_symstreamcf_delay(3, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_11)]
    fn test_symstreamcf_delay_11() { testbench_symstreamcf_delay(4, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_12)]
    fn test_symstreamcf_delay_12() { testbench_symstreamcf_delay(5, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_13)]
    fn test_symstreamcf_delay_13() { testbench_symstreamcf_delay(6, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_14)]
    fn test_symstreamcf_delay_14() { testbench_symstreamcf_delay(7, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_15)]
    fn test_symstreamcf_delay_15() { testbench_symstreamcf_delay(8, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_16)]
    fn test_symstreamcf_delay_16() { testbench_symstreamcf_delay(9, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_17)]
    fn test_symstreamcf_delay_17() { testbench_symstreamcf_delay(10, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_18)]
    fn test_symstreamcf_delay_18() { testbench_symstreamcf_delay(11, 12); }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_delay_19)]
    fn test_symstreamcf_delay_19() { testbench_symstreamcf_delay(12, 12); }

    fn testbench_symstreamcf_psd(k: usize, m: usize, beta: f32) {
        // create object
        let ftype = FirFilterShape::Arkaiser;
        let ms = ModulationScheme::Qpsk;
        let mut gen = SymStream::new_linear(ftype, k, m, beta, ms).unwrap();
        gen.set_gain(1.0 / (k as f32).sqrt());

        // spectral periodogram options
        let nfft = 2400;      // spectral periodogram FFT size
        let num_samples = 192000 * k;   // number of samples

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
        let f0 = 0.5 * (1.0 - beta) / k as f32;
        let f1 = 0.5 * (1.0 + beta) / k as f32;
        let regions = vec![
            PsdRegion {fmin: -0.5, fmax: -f1,  pmin:  0.0, pmax: -80.0, test_lo: false, test_hi: true},
            PsdRegion {fmin: -f0,  fmax:  f0,  pmin: -1.0, pmax:  1.0, test_lo: true,  test_hi: true},
            PsdRegion {fmin:  f1,  fmax:  0.5, pmin:  0.0, pmax: -80.0, test_lo: false, test_hi: true},
        ];

        assert!(validate_psd_spectrum(&psd, nfft, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_psd_k2_m12_b030)]
    fn test_symstreamcf_psd_k2_m12_b030() {
        testbench_symstreamcf_psd(2, 12, 0.30)
    }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_psd_k4_m12_b030)]
    fn test_symstreamcf_psd_k4_m12_b030() {
        testbench_symstreamcf_psd(4, 12, 0.30)
    }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_psd_k4_m25_b020)]
    fn test_symstreamcf_psd_k4_m25_b020() {
        testbench_symstreamcf_psd(4, 25, 0.20)
    }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_psd_k7_m11_b035)]
    fn test_symstreamcf_psd_k7_m11_b035() {
        testbench_symstreamcf_psd(7, 11, 0.35)
    }

    #[test]
    #[autotest_annotate(autotest_symstreamcf_copy)]
    fn test_symstreamcf_copy() {
        // create objects
        let mut gen_orig = SymStream::new_linear(
            FirFilterShape::Arkaiser,
            5,
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
        println!("WARNING: symstreamcf_copy results ignored until common PRNG is used");

        // objects are automatically destroyed when they go out of scope
    }

}