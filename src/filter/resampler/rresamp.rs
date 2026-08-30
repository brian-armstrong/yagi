use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter::{self, FirPfbFilter, FirFilterShape};
use crate::math::gcd;

use num_complex::ComplexFloat;

#[derive(Clone, Debug)]
pub struct Rresamp<T, Coeff = T> {
    p: usize,
    q: usize,
    m: usize,
    block_len: usize,
    pfb: FirPfbFilter<T, Coeff>,
}

impl<T, Coeff> Rresamp<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    pub fn new(interp: usize, decim: usize, m: usize, h: &[Coeff]) -> Result<Self> {
        if interp == 0 {
            return Err(Error::Config("interpolation rate must be greater than zero".into()));
        }
        if decim == 0 {
            return Err(Error::Config("decimation rate must be greater than zero".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter semi-length must be greater than zero".into()));
        }

        let pfb = FirPfbFilter::new(interp, h, 2 * interp * m)?;

        let mut q = Self {
            p: interp,
            q: decim,
            m,
            block_len: 1,
            pfb,
        };

        q.reset();
        Ok(q)
    }

    pub fn new_kaiser(interp: usize, decim: usize, m: usize, bw: f32, as_: f32) -> Result<Self> {
        let gcd = gcd(interp as u32, decim as u32)? as usize;
        let interp = interp / gcd;
        let decim = decim / gcd;

        let bw = if bw < 0.0 {
            if interp > decim { 0.5 } else { 0.5 * interp as f32 / decim as f32 }
        } else if bw > 0.5 {
            return Err(Error::Config(format!("invalid bandwidth ({}), must be less than 0.5", bw)));
        } else {
            bw
        };

        let h_len = 2 * interp * m + 1;
        let hf = filter::fir_design_kaiser(h_len, bw / interp as f32, as_, 0.0)?;

        let h: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();

        let mut q = Self::new(interp, decim, m, &h)?;
        q.set_scale((2.0 * bw * ((q.q as f32) / (q.p as f32)).sqrt()).into());
        q.block_len = gcd;

        Ok(q)
    }

    pub fn new_prototype(ftype: FirFilterShape, interp: usize, decim: usize, m: usize, beta: f32) -> Result<Self> {
        let gcd = gcd(interp as u32, decim as u32)? as usize;
        let interp = interp / gcd;
        let decim = decim / gcd;

        let decim_flag = interp < decim;
        let k = if decim_flag { decim } else { interp };
        let hf = filter::fir_design_prototype(ftype, k, m, beta, 0.0)?;

        let h: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();

        let mut q = Self::new(interp, decim, m, &h)?;
        q.block_len = gcd;

        let rate = q.get_rate();
        q.set_scale((if decim_flag { rate.sqrt() } else { 1.0 / rate.sqrt() }).into());

        Ok(q)
    }

    pub fn new_default(interp: usize, decim: usize) -> Result<Self> {
        let m = 12;
        let bw = 0.5;
        let as_ = 60.0;
        Self::new_kaiser(interp, decim, m, bw, as_)
    }

    pub fn reset(&mut self) -> () {
        self.pfb.reset()
    }

    pub fn set_scale(&mut self, scale: Coeff) -> () {
        self.pfb.set_scale(scale)
    }

    pub fn get_scale(&self) -> Coeff {
        self.pfb.get_scale()
    }

    pub fn get_delay(&self) -> usize {
        self.m
    }

    pub fn get_block_len(&self) -> usize {
        self.block_len
    }

    pub fn get_rate(&self) -> f32 {
        self.p as f32 / self.q as f32
    }

    pub fn get_p(&self) -> usize {
        self.p * self.block_len
    }

    pub fn get_interp(&self) -> usize {
        self.p
    }

    pub fn get_q(&self) -> usize {
        self.q * self.block_len
    }

    pub fn get_decim(&self) -> usize {
        self.q
    }

    pub fn get_num_output(&self, num_input: usize) -> usize {
        let input_block_size = self.q * self.block_len;
        let output_block_size = self.p * self.block_len;
        let num_blocks = num_input / input_block_size;
        num_blocks * output_block_size
    }

    pub fn get_max_input(&self, max_output: usize) -> usize {
        let input_block_size = self.q * self.block_len;
        let output_block_size = self.p * self.block_len;
        let max_blocks = max_output / output_block_size;
        max_blocks * input_block_size
    }

    pub fn write(&mut self, buf: &[T]) -> () {
        self.pfb.write(buf)
    }

    pub fn execute(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        for i in 0..self.block_len {
            let q = self.q;
            let p = self.p;
            self.execute_primitive(&x[i * q..(i + 1) * q], &mut y[i * p..(i + 1) * p])?;
        }
        Ok(())
    }

    pub fn execute_block(&mut self, x: &[T], n: usize, y: &mut [T]) -> Result<()> {
        for i in 0..n {
            let q = self.q;
            let p = self.p;
            self.execute(&x[i * q..(i + 1) * q], &mut y[i * p..(i + 1) * p])?;
        }
        Ok(())
    }

    fn execute_primitive(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        let mut index = 0;
        let mut n = 0;
        for i in 0..self.q {
            self.pfb.push(x[i]);

            while index < self.p {
                y[n] = self.pfb.execute(index)?;
                n += 1;
                index += self.q;
            }

            index -= self.p;
        }

        if index != 0 {
            return Err(Error::Internal(format!("index={} (expected 0)", index)));
        } else if n != self.p {
            return Err(Error::Internal(format!("n={} (expected P={})", n, self.p)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_abs_diff_eq;
    use crate::math::{hamming, WindowType};
    use crate::modem::modem::ModulationScheme;
    use crate::framing::symstreamr::SymStreamR;
    use crate::fft::spgram::Spgram;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_spectrum};
    use num_complex::Complex32;

    fn test_harness_rresamp_crcf_part(p: usize, q: usize, m: usize, n: usize) {
        // semi-fixed options
        let tol = 1e-12f32; // error tolerance (should be basically zero)
        let bw = 0.5f32;    // resampling filter bandwidth
        let as_ = 60.0f32;  // resampling filter stop-band attenuation [dB]

        // create two identical resampler objects
        let mut q0 = Rresamp::<Complex32>::new_kaiser(p, q, m, bw, as_).unwrap();
        let mut q1 = Rresamp::<Complex32>::new_kaiser(p, q, m, bw, as_).unwrap();

        // full input, output buffers
        let mut buf_in = vec![num_complex::Complex32::new(0.0, 0.0); 2 * q * n];
        let mut buf_out_0 = vec![num_complex::Complex32::new(0.0, 0.0); 2 * p * n];
        let mut buf_out_1 = vec![num_complex::Complex32::new(0.0, 0.0); 2 * p * n];

        // generate input signal (pulse, but can really be anything)
        for i in 0..(2 * q * n) {
            buf_in[i] = hamming(i, 2 * q * n).unwrap() * num_complex::Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * 0.037 * i as f32);
        }

        // run resampler normally in one large block (2*q*n inputs, 2*p*n outputs)
        q0.execute_block(&buf_in, 2 * n, &mut buf_out_0).unwrap();

        // reset and run with separate resamplers (e.g. in two threads)
        q0.reset();
        // first block runs as normal
        q0.execute_block(&buf_in, n, &mut buf_out_1[..p * n]).unwrap();
        // initialize second block with q*m samples to account for delay
        for i in 0..m {
            q1.write(&buf_in[q * n - (m - i) * q..q * n - (m - i - 1) * q]);
        }
        // run remainder of second block as normal
        q1.execute_block(&buf_in[q * n..], n, &mut buf_out_1[p * n..]).unwrap();

        // compare output buffers between normal and partitioned operation
        for i in 0..(2 * p * n) {
            assert_abs_diff_eq!(buf_out_0[i].re, buf_out_1[i].re, epsilon = tol);
            assert_abs_diff_eq!(buf_out_0[i].im, buf_out_1[i].im, epsilon = tol);
        }
    }

    // actual tests
    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P1_Q5)]
    fn test_rresamp_crcf_part_p1_q5() { test_harness_rresamp_crcf_part(1, 5, 15, 20); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P2_Q5)]
    fn test_rresamp_crcf_part_p2_q5() { test_harness_rresamp_crcf_part(2, 5, 15, 20); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P3_Q5)]
    fn test_rresamp_crcf_part_p3_q5() { test_harness_rresamp_crcf_part(3, 5, 15, 20); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P6_Q5)]
    fn test_rresamp_crcf_part_p6_q5() { test_harness_rresamp_crcf_part(6, 5, 15, 20); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P8_Q5)]
    fn test_rresamp_crcf_part_p8_q5() { test_harness_rresamp_crcf_part(8, 5, 15, 20); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_part_P9_Q5)]
    fn test_rresamp_crcf_part_p9_q5() { test_harness_rresamp_crcf_part(9, 5, 15, 20); }

    fn test_rresamp_crcf(method: &str,
                         interp: usize,
                         decim: usize,
                         m: usize,
                         bw: f32,
                         as_: f32)
    {
        // options
        let n    = 800000; // number of output samples to analyze
        let nfft = 800;    // number of bins in transform
        let tol  = 0.5f32;   // error tolerance [dB]

        // create resampler with rate interp/decim
        let mut resamp = match method {
            "baseline" => Rresamp::<Complex32, f32>::new_kaiser(interp, decim, m, bw, as_).unwrap(),
            "default" => Rresamp::<Complex32, f32>::new_default(interp, decim).unwrap(),
            _ => {
                let ftype = FirFilterShape::from_str(method).unwrap();
                let beta = bw; // rename to avoid confusion
                Rresamp::<Complex32, f32>::new_prototype(ftype, interp, decim, m, beta).unwrap()
            }
        };

        let r = resamp.get_rate();

        // create and configure objects
        let bw = 0.2f32; // target output bandwidth
        let mut q   = Spgram::<Complex32>::new(nfft, WindowType::Hann, nfft/2, nfft/4).unwrap();
        let mut gen = SymStreamR::new_linear(FirFilterShape::Kaiser, r*bw, 25, 0.2, ModulationScheme::Qpsk).unwrap();
        gen.set_gain((bw*r).sqrt());

        // generate samples and push through spgram object
        let mut buf_0 = vec![num_complex::Complex32::new(0.0, 0.0); decim]; // input buffer
        let mut buf_1 = vec![num_complex::Complex32::new(0.0, 0.0); interp]; // output buffer
        while q.get_num_samples_total() < n {
            // generate block of samples
            gen.write_samples(&mut buf_0).unwrap();

            // resample
            resamp.execute(&buf_0, &mut buf_1).unwrap();

            // run samples through the spgram object
            q.write(&buf_1);
        }

        // verify result
        let psd = q.get_psd();
        let regions = vec![
            PsdRegion { fmin: -0.5,    fmax: -0.6*bw, pmin: 0.0,   pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.4*bw, fmax: 0.4*bw,  pmin: 0.0-tol, pmax: 0.0+tol, test_lo: true,  test_hi: true },
            PsdRegion { fmin: 0.6*bw,  fmax: 0.5,     pmin: 0.0,   pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_spectrum(&psd, nfft, &regions).unwrap());
    }

    // baseline tests using create_kaiser() method
    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P1_Q5)]
    fn test_rresamp_crcf_baseline_p1_q5() { test_rresamp_crcf("baseline", 1, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P2_Q5)]
    fn test_rresamp_crcf_baseline_p2_q5() { test_rresamp_crcf("baseline", 2, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P3_Q5)]
    fn test_rresamp_crcf_baseline_p3_q5() { test_rresamp_crcf("baseline", 3, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P6_Q5)]
    fn test_rresamp_crcf_baseline_p6_q5() { test_rresamp_crcf("baseline", 6, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P8_Q5)]
    fn test_rresamp_crcf_baseline_p8_q5() { test_rresamp_crcf("baseline", 8, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_baseline_P9_Q5)]
    fn test_rresamp_crcf_baseline_p9_q5() { test_rresamp_crcf("baseline", 9, 5, 15, -1.0, 60.0); }

    // tests using create_default() method
    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P1_Q5)]
    fn test_rresamp_crcf_default_p1_q5() { test_rresamp_crcf("default", 1, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P2_Q5)]
    fn test_rresamp_crcf_default_p2_q5() { test_rresamp_crcf("default", 2, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P3_Q5)]
    fn test_rresamp_crcf_default_p3_q5() { test_rresamp_crcf("default", 3, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P6_Q5)]
    fn test_rresamp_crcf_default_p6_q5() { test_rresamp_crcf("default", 6, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P8_Q5)]
    fn test_rresamp_crcf_default_p8_q5() { test_rresamp_crcf("default", 8, 5, 15, -1.0, 60.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_default_P9_Q5)]
    fn test_rresamp_crcf_default_p9_q5() { test_rresamp_crcf("default", 9, 5, 15, -1.0, 60.0); }

    // tests using create_prototype() method
    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_arkaiser_P3_Q5)]
    fn test_rresamp_crcf_arkaiser_p3_q5() { test_rresamp_crcf("arkaiser", 3, 5, 40, 0.2, 50.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_arkaiser_P5_Q3)]
    fn test_rresamp_crcf_arkaiser_p5_q3() { test_rresamp_crcf("arkaiser", 5, 3, 40, 0.2, 50.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_rrcos_P3_Q5)]
    fn test_rresamp_crcf_rrcos_p3_q5() { test_rresamp_crcf("rrcos", 3, 5, 40, 0.2, 50.0); }

    #[test]
    #[autotest_annotate(autotest_rresamp_crcf_rrcos_P5_Q3)]
    fn test_rresamp_crcf_rrcos_p5_q3() { test_rresamp_crcf("rrcos", 5, 3, 40, 0.2, 50.0); }

    fn testbench_rresamp_crcf_num_output(interp: usize, decim: usize) {
        let mut resamp = Rresamp::<Complex32, f32>::new_default(interp, decim).unwrap();
        let q = resamp.get_q();
        let p = resamp.get_p();

        // allocate buffers
        let max_blocks = 20;
        let buf_in = vec![Complex32::new(0.0, 0.0); max_blocks * q];
        let mut buf_out = vec![Complex32::new(0.0, 0.0); max_blocks * p];

        // test various block counts
        for num_blocks in 1..=max_blocks {
            let num_input = num_blocks * q;
            let expected_output = num_blocks * p;

            assert_eq!(resamp.get_num_output(num_input), expected_output);

            // actually run the resampler to verify
            resamp.execute_block(&buf_in[..num_input], num_blocks, &mut buf_out[..expected_output]).unwrap();
        }
    }

    #[test]
    fn test_rresamp_crcf_num_output_0() { testbench_rresamp_crcf_num_output(1, 5); }

    #[test]
    fn test_rresamp_crcf_num_output_1() { testbench_rresamp_crcf_num_output(3, 5); }

    #[test]
    fn test_rresamp_crcf_num_output_2() { testbench_rresamp_crcf_num_output(5, 3); }

    #[test]
    fn test_rresamp_crcf_num_output_3() { testbench_rresamp_crcf_num_output(8, 5); }

    fn testbench_rresamp_crcf_max_input(interp: usize, decim: usize) {
        let resamp = Rresamp::<Complex32, f32>::new_default(interp, decim).unwrap();
        let q = resamp.get_q();

        // test various output limits
        for output_limit in [1, 2, 5, 10, 20, 50, 100] {
            let max_input = resamp.get_max_input(output_limit);
            let num_output = resamp.get_num_output(max_input);

            // get_max_input returns the max inputs that produce at most output_limit outputs
            assert!(
                num_output <= output_limit,
                "interp={}, decim={}, limit={}, max_input={}, num_output={}",
                interp, decim, output_limit, max_input, num_output
            );

            // verify that one more block would exceed the limit
            let next_input = max_input + q;
            let next_output = resamp.get_num_output(next_input);
            assert!(
                next_output > output_limit,
                "interp={}, decim={}, limit={}, next_input={}, next_output={}",
                interp, decim, output_limit, next_input, next_output
            );
        }
    }

    #[test]
    fn test_rresamp_crcf_max_input_0() { testbench_rresamp_crcf_max_input(1, 5); }

    #[test]
    fn test_rresamp_crcf_max_input_1() { testbench_rresamp_crcf_max_input(3, 5); }

    #[test]
    fn test_rresamp_crcf_max_input_2() { testbench_rresamp_crcf_max_input(5, 3); }

    #[test]
    fn test_rresamp_crcf_max_input_3() { testbench_rresamp_crcf_max_input(8, 5); }
}