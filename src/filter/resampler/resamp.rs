use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter::{self, FirPfbFilter};
use crate::math::nextpow2;

use num_complex::ComplexFloat;

#[derive(Clone, Debug)]
pub struct Resamp<T, Coeff = T> {
    m: usize,
    r: f32,
    step: u32,
    phase: u32,
    bits_index: usize,
    pfb: FirPfbFilter<T, Coeff>,
}

impl<T, Coeff> Resamp<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + std::ops::Mul<Coeff, Output = T>,
    [Coeff]: DotProd<T, Output = T>,
{
    pub fn new(rate: f32, m: usize, fc: f32, as_: f32, npfb: usize) -> Result<Self> {
        if rate <= 0.0 {
            return Err(Error::Config("resampling rate must be greater than zero".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter semi-length must be greater than zero".into()));
        }
        if fc <= 0.0 || fc >= 0.5 {
            return Err(Error::Config("filter cutoff must be in (0,0.5)".into()));
        }
        if as_ <= 0.0 {
            return Err(Error::Config("filter stop-band suppression must be greater than zero".into()));
        }

        let bits = nextpow2(npfb as u32)? as usize;
        if bits < 1 || bits > 16 {
            return Err(Error::Config("number of filter banks must be in (2^0,2^16)".into()));
        }

        let npfb = 1 << bits;

        // design filter
        let n = 2 * m * npfb + 1;
        let hf = filter::fir_design_kaiser(n, fc / (npfb as f32), as_, 0.0)?;

        // normalize filter coefficients by DC gain
        let gain = hf.iter().sum::<f32>();
        let gain = (npfb as f32) / gain;

        // copy to type-specific array, applying gain
        let h: Vec<Coeff> = hf.iter().map(|&x| (x * gain).into()).collect();

        let pfb = FirPfbFilter::new(npfb, &h, n - 1)?;

        let mut q = Self {
            m,
            r: rate,
            step: 0,
            phase: 0,
            bits_index: bits,
            pfb,
        };

        q.set_rate(rate)?;

        q.reset();
        Ok(q)
    }

    pub fn new_default(rate: f32) -> Result<Self> {
        if rate <= 0.0 {
            return Err(Error::Config("resampling rate must be greater than zero".into()));
        }

        let m = 7;
        let fc = 0.25;
        let as_ = 60.0;
        let npfb = 256;

        Self::new(rate, m, fc, as_, npfb)
    }

    pub fn reset(&mut self) -> () {
        self.phase = 0;
        self.pfb.reset();
    }

    pub fn get_delay(&self) -> usize {
        self.m
    }

    pub fn set_rate(&mut self, rate: f32) -> Result<()> {
        if rate <= 0.0 {
            return Err(Error::Config("resampling rate must be greater than zero".into()));
        } else if rate < 0.004 || rate > 250.0 {
            return Err(Error::Config("resampling rate must be in [0.004,250]".into()));
        }

        self.r = rate;
        self.step = ((1 << 24) as f32 / self.r).round() as u32;

        Ok(())
    }

    pub fn get_rate(&self) -> f32 {
        self.r
    }

    pub fn adjust_rate(&mut self, gamma: f32) -> Result<()> {
        if gamma <= 0.0 {
            return Err(Error::Config(format!("resampling adjustment ({}) must be greater than zero", gamma)));
        }

        self.set_rate(self.r * gamma)
    }

    // pub fn set_timing_phase(&mut self, _tau: f32) -> Result<()> {
    //     unimplemented!()
    // }

    // pub fn adjust_timing_phase(&mut self, _delta: f32) -> Result<()> {
    //     unimplemented!()
    // }

    pub fn get_num_output(&self, num_input: usize) -> usize {
        let mut phase = self.phase;
        let mut num_output = 0;
        for _ in 0..num_input {
            while phase <= 0x00ffffff {
                num_output += 1;
                phase += self.step;
            }
            phase -= 1 << 24;
        }
        num_output
    }

    pub fn get_max_input(&self, max_output: usize) -> usize {
        let mut phase = self.phase;
        let mut output_count = 0;
        let mut num_input = 0;
        loop {
            while phase <= 0x00ffffff {
                output_count += 1;
                if output_count > max_output {
                    return num_input;
                }
                phase += self.step;
            }
            phase -= 1 << 24;
            num_input += 1;
        }
    }

    pub fn execute(&mut self, x: T, y: &mut [T]) -> Result<usize> {
        self.pfb.push(x);

        let mut n = 0;
        while self.phase <= 0x00ffffff {
            let index = self.phase >> (24 - self.bits_index);
            y[n] = self.pfb.execute(index as usize)?;
            n += 1;
            self.phase += self.step;
        }
        self.phase -= 1 << 24;

        Ok(n)
    }

    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<usize> {
        let mut ny = 0;

        for &xi in x {
            let num_written = self.execute(xi, &mut y[ny..])?;
            ny += num_written;
        }

        Ok(ny)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use num_complex::Complex32;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_signal};
    use crate::random::randnf;

    fn testbench_resamp_crcf(r: f32, as_db: f32, _id: usize) {
        // options
        let bw: f32 = 0.25;  // target output signal bandwidth
        let tol: f32 = 0.5;  // output PSD error tolerance [dB]
        let m: usize = 20;   // resampler semi-length
        let npfb: usize = 2048; // number of filters in bank
        let fc: f32 = 0.45;  // resampler cut-off frequency

        // create resampler
        let mut resamp = Resamp::<Complex32>::new(r, m, fc, as_db, npfb).unwrap();

        // generate pulse with sharp transition and very narrow side-lobes
        let p = (40.0 / r) as usize;
        let pulse_len = 4 * p + 1;
        let pulse = filter::fir_design_kaiser(pulse_len, 0.5 * r * bw, 120.0, 0.0).unwrap();

        // allocate buffers and copy input
        let num_input = pulse_len + 2 * m + 1;
        let num_output = resamp.get_num_output(num_input);
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); num_input];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); num_output];

        for i in 0..num_input {
            buf_0[i] = if i < pulse_len {
                Complex32::new(pulse[i] * bw, 0.0)
            } else {
                Complex32::new(0.0, 0.0)
            };
        }

        // resample
        let nw = resamp.execute_block(&buf_0, &mut buf_1).unwrap();

        // verify result
        let regions = vec![
            PsdRegion { fmin: -0.5, fmax: -0.6 * bw, pmin: 0.0, pmax: -as_db + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.4 * bw, fmax: 0.4 * bw, pmin: -tol, pmax: tol, test_lo: true, test_hi: true },
            PsdRegion { fmin: 0.6 * bw, fmax: 0.5, pmin: 0.0, pmax: -as_db + tol, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_signal(&buf_1[..nw], &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_00)]
    fn test_resamp_crcf_00() {
        testbench_resamp_crcf(0.127115323, 60.0, 0);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_01)]
    fn test_resamp_crcf_01() {
        testbench_resamp_crcf(0.373737373, 60.0, 1);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_02)]
    fn test_resamp_crcf_02() {
        testbench_resamp_crcf(0.676543210, 60.0, 2);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_03)]
    fn test_resamp_crcf_03() {
        testbench_resamp_crcf(0.973621947, 60.0, 3);
    }

    // Commented out tests
    // #[test]
    // fn test_resamp_crcf_04() {
    //     testbench_resamp_crcf(1.023832447, 60.0, 4);
    // }
    // #[test]
    // fn test_resamp_crcf_05() {
    //     testbench_resamp_crcf(2.182634827, 60.0, 5);
    // }
    // #[test]
    // fn test_resamp_crcf_06() {
    //     testbench_resamp_crcf(8.123980823, 60.0, 6);
    // }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_10)]
    fn test_resamp_crcf_10() {
        testbench_resamp_crcf(0.127115323, 80.0, 10);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_11)]
    fn test_resamp_crcf_11() {
        testbench_resamp_crcf(0.373737373, 80.0, 11);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_12)]
    fn test_resamp_crcf_12() {
        testbench_resamp_crcf(0.676543210, 80.0, 12);
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_13)]
    fn test_resamp_crcf_13() {
        testbench_resamp_crcf(0.973621947, 80.0, 13);
    }

    fn testbench_resamp_crcf_num_output(rate: f32, npfb: usize) {
        if cfg!(test) {
            println!("testing resamp_crcf_get_num_output() with r={}, npfb={}", rate, npfb);
        }

        // create object
        let fc = 0.4f32;
        let as_db = 60.0f32;
        let m = 20;
        let mut resamp = Resamp::<Complex32>::new(rate, m, fc, as_db, npfb).unwrap();

        // sizes to test in sequence
        let sizes = [1, 2, 3, 20, 7, 64, 4, 4, 4, 27];

        // allocate buffers (over-provision output to help avoid segmentation faults on error)
        let max_input = 64;
        let max_output = 16 + (4.0 * max_input as f32 * rate) as usize;
        let buf_0 = vec![Complex32::new(0.0, 0.0); max_input];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); max_output];

        // run numerous blocks
        for _b in 0..8 {
            for (_i, &size) in sizes.iter().enumerate() {
                let num_input = size;
                let num_output = resamp.get_num_output(num_input);
                let num_written = resamp.execute_block(&buf_0[..num_input], &mut buf_1).unwrap();
                assert_eq!(num_output, num_written);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_0)]
    fn test_resamp_crcf_num_output_0() { testbench_resamp_crcf_num_output(1.00, 64); }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_1)]
    fn test_resamp_crcf_num_output_1() { testbench_resamp_crcf_num_output(1.00, 256); }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_2)]
    fn test_resamp_crcf_num_output_2() { testbench_resamp_crcf_num_output(0.50, 256); }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_3)]
    fn test_resamp_crcf_num_output_3() { testbench_resamp_crcf_num_output(2f32.sqrt(), 256); }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_4)]
    fn test_resamp_crcf_num_output_4() { testbench_resamp_crcf_num_output(17f32.sqrt(), 16); }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_5)]
    fn test_resamp_crcf_num_output_5() { testbench_resamp_crcf_num_output(1.0 / std::f32::consts::PI, 64); }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_6)]
    fn test_resamp_crcf_num_output_6() { testbench_resamp_crcf_num_output(5.0f32.exp(), 64); }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_num_output_7)]
    fn test_resamp_crcf_num_output_7() { testbench_resamp_crcf_num_output((-5.0f32).exp(), 64); }

    #[test]
    #[autotest_annotate(autotest_resamp_crcf_copy)]
    fn test_resamp_crcf_copy() {
        // create object with irregular parameters
        let rate = 0.71239213987520f32;
        let mut q0 = Resamp::<Complex32>::new(rate, 17, 0.37, 60.0, 64).unwrap();

        // run samples through filter
        let num_samples = 80;
        let mut y0 = vec![Complex32::new(0.0, 0.0); 1];
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let _nw0 = q0.execute(v, &mut y0).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters and check equality
        let mut y1 = vec![Complex32::new(0.0, 0.0); 1];
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let nw0 = q0.execute(v, &mut y0).unwrap();
            let nw1 = q1.execute(v, &mut y1).unwrap();

            // check that either 0 or 1 samples were written
            assert!(nw0 < 2);
            assert!(nw1 < 2);

            // check that the same number of samples were written
            assert_eq!(nw0, nw1);

            // check output sample values
            if nw0 == 1 && nw1 == 1 {
                assert_eq!(y0, y1);
            }
        }
    }

    fn testbench_resamp_crcf_max_input(rate: f32, npfb: usize) {
        // create object
        let fc = 0.4f32;
        let as_db = 60.0f32;
        let m = 20;
        let mut resamp = Resamp::<Complex32>::new(rate, m, fc, as_db, npfb).unwrap();

        // allocate buffers
        let max_input = 1024;
        let max_output = 16 + (4.0 * max_input as f32 * rate.max(1.0)) as usize;
        let buf_0 = vec![Complex32::new(0.0, 0.0); max_input];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); max_output];

        // run blocks and verify get_max_input is consistent with get_num_output
        for output_limit in [1, 2, 3, 5, 10, 20, 50, 100] {
            let num_input = resamp.get_max_input(output_limit);
            let num_output = resamp.get_num_output(num_input);

            // get_max_input returns the max inputs that produce at most output_limit outputs
            assert!(
                num_output <= output_limit,
                "rate={}, limit={}, num_input={}, num_output={}",
                rate, output_limit, num_input, num_output
            );

            // verify that one more input would exceed the limit
            let num_output_more = resamp.get_num_output(num_input + 1);
            assert!(
                num_output_more > output_limit,
                "rate={}, limit={}, num_input+1={}, num_output_more={}",
                rate, output_limit, num_input + 1, num_output_more
            );

            // actually run the resampler to verify
            let num_written = resamp.execute_block(&buf_0[..num_input], &mut buf_1).unwrap();
            assert_eq!(num_output, num_written);
        }
    }

    #[test]
    fn test_resamp_crcf_max_input_0() { testbench_resamp_crcf_max_input(1.00, 64); }

    #[test]
    fn test_resamp_crcf_max_input_1() { testbench_resamp_crcf_max_input(0.50, 256); }

    #[test]
    fn test_resamp_crcf_max_input_2() { testbench_resamp_crcf_max_input(2.0, 256); }

    #[test]
    fn test_resamp_crcf_max_input_3() { testbench_resamp_crcf_max_input(0.127115323, 64); }

    #[test]
    fn test_resamp_crcf_max_input_4() { testbench_resamp_crcf_max_input(std::f32::consts::PI, 64); }

    #[test]
    fn test_resamp_crcf_max_input_5() { testbench_resamp_crcf_max_input(1.0 / std::f32::consts::PI, 64); }
}