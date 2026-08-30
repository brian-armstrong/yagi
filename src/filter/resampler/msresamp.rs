use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter::resampler::resamp::Resamp;
use crate::filter::resampler::resamp2::Resamp2Coeff;
use crate::filter::resampler::msresamp2::{MsResamp2, ResampType};
use std::f32;

use num_complex::ComplexFloat;

#[derive(Clone, Debug)]
pub struct MsResamp<T, Coeff = T> {
    rate: f32,
    type_: ResampType,
    rate_arbitrary: f32,
    num_halfband_stages: usize,
    buffer: Vec<T>,
    buffer_index: usize,
    halfband_resamp: MsResamp2<T, Coeff>,
    arbitrary_resamp: Resamp<T, Coeff>,
}

impl<T, Coeff> MsResamp<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32> + Resamp2Coeff,
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + From<f32> + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    pub fn new(rate: f32, as_: f32) -> Result<Self> {
        if rate <= 0.0 {
            return Err(Error::Config("resampling rate must be greater than zero".into()));
        }

        let type_ = if rate > 1.0 { ResampType::Interp } else { ResampType::Decim };

        let mut rate_arbitrary = rate;
        let mut num_halfband_stages = 0;

        match type_ {
            ResampType::Interp => {
                while rate_arbitrary > 2.0 {
                    num_halfband_stages += 1;
                    rate_arbitrary *= 0.5;
                }
            }
            ResampType::Decim => {
                while rate_arbitrary < 0.5 {
                    num_halfband_stages += 1;
                    rate_arbitrary *= 2.0;
                }

            }
        }

        let buffer_len = 4 + (1 << num_halfband_stages);
        let buffer = vec![T::default(); buffer_len];

        // TODO: Compute appropriate cut-off frequency
        let halfband_resamp = MsResamp2::new(type_, num_halfband_stages, 0.4, 0.0, as_)?;

        // TODO: Compute appropriate parameters
        let arbitrary_resamp = Resamp::new(
            rate_arbitrary,
            7,
            f32::min(0.515 * rate_arbitrary, 0.49),
            as_,
            256,
        )?;

        Ok(Self {
            rate,
            type_,
            rate_arbitrary,
            num_halfband_stages,
            buffer,
            buffer_index: 0,
            halfband_resamp,
            arbitrary_resamp,
        })
    }

    pub fn reset(&mut self) {
        self.halfband_resamp.reset();
        self.arbitrary_resamp.reset();
        self.buffer_index = 0;
    }

    pub fn get_delay(&self) -> f32 {
        let delay_halfband = self.halfband_resamp.get_delay();
        let delay_arbitrary = self.arbitrary_resamp.get_delay() as f32;

        // compute delay based on interpolation or decimation type
        if self.num_halfband_stages == 0 {
            // no half-band stages; just arbitrary resampler delay
            delay_arbitrary
        } else if self.type_ == ResampType::Interp {
            // interpolation
            delay_halfband / self.rate_arbitrary + delay_arbitrary
        } else {
            // decimation
            let m = 1 << self.num_halfband_stages;
            delay_halfband + m as f32 * delay_arbitrary
        }
    }

    pub fn get_rate(&self) -> f32 {
        self.rate
    }

    pub fn get_num_output(&self, num_input: usize) -> usize {
        match self.type_ {
            ResampType::Interp => {
                let n = self.arbitrary_resamp.get_num_output(num_input);
                n * (1 << self.num_halfband_stages)
            }
            ResampType::Decim => {
                let n = (self.buffer_index + num_input) >> self.num_halfband_stages;
                self.arbitrary_resamp.get_num_output(n)
            }
        }
    }

    pub fn get_max_input(&self, max_output: usize) -> usize {
        match self.type_ {
            ResampType::Interp => {
                let halfband_factor = 1 << self.num_halfband_stages;
                let max_arbitrary_outputs = max_output / halfband_factor;
                self.arbitrary_resamp.get_max_input(max_arbitrary_outputs)
            }
            ResampType::Decim => {
                let halfband_factor = 1 << self.num_halfband_stages;
                let max_halfband_outputs = self.arbitrary_resamp.get_max_input(max_output);

                if max_halfband_outputs == 0 {
                    halfband_factor - 1 - self.buffer_index
                } else {
                    max_halfband_outputs * halfband_factor - self.buffer_index + (halfband_factor - 1)
                }
            }
        }
    }

    pub fn execute(&mut self, x: &[T], y: &mut [T]) -> Result<usize> {
        match self.type_ {
            ResampType::Interp => self.interp_execute(x, y),
            ResampType::Decim => self.decim_execute(x, y),
        }
    }

    fn interp_execute(&mut self, x: &[T], y: &mut [T]) -> Result<usize> {
        let mut ny = 0;

        for &xi in x {
            let nw = self.arbitrary_resamp.execute(xi, &mut self.buffer[..])?;

            for k in 0..nw {
                self.halfband_resamp.execute(&[self.buffer[k]], &mut y[ny..])?;
                ny += 1 << self.num_halfband_stages;
            }
        }

        Ok(ny)
    }

    fn decim_execute(&mut self, x: &[T], y: &mut [T]) -> Result<usize> {
        let mut ny = 0;
        let m = 1 << self.num_halfband_stages;

        for &xi in x {
            self.buffer[self.buffer_index] = xi;
            self.buffer_index += 1;

            if self.buffer_index == m {
                let mut halfband_output = T::default();
                self.halfband_resamp.execute(&self.buffer[..m], std::slice::from_mut(&mut halfband_output))?;

                let nw = self.arbitrary_resamp.execute(halfband_output, &mut y[ny..])?;
                ny += nw;

                self.buffer_index = 0;
            }
        }

        Ok(ny)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use num_complex::Complex32;
    use crate::random::randnf;
    use crate::fft::spgram::Spgram;
    use crate::framing::symstreamr::SymStreamR;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_spectrum};
    use crate::modem::modem::ModulationScheme;
    use crate::math::WindowType;
    use crate::filter::FirFilterShape;

    fn testbench_msresamp_crcf(r: f32, as_: f32) {
        // options
        let n = 800000;      // number of output samples to analyze
        let bw = 0.2f32; // target output bandwidth
        let nfft = 800;
        let tol = 0.5f32;

        // create and configure objects
        let mut q = Spgram::<Complex32>::new(nfft, WindowType::Hann, nfft/2, nfft/4).unwrap();
        let mut gen = SymStreamR::new_linear(FirFilterShape::Kaiser, r*bw, 25, 0.2, ModulationScheme::Qpsk).unwrap();
        gen.set_gain((bw as f32).sqrt());
        let mut resamp = MsResamp::<Complex32, f32>::new(r, as_).unwrap();

        // generate samples and push through spgram object
        let buf_len = 256;
        let mut buf_0 = vec![Complex32::default(); buf_len]; // input buffer
        let mut buf_1 = vec![Complex32::default(); buf_len]; // output buffer
        while q.get_num_samples_total() < n {
            // generate block of samples
            gen.write_samples(&mut buf_0).unwrap();

            // resample
            let nw = resamp.execute(&buf_0, &mut buf_1).unwrap();

            // run samples through the spgram object
            q.write(&buf_1[..nw]);
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

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_01)]
    fn test_msresamp_crcf_01() { testbench_msresamp_crcf(0.127115323, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_02)]
    fn test_msresamp_crcf_02() { testbench_msresamp_crcf(0.373737373, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_03)]
    fn test_msresamp_crcf_03() { testbench_msresamp_crcf(0.676543210, 60.0); }

    // #[test]
    // #[autotest_annotate(xautotest_msresamp_crcf_04)]
    // fn test_msresamp_crcf_04() { testbench_msresamp_crcf(0.127115323, 80.0); }

    fn testbench_msresamp_crcf_num_output(rate: f32) {
        // create object
        let as_ = 60.0f32;
        let mut q = MsResamp::<Complex32, f32>::new(rate, as_).unwrap();

        // sizes to test in sequence
        let s = if rate < 0.1 { 131 } else { 1 }; // scale: increase for large decimation rates
        let sizes = [1*s, 2*s, 3*s, 20*s, 7*s, 64*s, 4*s, 4*s, 4*s, 27*s];

        // allocate buffers (over-provision output to help avoid segmentation faults on error)
        let max_input = 64 * s;
        let max_output = 16 + (4.0 * max_input as f32 * rate) as usize;
        let buf_0 = vec![Complex32::new(0.0, 0.0); max_input];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); max_output];

        // run numerous blocks
        for _b in 0..8 {
            for (_i, &size) in sizes.iter().enumerate() {
                let num_input = size;
                let num_output = q.get_num_output(num_input);
                let num_written = q.execute(&buf_0[..num_input], &mut buf_1[..]).unwrap();
                assert_eq!(num_output, num_written);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_num_output_0)]
    fn test_msresamp_crcf_num_output_0() { testbench_msresamp_crcf_num_output(1.00); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_num_output_1)]
    fn test_msresamp_crcf_num_output_1() { testbench_msresamp_crcf_num_output(1e3); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_num_output_2)]
    fn test_msresamp_crcf_num_output_2() { testbench_msresamp_crcf_num_output(1e-3); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_num_output_3)]
    fn test_msresamp_crcf_num_output_3() { testbench_msresamp_crcf_num_output(2f32.sqrt()); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_num_output_4)]
    fn test_msresamp_crcf_num_output_4() { testbench_msresamp_crcf_num_output(17f32.sqrt()); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_num_output_5)]
    fn test_msresamp_crcf_num_output_5() { testbench_msresamp_crcf_num_output(1.0 / std::f32::consts::PI); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_num_output_6)]
    fn test_msresamp_crcf_num_output_6() { testbench_msresamp_crcf_num_output(8.0f32.exp()); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_num_output_7)]
    fn test_msresamp_crcf_num_output_7() { testbench_msresamp_crcf_num_output((-8.0f32).exp()); }

    fn testbench_msresamp_crcf_max_input(rate: f32) {
        // create object
        let as_ = 60.0f32;
        let mut q = MsResamp::<Complex32, f32>::new(rate, as_).unwrap();

        // allocate buffers
        let max_input = 2048;
        let max_output = 16 + (4.0 * max_input as f32 * rate.max(1.0)) as usize;
        let buf_0 = vec![Complex32::new(0.0, 0.0); max_input];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); max_output];

        // test various output limits
        for output_limit in [1, 2, 3, 5, 10, 20, 50, 100] {
            let num_input = q.get_max_input(output_limit);
            let num_output = q.get_num_output(num_input);

            // get_max_input returns the max inputs that produce at most output_limit outputs
            assert!(
                num_output <= output_limit,
                "rate={}, limit={}, num_input={}, num_output={}",
                rate, output_limit, num_input, num_output
            );

            // verify that one more input would exceed the limit
            let num_output_more = q.get_num_output(num_input + 1);
            assert!(
                num_output_more > output_limit,
                "rate={}, limit={}, num_input+1={}, num_output_more={}",
                rate, output_limit, num_input + 1, num_output_more
            );

            // actually run the resampler to verify
            let num_written = q.execute(&buf_0[..num_input], &mut buf_1).unwrap();
            assert_eq!(num_output, num_written);
        }
    }

    #[test]
    fn test_msresamp_crcf_max_input_0() { testbench_msresamp_crcf_max_input(1.00); }

    #[test]
    fn test_msresamp_crcf_max_input_1() { testbench_msresamp_crcf_max_input(0.50); }

    #[test]
    fn test_msresamp_crcf_max_input_2() { testbench_msresamp_crcf_max_input(2.0); }

    #[test]
    fn test_msresamp_crcf_max_input_3() { testbench_msresamp_crcf_max_input(0.127115323); }

    #[test]
    fn test_msresamp_crcf_max_input_4() { testbench_msresamp_crcf_max_input(std::f32::consts::PI); }

    #[test]
    fn test_msresamp_crcf_max_input_5() { testbench_msresamp_crcf_max_input(1.0 / std::f32::consts::PI); }

    #[test]
    #[autotest_annotate(autotest_msresamp_crcf_copy)]
    fn test_msresamp_crcf_copy() {
        // create initial object
        let rate = 0.071239213987520f32;
        let mut q0 = MsResamp::<Complex32, f32>::new(rate, 60.0f32).unwrap();

        // run samples through filter
        let buf_len = 640;
        let mut buf = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); buf_len];
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); buf_len];

        for i in 0..buf_len {
            buf[i] = Complex32::new(randnf(), randnf());
        }

        let _nw_0 = q0.execute(&buf, &mut buf_0).unwrap();

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters and check equality
        for i in 0..buf_len {
            buf[i] = Complex32::new(randnf(), randnf());
        }

        let nw_0 = q0.execute(&buf, &mut buf_0).unwrap();
        let nw_1 = q1.execute(&buf, &mut buf_1).unwrap();

        // check that the same number of samples were written
        assert_eq!(nw_0, nw_1);

        // check output sample values
        assert_eq!(&buf_0[..nw_0], &buf_1[..nw_1]);
    }
}