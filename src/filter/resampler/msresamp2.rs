use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter::resampler::resamp2::{Resamp2, Resamp2Coeff};
use crate::filter;

use num_complex::ComplexFloat;

#[derive(Clone, Debug)]
pub struct MsResamp2<T, Coeff = T> {
    type_: ResampType,
    num_stages: usize,
    rate: usize,
    fc: f32,
    f0: f32,
    as_: f32,
    zeta: Coeff,
    buffer0: Vec<T>,
    buffer1: Vec<T>,
    fc_stage: Vec<f32>,
    f0_stage: Vec<f32>,
    as_stage: Vec<f32>,
    m_stage: Vec<usize>,
    resamp2: Vec<Resamp2<T, Coeff>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResampType {
    Decim,
    Interp,
}

impl<T, Coeff> MsResamp2<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32> + Resamp2Coeff,
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + From<f32> + std::ops::Mul<Coeff, Output = T>,
    [Coeff]: DotProd<T, Output = T>,
{
    pub fn new(type_: ResampType, num_stages: usize, fc: f32, f0: f32, as_: f32) -> Result<Self> {
        if num_stages > 16 {
            return Err(Error::Config("number of stages should not exceed 16".into()));
        }
        if fc <= 0.0 || fc >= 0.5 {
            return Err(Error::Config("cut-off frequency must be in (0,0.5)".into()));
        }
        if f0 != 0.0 {
            return Err(Error::Config("non-zero center frequency not yet supported".into()));
        }

        let rate = 1 << num_stages;
        let mut q = Self {
            type_,
            num_stages,
            rate,
            fc,
            f0,
            as_,
            zeta: (1.0 / rate as f32).into(),
            buffer0: vec![T::default(); rate],
            buffer1: vec![T::default(); rate],
            fc_stage: vec![0.0; num_stages],
            f0_stage: vec![0.0; num_stages],
            as_stage: vec![0.0; num_stages],
            m_stage: vec![0; num_stages],
            resamp2: Vec::with_capacity(num_stages),
        };

        // design stages
        let mut fc = q.fc;
        let mut f0 = q.f0;
        let as_ = q.as_ + 5.0;
        for i in 0..q.num_stages {
            fc = if i == 1 {
                (0.5 - fc) / 2.0
            } else {
                0.5 * fc
            };
            f0 = 0.5 * f0;
            let ft = 2.0 * (0.25 - fc);

            // compute filter length
            let h_len = filter::estimate_req_filter_len(ft, as_)?;
            let m = ((h_len as f32 - 1.0) / 4.0).ceil() as usize;

            q.fc_stage[i] = fc;
            q.f0_stage[i] = f0;
            q.as_stage[i] = as_;
            q.m_stage[i] = m.max(3);

            // create half-band resampler
            q.resamp2.push(Resamp2::<T, Coeff>::new(q.m_stage[i], q.f0_stage[i], q.as_stage[i])?);
        }

        q.reset();
        Ok(q)
    }

    pub fn reset(&mut self) {
        for resamp in &mut self.resamp2 {
            resamp.reset();
        }
        self.buffer0.iter_mut().for_each(|x| *x = T::default());
        self.buffer1.iter_mut().for_each(|x| *x = T::default());
    }

    pub fn get_rate(&self) -> f32 {
        match self.type_ {
            ResampType::Interp => self.rate as f32,
            ResampType::Decim => 1.0 / self.rate as f32,
        }
    }

    pub fn get_num_stages(&self) -> usize {
        self.num_stages
    }

    pub fn get_type(&self) -> ResampType {
        self.type_
    }

    pub fn get_delay(&self) -> f32 {
        let mut delay = 0.0;
        match self.type_ {
            ResampType::Interp => {
                for i in 0..self.num_stages {
                    let m = self.m_stage[self.num_stages - i - 1];
                    delay *= 0.5;
                    delay += m as f32;
                }
            }
            ResampType::Decim => {
                for i in 0..self.num_stages {
                    let m = self.m_stage[i];
                    delay *= 2.0;
                    delay += 2.0 * m as f32 - 1.0;
                }
            }
        }
        delay
    }

    pub fn execute(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if self.num_stages == 0 {
            y[0] = x[0];
            return Ok(());
        }

        match self.type_ {
            ResampType::Interp => self.interp_execute(x[0], y),
            ResampType::Decim => {
                y[0] = self.decim_execute(x)?;
                Ok(())
            }
        }
    }

    pub fn interp_execute(&mut self, x: T, y: &mut [T]) -> Result<()> {
        let mut b0 = self.buffer0.as_mut_slice();
        let mut b1 = self.buffer1.as_mut_slice();

        b0[0] = x;

        for s in 0..self.num_stages {
            let k = 1 << s;

            for i in 0..k {
                self.resamp2[s].interp_execute(b0[i], &mut b1[2 * i..2 * i + 2])?;
            }

            std::mem::swap(&mut b0, &mut b1);
        }

        let k = 1 << self.num_stages;
        let (y, _) = y.split_at_mut(k);
        y.copy_from_slice(b0);
        Ok(())
    }

    pub fn decim_execute(&mut self, x: &[T]) -> Result<T> {
        let mut b0 = &mut self.buffer0;
        let mut b1 = &mut self.buffer1;

        for s in 0..self.num_stages {
            let k = 1 << (self.num_stages - s - 1);
            let g = self.num_stages - s - 1;

            // first stage reads the input directly. later stages read the prior
            // buffer. sourcing stage 0 from `x` avoids copying it into a scratch buffer.
            let src: &[T] = if s == 0 { x } else { b0 };

            for i in 0..k {
                b1[i] = self.resamp2[g].decim_execute(&src[2 * i..2 * i + 2])?;
            }

            std::mem::swap(&mut b0, &mut b1);
        }

        Ok(b0[0] * self.zeta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use num_complex::Complex32;
    use crate::{random::randnf, utility::test_helpers::{validate_psd_signal, PsdRegion}};

    fn testbench_msresamp2_crcf_interp(num_stages: usize, fc: f32, as_: f32) {
        // create and configure objects
        let mut resamp = MsResamp2::<Complex32, f32>::new(ResampType::Interp, num_stages, fc, 0.0, as_).unwrap();
        let delay = resamp.get_delay();

        // generate samples and push through spgram object
        let m = 1 << num_stages; // interpolation rate
        let mut buf_len = 0;
        let mut num_blocks = 0;
        while (buf_len as f32) < 2.0 * m as f32 * delay {
            buf_len += m;
            num_blocks += 1;
        }
        let mut buf = vec![Complex32::new(0.0, 0.0); buf_len];
        
        for i in 0..num_blocks {
            let x = if i == 0 { Complex32::new(1.0, 0.0) } else { Complex32::new(0.0, 0.0) };

            // generate block of samples
            resamp.execute(&[x], &mut buf[i * m..(i + 1) * m]).unwrap();
        }

        // scale by samples/symbol
        // TODO replace with vectorcf_mulscalar when exists
        for sample in &mut buf {
            *sample /= m as f32;
        }

        // verify result
        let f0 = fc / m as f32;
        let f1 = 1.0 / m as f32 - f0;
        let regions = vec![
            PsdRegion { fmin: -0.5, fmax: -f1, pmin: 0.0, pmax: -as_, test_lo: false, test_hi: true },
            PsdRegion { fmin: -f0, fmax: f0, pmin: -0.1, pmax: 0.1, test_lo: true, test_hi: true },
            PsdRegion { fmin: f1, fmax: 0.5, pmin: 0.0, pmax: -as_, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_signal(&buf, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_01)]
    fn test_msresamp2_crcf_interp_01() { testbench_msresamp2_crcf_interp(1, 0.25, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_02)]
    fn test_msresamp2_crcf_interp_02() { testbench_msresamp2_crcf_interp(2, 0.25, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_03)]
    fn test_msresamp2_crcf_interp_03() { testbench_msresamp2_crcf_interp(3, 0.25, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_04)]
    fn test_msresamp2_crcf_interp_04() { testbench_msresamp2_crcf_interp(4, 0.25, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_05)]
    fn test_msresamp2_crcf_interp_05() { testbench_msresamp2_crcf_interp(1, 0.45, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_06)]
    fn test_msresamp2_crcf_interp_06() { testbench_msresamp2_crcf_interp(2, 0.45, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_07)]
    fn test_msresamp2_crcf_interp_07() { testbench_msresamp2_crcf_interp(3, 0.45, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_08)]
    fn test_msresamp2_crcf_interp_08() { testbench_msresamp2_crcf_interp(4, 0.45, 60.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_09)]
    fn test_msresamp2_crcf_interp_09() { testbench_msresamp2_crcf_interp(3, 0.45, 80.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_10)]
    fn test_msresamp2_crcf_interp_10() { testbench_msresamp2_crcf_interp(3, 0.45, 90.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_11)]
    fn test_msresamp2_crcf_interp_11() { testbench_msresamp2_crcf_interp(3, 0.45, 100.0); }

    // #[test]
    // fn test_msresamp2_crcf_interp_12() { testbench_msresamp2_crcf_interp(3, 0.45, 120.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_copy)]
    fn test_msresamp2_copy() {
        // create original resampler
        let num_stages = 4;
        let mut q0 = MsResamp2::<Complex32, f32>::new(
            ResampType::Interp,
            num_stages,
            0.4,
            0.0,
            60.0
        ).unwrap();

        // allocate buffers for output
        let m = 1 << num_stages; // interpolation factor
        let mut y0 = vec![Complex32::new(0.0, 0.0); m];
        let mut y1 = vec![Complex32::new(0.0, 0.0); m];

        // push samples through original object
        let num_samples = 35;
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            q0.execute(&[v], &mut y0).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run random samples through both filters and compare
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            q0.execute(&[v], &mut y0).unwrap();
            q1.execute(&[v], &mut y1).unwrap();
            assert_eq!(y0, y1);
        }

        // Rust's RAII will handle cleanup automatically
    }
}