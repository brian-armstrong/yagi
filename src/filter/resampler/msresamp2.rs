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
    // scratch for the execute_block path. initializes empty
    block0: Vec<T>,
    block1: Vec<T>,
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
    [T]: DotProd<Coeff, Output = T>,
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
            block0: Vec::new(),
            block1: Vec::new(),
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

    fn interp_block_caps(&self, n_in: usize) -> (usize, usize) {
        let n_out = n_in << self.num_stages;
        // in the interest of minimizing the size, figure out which buffer goes last
        // (later interp stages are larger)
        // e.g. for 1 stage: x -> y (no scratch needed)
        // for 2 stages: x -> b1 -> y
        // for 3 stages: x -> b1 -> b0 -> y (b0 is largest)
        // for 4 stages: x -> b1 -> b0 -> b1 -> y (b0 is largest)
        let last_interior_writes_b1 = self.num_stages >= 2 && self.num_stages % 2 == 0;
        if last_interior_writes_b1 {
            (n_out / 4, n_out / 2)
        } else {
            (n_out / 2, n_out / 4)
        }
    }

    fn decim_block_caps(&self, n_in: usize) -> (usize, usize) {
        (n_in / 4, n_in / 2)
    }

    fn grow_block_scratch(&mut self, cap0: usize, cap1: usize) {
        if cap0 > self.block0.len() {
            self.block0.resize(cap0, T::default());
        }
        if cap1 > self.block1.len() {
            self.block1.resize(cap1, T::default());
        }
    }

    /// Pre-size the block scratch for interpolating blocks of up to `n` input
    /// samples so that the first [`interp_execute_block`](Self::interp_execute_block) call does not
    /// allocate. Larger blocks still grow the scratch on demand.
    pub fn reserve_interp_block(&mut self, n: usize) {
        let (cap0, cap1) = self.interp_block_caps(n);
        self.grow_block_scratch(cap0, cap1);
    }

    /// Pre-size the block scratch for decimating blocks of up to `n` input
    /// samples so that the first [`decim_execute_block`](Self::decim_execute_block) call does not
    /// allocate. Larger blocks still grow the scratch on demand.
    pub fn reserve_decim_block(&mut self, n: usize) {
        let (cap0, cap1) = self.decim_block_caps(n);
        self.grow_block_scratch(cap0, cap1);

        let mut len = n;
        for s in 0..self.num_stages {
            let g = self.num_stages - s - 1;
            self.resamp2[g].reserve_decim_block(len);
            len /= 2;
        }
    }

    /// Pre-size the block scratch for executing blocks of up to `n` input
    /// samples so that the first [`execute_block`](Self::execute_block) call does not
    /// allocate. Larger blocks still grow the scratch on demand.
    pub fn reserve_block(&mut self, n: usize) {
        match self.type_ {
            ResampType::Interp => self.reserve_interp_block(n),
            ResampType::Decim => self.reserve_decim_block(n),
        }
    }

    /// Execute the cascade over a block of input samples.
    ///
    /// This function will generally be more efficient than running [`execute`](Self::execute)
    /// on each sample in a slice. Dot product kernels called here may run in a different
    /// order, so results can differ by floating-point rounding.
    ///
    /// # Arguments
    ///
    /// * `x` - input samples: `n` for interp, `rate * n` for decim
    /// * `y` - output samples: `rate * n` for interp, `n` for decim
    ///
    /// Returns the number of output samples written.
    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<usize> {
        if self.type_ == ResampType::Decim && x.len() % self.rate != 0 {
            return Err(Error::Config(format!(
                "decimation input length ({}) must be a multiple of the rate ({})",
                x.len(), self.rate,
            )));
        }

        let required_output = match self.type_ {
            ResampType::Interp => x.len().checked_mul(self.rate).ok_or_else(|| {
                Error::Range("interpolation output length overflow".into())
            })?,
            ResampType::Decim => x.len() / self.rate,
        };
        if y.len() < required_output {
            return Err(Error::Config(format!(
                "output length ({}) must be at least {}",
                y.len(), required_output,
            )));
        }

        if self.num_stages == 0 {
            let n = x.len();
            y[..n].copy_from_slice(x);
            return Ok(n);
        }

        match self.type_ {
            ResampType::Interp => self.interp_execute_block(x, y),
            ResampType::Decim => self.decim_execute_block(x, y),
        }
    }

    pub fn interp_execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<usize> {
        let n = x.len();
        let n_out = n << self.num_stages;

        let (cap0, cap1) = self.interp_block_caps(n);
        self.grow_block_scratch(cap0, cap1);
        let mut b0 = std::mem::take(&mut self.block0);
        let mut b1 = std::mem::take(&mut self.block1);

        // specializations: the first read reads directly from x, the last
        // write writes directly into y. in all other cases, use the b0/b1 scratch
        let mut len = n;
        for s in 0..self.num_stages {
            let last = s == self.num_stages - 1;
            let src: &[T] = if s == 0 { x } else { &b0[..len] };
            if last {
                self.resamp2[s].interp_execute_block(&src[..len], &mut y[..2 * len])?;
            } else {
                self.resamp2[s].interp_execute_block(&src[..len], &mut b1[..2 * len])?;
                std::mem::swap(&mut b0, &mut b1);
            }
            len *= 2;
        }

        // without this swap, we'd toggle which buffer is which on each run, causing
        // both to be the same size
        if self.num_stages % 2 == 0 {
            std::mem::swap(&mut b0, &mut b1);
        }
        self.block0 = b0;
        self.block1 = b1;
        Ok(n_out)
    }

    pub fn decim_execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<usize> {
        let n_in = x.len();
        let n_out = n_in >> self.num_stages;

        let (cap0, cap1) = self.decim_block_caps(n_in);
        self.grow_block_scratch(cap0, cap1);
        let mut b0 = std::mem::take(&mut self.block0);
        let mut b1 = std::mem::take(&mut self.block1);

        // same specializations as interp_execute_block (read x, write y)
        let mut len = n_in;
        for s in 0..self.num_stages {
            let g = self.num_stages - s - 1;
            let last = s == self.num_stages - 1;
            let src: &[T] = if s == 0 { x } else { &b0[..len] };
            if last {
                self.resamp2[g].decim_execute_block(&src[..len], &mut y[..len / 2])?;
            } else {
                self.resamp2[g].decim_execute_block(&src[..len], &mut b1[..len / 2])?;
                std::mem::swap(&mut b0, &mut b1);
            }
            len /= 2;
        }

        for yi in y[..n_out].iter_mut() {
            *yi = *yi * self.zeta;
        }

        // don't swap shorter and longer buffers across the call
        if self.num_stages % 2 == 0 {
            std::mem::swap(&mut b0, &mut b1);
        }
        self.block0 = b0;
        self.block1 = b1;
        Ok(n_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
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

    fn assert_complex_slices_approx_equal(actual: &[Complex32], expected: &[Complex32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-5);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-5);
        }
    }

    fn testbench_block_matches(type_: ResampType, num_stages: usize) {
        let rate = 1usize << num_stages;
        let n = 40; // output symbols for interp, output samples for decim

        let mut q_sample = MsResamp2::<Complex32, f32>::new(type_, num_stages, 0.4, 0.0, 60.0).unwrap();
        let mut q_block = q_sample.clone();

        let (num_in, num_out) = match type_ {
            ResampType::Interp => (n, n * rate),
            ResampType::Decim => (n * rate, n),
        };

        let x: Vec<Complex32> = (0..num_in).map(|_| Complex32::new(randnf(), randnf())).collect();

        let mut y_ref = vec![Complex32::new(0.0, 0.0); num_out];
        match type_ {
            ResampType::Interp => {
                for i in 0..n {
                    q_sample.execute(&x[i..i + 1], &mut y_ref[i * rate..(i + 1) * rate]).unwrap();
                }
            }
            ResampType::Decim => {
                for i in 0..n {
                    q_sample.execute(&x[i * rate..(i + 1) * rate], &mut y_ref[i..i + 1]).unwrap();
                }
            }
        }

        let mut y_block = vec![Complex32::new(0.0, 0.0); num_out];
        let written = q_block.execute_block(&x, &mut y_block).unwrap();

        assert_eq!(written, num_out);
        assert_complex_slices_approx_equal(&y_block, &y_ref);
    }

    #[test]
    fn test_msresamp2_block_interp() {
        for num_stages in 1..=4 {
            testbench_block_matches(ResampType::Interp, num_stages);
        }
    }

    #[test]
    fn test_msresamp2_block_decim() {
        for num_stages in 1..=4 {
            testbench_block_matches(ResampType::Decim, num_stages);
        }
    }

    #[test]
    fn test_msresamp2_block_resizing() {
        let num_stages = 3;
        let rate = 1usize << num_stages;
        let n = 30;

        let mut q_sample = MsResamp2::<Complex32, f32>::new(ResampType::Decim, num_stages, 0.4, 0.0, 60.0).unwrap();
        let mut q_block = q_sample.clone();

        // split the block work unevenly so the second call is larger than the
        // first, forcing the scratch buffers to regrow mid-stream.
        let split = n; // first call: n outputs; second: 2n outputs
        let total = 3 * n;
        let x: Vec<Complex32> = (0..total * rate).map(|_| Complex32::new(randnf(), randnf())).collect();

        let mut y_ref = vec![Complex32::new(0.0, 0.0); total];
        for i in 0..total {
            q_sample.execute(&x[i * rate..(i + 1) * rate], &mut y_ref[i..i + 1]).unwrap();
        }

        let mut y_block = vec![Complex32::new(0.0, 0.0); total];
        q_block.execute_block(&x[..split * rate], &mut y_block[..split]).unwrap();
        q_block.execute_block(&x[split * rate..], &mut y_block[split..]).unwrap();

        assert_complex_slices_approx_equal(&y_block, &y_ref);
    }

    #[test]
    fn test_msresamp2_block_decim_rejects_partial_input() {
        let num_stages = 3;
        let rate = 1usize << num_stages;
        let mut q = MsResamp2::<Complex32, f32>::new(ResampType::Decim, num_stages, 0.4, 0.0, 60.0).unwrap();
        let mut q_ref = q.clone();

        let bad = vec![Complex32::new(1.0, -1.0); rate + 1];
        let mut bad_output = vec![Complex32::default(); 2];
        assert!(q.execute_block(&bad, &mut bad_output).is_err());

        let x: Vec<Complex32> = (0..4 * rate).map(|_| Complex32::new(randnf(), randnf())).collect();
        let mut y = vec![Complex32::default(); 4];
        let mut y_ref = vec![Complex32::default(); 4];
        q.execute_block(&x, &mut y).unwrap();
        q_ref.execute_block(&x, &mut y_ref).unwrap();
        assert_eq!(y, y_ref);
    }

    #[test]
    fn test_msresamp2_block_rejects_short_output() {
        for &type_ in &[ResampType::Interp, ResampType::Decim] {
            let num_stages = 3;
            let rate = 1usize << num_stages;
            let n_in = match type_ { ResampType::Interp => 4, ResampType::Decim => 4 * rate };
            let n_out = match type_ { ResampType::Interp => n_in * rate, ResampType::Decim => n_in / rate };
            let mut q = MsResamp2::<Complex32, f32>::new(type_, num_stages, 0.4, 0.0, 60.0).unwrap();
            let mut q_ref = q.clone();
            let x: Vec<Complex32> = (0..n_in).map(|_| Complex32::new(randnf(), randnf())).collect();

            let mut short = vec![Complex32::default(); n_out - 1];
            assert!(q.execute_block(&x, &mut short).is_err());

            let mut y = vec![Complex32::default(); n_out];
            let mut y_ref = vec![Complex32::default(); n_out];
            q.execute_block(&x, &mut y).unwrap();
            q_ref.execute_block(&x, &mut y_ref).unwrap();
            assert_eq!(y, y_ref);
        }
    }
}
