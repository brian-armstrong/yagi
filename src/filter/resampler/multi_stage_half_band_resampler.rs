use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter;
use crate::filter::{HalfBandResampler, Sample};

/// Cascade of half-band resamplers that changes the rate by a power of two.
///
/// The cascade interpolates or decimates by `2^num_stages`, one factor of two
/// per stage. A new cascade starts with zero-valued history. The cascade
/// compensates for the gain of the decimating stages, so a constant input
/// settles to a gain of about one in either direction.
///
/// Use [`Self::execute_block`] for streaming. It checks buffer lengths before
/// changing any state. [`Self::execute`] runs a single step with the same
/// checks. The direction-specific methods also check their buffer lengths.
/// With zero stages, the methods copy input to output unchanged.
#[derive(Clone, Debug)]
#[doc(alias = "MsResamp2")]
pub struct MultiStageHalfBandResampler<T, Coeff = T> {
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
    resamp2: Vec<HalfBandResampler<T, Coeff>>,
    // scratch for the execute_block path. initializes empty
    block0: Vec<T>,
    block1: Vec<T>,
}

/// Direction of a resampler's sample-rate change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResampType {
    /// Lower the sample rate.
    Decim,
    /// Raise the sample rate.
    Interp,
}

impl<T, Coeff> MultiStageHalfBandResampler<T, Coeff>
where
    Coeff: Sample,
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates a half-band cascade with `num_stages` stages.
    ///
    /// # Arguments
    ///
    /// - `num_stages`: in `[0, 16]`. The rate changes by `2^num_stages`.
    /// - `cutoff_frequency`: in cycles per sample at the lower rate, which is
    ///   the output when decimating and the input when interpolating. Must lie
    ///   in `(0, 0.5)`. It sets the transition width of each stage rather than
    ///   an exact passband edge.
    /// - `center_frequency`: must be zero. Other center frequencies are not
    ///   supported yet.
    /// - `stopband_attenuation`: in dB.
    ///
    /// # Errors
    ///
    /// Returns an error when `num_stages` is above 16, `cutoff_frequency` is
    /// nonfinite or outside its range, `center_frequency` is nonzero, or a
    /// stage design rejects `stopband_attenuation`.
    pub fn new(
        resampler_type: ResampType,
        num_stages: usize,
        cutoff_frequency: f32,
        center_frequency: f32,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        if num_stages > 16 {
            return Err(Error::Config("number of stages should not exceed 16".into()));
        }
        if !(cutoff_frequency > 0.0 && cutoff_frequency < 0.5) {
            return Err(Error::Config("cut-off frequency must be in (0,0.5)".into()));
        }
        if center_frequency != 0.0 {
            return Err(Error::Config("non-zero center frequency not yet supported".into()));
        }

        let rate = 1 << num_stages;
        let mut q = Self {
            type_: resampler_type,
            num_stages,
            rate,
            fc: cutoff_frequency,
            f0: center_frequency,
            as_: stopband_attenuation,
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
            fc = if i == 1 { (0.5 - fc) / 2.0 } else { 0.5 * fc };
            f0 *= 0.5;
            let ft = 2.0 * (0.25 - fc);

            // compute filter length
            let h_len = filter::fir::design::estimate_req_filter_len(ft, as_)?;
            let m = ((h_len as f32 - 1.0) / 4.0).ceil() as usize;

            q.fc_stage[i] = fc;
            q.f0_stage[i] = f0;
            q.as_stage[i] = as_;
            q.m_stage[i] = m.max(3);

            // create half-band resampler
            q.resamp2.push(HalfBandResampler::<T, Coeff>::new(q.m_stage[i], q.f0_stage[i], q.as_stage[i])?);
        }

        q.reset();
        Ok(q)
    }

    /// Clears the history of every stage to zeros.
    pub fn reset(&mut self) {
        for resamp in &mut self.resamp2 {
            resamp.reset();
        }
        self.buffer0.iter_mut().for_each(|x| *x = T::default());
        self.buffer1.iter_mut().for_each(|x| *x = T::default());
    }

    /// Returns the output rate divided by the input rate.
    pub fn rate(&self) -> f32 {
        match self.type_ {
            ResampType::Interp => self.rate as f32,
            ResampType::Decim => 1.0 / self.rate as f32,
        }
    }

    /// Returns the number of half-band stages.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Returns whether the cascade interpolates or decimates.
    pub fn resampler_type(&self) -> ResampType {
        self.type_
    }

    /// Returns the delay through the cascade in input samples.
    pub fn delay(&self) -> f32 {
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

    /// Runs one step of the cascade.
    ///
    /// When interpolating, a step consumes one input sample and writes
    /// `2^num_stages` output samples. When decimating, it consumes
    /// `2^num_stages` input samples and writes one output sample. Extra input
    /// is ignored and extra output is left unchanged. With zero stages, a step
    /// copies one sample.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` or `output` is too short for one step. The
    /// cascade and `output` are unchanged on error.
    pub fn execute(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        let (required_input, required_output) = match self.type_ {
            ResampType::Interp => (1, 1usize << self.num_stages),
            ResampType::Decim => (self.rate, 1),
        };
        if input.len() < required_input {
            return Err(Error::Config("input length too small for half-band resampler step".into()));
        }
        if output.len() < required_output {
            return Err(Error::Config("output length too small for half-band resampler step".into()));
        }

        if self.num_stages == 0 {
            output[0] = input[0];
            return Ok(());
        }

        match self.type_ {
            ResampType::Interp => self.interp_execute(input[0], output),
            ResampType::Decim => {
                output[0] = self.decim_execute(input)?;
                Ok(())
            }
        }
    }

    /// Consumes one input sample and writes `2^num_stages` output samples.
    ///
    /// Extra output is left unchanged. With zero stages, this copies `input`
    /// to `output[0]`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` holds fewer than `2^num_stages` samples.
    /// The cascade and `output` are unchanged on error.
    pub fn interp_execute(&mut self, input: T, output: &mut [T]) -> Result<()> {
        if output.len() < self.rate {
            return Err(Error::Config("output length too small for half-band interpolation step".into()));
        }
        let mut b0 = self.buffer0.as_mut_slice();
        let mut b1 = self.buffer1.as_mut_slice();

        b0[0] = input;

        for s in 0..self.num_stages {
            let k = 1 << s;

            for i in 0..k {
                self.resamp2[s].interp_execute(b0[i], &mut b1[2 * i..2 * i + 2])?;
            }

            std::mem::swap(&mut b0, &mut b1);
        }

        let k = 1 << self.num_stages;
        let (output, _) = output.split_at_mut(k);
        output.copy_from_slice(b0);
        Ok(())
    }

    /// Consumes `2^num_stages` input samples and returns one output sample.
    ///
    /// Extra input is ignored. With zero stages, this returns `input[0]`.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` holds fewer than `2^num_stages` samples.
    /// The cascade is unchanged on error.
    pub fn decim_execute(&mut self, input: &[T]) -> Result<T> {
        if input.len() < self.rate {
            return Err(Error::Config("input length too small for half-band decimation step".into()));
        }
        if self.num_stages == 0 {
            return Ok(input[0] * self.zeta);
        }

        let mut b0 = &mut self.buffer0;
        let mut b1 = &mut self.buffer1;

        for s in 0..self.num_stages {
            let k = 1 << (self.num_stages - s - 1);
            let g = self.num_stages - s - 1;

            // first stage reads the input directly. later stages read the prior
            // buffer. sourcing stage 0 from `input` avoids copying it into a scratch buffer.
            let src: &[T] = if s == 0 { input } else { b0 };

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
        let last_interior_writes_b1 = self.num_stages >= 2 && self.num_stages.is_multiple_of(2);
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

    /// Grows the workspace used by [`Self::interp_execute_block`].
    ///
    /// Later calls with up to `n` input samples then run without allocating.
    /// `n * 2^num_stages` must fit in `usize`, or the workspace is sized
    /// wrongly.
    pub fn reserve_interp_block(&mut self, n: usize) {
        let (cap0, cap1) = self.interp_block_caps(n);
        self.grow_block_scratch(cap0, cap1);
    }

    /// Grows the workspace used by [`Self::decim_execute_block`].
    ///
    /// Later calls with up to `n` input samples then run without allocating.
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

    /// Grows the workspace used by [`Self::execute_block`].
    ///
    /// Later calls with up to `n` input samples then run without allocating.
    /// When interpolating, `n * 2^num_stages` must fit in `usize`.
    pub fn reserve_block(&mut self, n: usize) {
        match self.type_ {
            ResampType::Interp => self.reserve_interp_block(n),
            ResampType::Decim => self.reserve_decim_block(n),
        }
    }

    /// Consumes a block of input samples and writes the resampled output.
    ///
    /// When interpolating, `output` must hold at least
    /// `input.len() * 2^num_stages` samples. When decimating, `input.len()`
    /// must be a multiple of `2^num_stages`, and `output` must hold at least
    /// `input.len() / 2^num_stages` samples. Extra output is left unchanged.
    /// Returns the number of samples written. With zero stages, the input is
    /// copied to the output.
    ///
    /// The results can differ slightly from repeated [`Self::execute`] calls,
    /// because the block dot products may sum in a different order. The call
    /// grows an internal workspace when needed. Use [`Self::reserve_block`] to
    /// allocate it in advance.
    ///
    /// # Errors
    ///
    /// Returns an error when a decimating `input` is not a whole number of
    /// steps, the interpolated length overflows `usize`, or `output` is too
    /// short. The cascade and `output` are unchanged on error.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if self.type_ == ResampType::Decim && !input.len().is_multiple_of(self.rate) {
            return Err(Error::Config(format!(
                "decimation input length ({}) must be a multiple of the rate ({})",
                input.len(),
                self.rate,
            )));
        }

        let required_output = match self.type_ {
            ResampType::Interp => input
                .len()
                .checked_mul(self.rate)
                .ok_or_else(|| Error::Range("interpolation output length overflow".into()))?,
            ResampType::Decim => input.len() / self.rate,
        };
        if output.len() < required_output {
            return Err(Error::Config(format!(
                "output length ({}) must be at least {}",
                output.len(),
                required_output,
            )));
        }

        if self.num_stages == 0 {
            let n = input.len();
            output[..n].copy_from_slice(input);
            return Ok(n);
        }

        match self.type_ {
            ResampType::Interp => self.interp_execute_block(input, output),
            ResampType::Decim => self.decim_execute_block(input, output),
        }
    }

    /// Interpolates a block of input samples through every stage.
    ///
    /// Returns the number of samples written, `input.len() * 2^num_stages`.
    /// With zero stages, this copies `input` to `output`. Extra output is
    /// left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `input.len() * 2^num_stages` overflows or
    /// `output` is too short. The cascade and `output` are unchanged on error.
    pub fn interp_execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let n = input.len();
        let n_out = n
            .checked_mul(self.rate)
            .ok_or_else(|| Error::Range("interpolation output length overflow".into()))?;
        if output.len() < n_out {
            return Err(Error::Config("output length too small for half-band interpolation block".into()));
        }
        if self.num_stages == 0 {
            output[..n].copy_from_slice(input);
            return Ok(n);
        }

        let (cap0, cap1) = self.interp_block_caps(n);
        self.grow_block_scratch(cap0, cap1);

        // specializations: the first read reads directly from input, the last
        // write writes directly into output. in all other cases, use the b0/b1 scratch
        let mut len = n;
        for s in 0..self.num_stages {
            let last = s == self.num_stages - 1;
            let src: &[T] = if s == 0 { input } else { &self.block0[..len] };
            if last {
                self.resamp2[s].interp_execute_block(&src[..len], &mut output[..2 * len])?;
            } else {
                self.resamp2[s].interp_execute_block(&src[..len], &mut self.block1[..2 * len])?;
                std::mem::swap(&mut self.block0, &mut self.block1);
            }
            len *= 2;
        }

        // without this swap, we'd toggle which buffer is which on each run, causing
        // both to be the same size
        if self.num_stages.is_multiple_of(2) {
            std::mem::swap(&mut self.block0, &mut self.block1);
        }
        Ok(n_out)
    }

    /// Decimates a block of input samples through every stage.
    ///
    /// Returns the number of samples written, `input.len() / 2^num_stages`.
    /// With zero stages, this copies `input` to `output`. Extra output is
    /// left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `input.len()` is not a multiple of
    /// `2^num_stages` or `output` is too short. The cascade and `output` are
    /// unchanged on error.
    pub fn decim_execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let n_in = input.len();
        if !n_in.is_multiple_of(self.rate) {
            return Err(Error::Config("decimation input length must be a multiple of the rate".into()));
        }
        let n_out = n_in / self.rate;
        if output.len() < n_out {
            return Err(Error::Config("output length too small for half-band decimation block".into()));
        }
        if self.num_stages == 0 {
            output[..n_in].copy_from_slice(input);
            return Ok(n_in);
        }

        let (cap0, cap1) = self.decim_block_caps(n_in);
        self.grow_block_scratch(cap0, cap1);

        // same specializations as interp_execute_block (read input, write output)
        let mut len = n_in;
        for s in 0..self.num_stages {
            let g = self.num_stages - s - 1;
            let last = s == self.num_stages - 1;
            let src: &[T] = if s == 0 { input } else { &self.block0[..len] };
            if last {
                self.resamp2[g].decim_execute_block(&src[..len], &mut output[..len / 2])?;
            } else {
                self.resamp2[g].decim_execute_block(&src[..len], &mut self.block1[..len / 2])?;
                std::mem::swap(&mut self.block0, &mut self.block1);
            }
            len /= 2;
        }

        for yi in output[..n_out].iter_mut() {
            *yi = *yi * self.zeta;
        }

        // don't swap shorter and longer buffers across the call
        if self.num_stages.is_multiple_of(2) {
            std::mem::swap(&mut self.block0, &mut self.block1);
        }
        Ok(n_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
    use crate::utility::test_helpers::{validate_psd_signal, PsdRegion};
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    #[test]
    fn test_msresamp2_zero_stage_single_sample() {
        let input = [Complex32::new(1.0, -2.0), Complex32::new(-3.0, 0.5), Complex32::new(0.0, 0.0)];
        for type_ in [ResampType::Decim, ResampType::Interp] {
            let mut q = MultiStageHalfBandResampler::<Complex32, f32>::new(type_, 0, 0.4, 0.0, 60.0).unwrap();
            for x in input {
                let sentinel = Complex32::new(99.0, -99.0);
                let mut output = [sentinel; 2];
                q.execute(&[x, sentinel], &mut output).unwrap();
                assert_eq!(output, [x, sentinel]);

                match type_ {
                    ResampType::Decim => assert_eq!(q.decim_execute(&[x, sentinel]).unwrap(), x),
                    ResampType::Interp => {
                        q.interp_execute(x, &mut output).unwrap();
                        assert_eq!(output, [x, sentinel]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_msresamp2_zero_stage_block() {
        let input = [Complex32::new(1.0, -2.0), Complex32::new(-3.0, 0.5), Complex32::new(0.0, 0.0)];
        let sentinel = Complex32::new(99.0, -99.0);
        for type_ in [ResampType::Decim, ResampType::Interp] {
            let mut q = MultiStageHalfBandResampler::<Complex32, f32>::new(type_, 0, 0.4, 0.0, 60.0).unwrap();
            // Exercise empty blocks and changing block lengths on the same cascade.
            for n in [0, 3, 1, 0, 2] {
                for checked in [false, true] {
                    let mut output = [sentinel; 4];
                    let written = if checked {
                        q.execute_block(&input[..n], &mut output).unwrap()
                    } else {
                        match type_ {
                            ResampType::Decim => q.decim_execute_block(&input[..n], &mut output).unwrap(),
                            ResampType::Interp => q.interp_execute_block(&input[..n], &mut output).unwrap(),
                        }
                    };
                    assert_eq!(written, n);
                    assert_eq!(&output[..n], &input[..n]);
                    assert!(output[n..].iter().all(|&x| x == sentinel));
                }
            }
        }
    }

    fn testbench_msresamp2_crcf_interp(num_stages: usize, fc: f32, as_: f32) {
        // create and configure objects
        let mut resamp =
            MultiStageHalfBandResampler::<Complex32, f32>::new(ResampType::Interp, num_stages, fc, 0.0, as_).unwrap();
        let delay = resamp.delay();

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
        #[rustfmt::skip]
        let regions = vec![
            PsdRegion { fmin: -0.5, fmax: -f1, pmin:  0.0, pmax: -as_, test_lo: false, test_hi: true },
            PsdRegion { fmin: -f0,  fmax: f0,  pmin: -0.1, pmax:  0.1, test_lo: true,  test_hi: true },
            PsdRegion { fmin: f1,   fmax: 0.5, pmin:  0.0, pmax: -as_, test_lo: false, test_hi: true },
        ];

        assert!(validate_psd_signal(&buf, &regions).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_01)]
    fn test_msresamp2_crcf_interp_01() {
        testbench_msresamp2_crcf_interp(1, 0.25, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_02)]
    fn test_msresamp2_crcf_interp_02() {
        testbench_msresamp2_crcf_interp(2, 0.25, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_03)]
    fn test_msresamp2_crcf_interp_03() {
        testbench_msresamp2_crcf_interp(3, 0.25, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_04)]
    fn test_msresamp2_crcf_interp_04() {
        testbench_msresamp2_crcf_interp(4, 0.25, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_05)]
    fn test_msresamp2_crcf_interp_05() {
        testbench_msresamp2_crcf_interp(1, 0.45, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_06)]
    fn test_msresamp2_crcf_interp_06() {
        testbench_msresamp2_crcf_interp(2, 0.45, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_07)]
    fn test_msresamp2_crcf_interp_07() {
        testbench_msresamp2_crcf_interp(3, 0.45, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_08)]
    fn test_msresamp2_crcf_interp_08() {
        testbench_msresamp2_crcf_interp(4, 0.45, 60.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_09)]
    fn test_msresamp2_crcf_interp_09() {
        testbench_msresamp2_crcf_interp(3, 0.45, 80.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_10)]
    fn test_msresamp2_crcf_interp_10() {
        testbench_msresamp2_crcf_interp(3, 0.45, 90.0);
    }

    #[test]
    #[autotest_annotate(autotest_msresamp2_crcf_interp_11)]
    fn test_msresamp2_crcf_interp_11() {
        testbench_msresamp2_crcf_interp(3, 0.45, 100.0);
    }

    // #[test]
    // fn test_msresamp2_crcf_interp_12() { testbench_msresamp2_crcf_interp(3, 0.45, 120.0); }

    #[test]
    #[autotest_annotate(autotest_msresamp2_copy)]
    fn test_msresamp2_copy() {
        // create original resampler
        let num_stages = 4;
        let mut q0 =
            MultiStageHalfBandResampler::<Complex32, f32>::new(ResampType::Interp, num_stages, 0.4, 0.0, 60.0).unwrap();

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

        let mut q_sample =
            MultiStageHalfBandResampler::<Complex32, f32>::new(type_, num_stages, 0.4, 0.0, 60.0).unwrap();
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

        let mut q_sample =
            MultiStageHalfBandResampler::<Complex32, f32>::new(ResampType::Decim, num_stages, 0.4, 0.0, 60.0).unwrap();
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
        let mut q =
            MultiStageHalfBandResampler::<Complex32, f32>::new(ResampType::Decim, num_stages, 0.4, 0.0, 60.0).unwrap();
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
            let n_in = match type_ {
                ResampType::Interp => 4,
                ResampType::Decim => 4 * rate,
            };
            let n_out = match type_ {
                ResampType::Interp => n_in * rate,
                ResampType::Decim => n_in / rate,
            };
            let mut q = MultiStageHalfBandResampler::<Complex32, f32>::new(type_, num_stages, 0.4, 0.0, 60.0).unwrap();
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

    #[test]
    fn test_msresamp2_single_step_capacity_without_advancing() {
        let mut interp =
            MultiStageHalfBandResampler::<Complex32, f32>::new(ResampType::Interp, 2, 0.4, 0.0, 60.0).unwrap();
        let mut interp_ref = interp.clone();
        let input = [Complex32::new(1.0, -0.5)];
        assert!(interp.execute(&input, &mut [Complex32::default(); 3]).is_err());
        let mut actual = [Complex32::new(-9.0, -9.0); 6];
        let mut expected = [Complex32::default(); 4];
        interp.execute(&input, &mut actual).unwrap();
        interp_ref.execute(&input, &mut expected).unwrap();
        assert_eq!(&actual[..4], &expected);
        assert_eq!(&actual[4..], &[Complex32::new(-9.0, -9.0); 2]);

        let mut decim =
            MultiStageHalfBandResampler::<Complex32, f32>::new(ResampType::Decim, 2, 0.4, 0.0, 60.0).unwrap();
        let mut decim_ref = decim.clone();
        let input = [Complex32::new(1.0, -0.5); 4];
        assert!(decim.execute(&input[..3], &mut [Complex32::default(); 1]).is_err());
        let mut actual = [Complex32::new(-9.0, -9.0); 2];
        let mut expected = [Complex32::default(); 1];
        decim.execute(&input, &mut actual).unwrap();
        decim_ref.execute(&input, &mut expected).unwrap();
        assert_eq!(actual[0], expected[0]);
        assert_eq!(actual[1], Complex32::new(-9.0, -9.0));
    }
}
