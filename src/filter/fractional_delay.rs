use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::{FirPolyphaseFilter, Sample};

/// Delay line with an adjustable, fractional delay.
///
/// The delay can be any value from zero through a fixed maximum, in steps of
/// `1 / num_filters` samples. A polyphase filter bank supplies the fractional
/// part, and its filter adds `filter_semi_length` samples of delay on top of
/// the requested delay.
///
/// The interpolating filter only approximates an ideal fractional delay, and
/// its gain falls near the Nyquist frequency. With the default filter and a
/// delay of 0.5 samples, the gain is about -3 dB at 0.45 cycles per sample
/// and -16 dB at 0.49.
///
/// Feed input with [`Self::push`] and read the delayed signal with
/// [`Self::execute`], or do both at once with [`Self::execute_block`]. The
/// delay can change at any time with [`Self::set_delay`] without clearing the
/// history.
#[derive(Clone, Debug)]
#[doc(alias = "Fdelay")]
pub struct FractionalDelay<T, Coeff = T> {
    nmax: usize,
    m: usize,
    npfb: usize,
    delay: f32,
    w: Window<T>,
    pfb: FirPolyphaseFilter<T, Coeff>,
    w_index: usize,
    f_index: usize,
}

impl<T, Coeff> FractionalDelay<T, Coeff>
where
    Coeff: Sample,
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates a fractional delay line.
    ///
    /// The delay can be set anywhere from zero to `maximum_delay` samples.
    /// Each requested delay is rounded to the nearest `1 / num_filters` of a
    /// sample. The interpolating filter spans `filter_semi_length` samples on
    /// each side of its center and delays the output by that many samples.
    ///
    /// # Warning
    ///
    /// A new delay line reports a delay of zero through [`Self::delay`], but
    /// it actually delays by one whole sample until the first
    /// [`Self::set_delay`] call. Call `set_delay(0.0)` first if the output
    /// must line up with the reported delay.
    ///
    /// # Errors
    ///
    /// Returns an error when `maximum_delay`, `filter_semi_length`, or
    /// `num_filters` is zero.
    ///
    /// # Panics
    ///
    /// Panics when the filter length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps and the design is
    /// invalid.
    pub fn new(maximum_delay: usize, filter_semi_length: usize, num_filters: usize) -> Result<Self> {
        if maximum_delay == 0 {
            return Err(Error::Config("maximum delay must be greater than zero".into()));
        }
        if filter_semi_length == 0 {
            return Err(Error::Config("filter semi-length must be greater than zero".into()));
        }
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than zero".into()));
        }

        let w = Window::new(maximum_delay + 1)?;
        let pfb = FirPolyphaseFilter::new_kaiser_simple(num_filters, filter_semi_length)?;

        let mut q = Self {
            nmax: maximum_delay,
            m: filter_semi_length,
            npfb: num_filters,
            delay: 0.0,
            w,
            pfb,
            w_index: maximum_delay - 1,
            f_index: 0,
        };

        q.reset();
        Ok(q)
    }

    /// Creates a delay line with the default filter configuration.
    ///
    /// The filter has a semi-length of eight samples and 64 phases, so it adds
    /// eight samples of delay and resolves the delay to 1/64 of a sample.
    ///
    /// # Errors
    ///
    /// Returns an error when `maximum_delay` is zero.
    pub fn from_max_delay(maximum_delay: usize) -> Result<Self> {
        Self::new(maximum_delay, 8, 64)
    }

    /// Clears the history and sets the reported delay to zero.
    ///
    /// Until the next [`Self::set_delay`] call, the line actually delays by one
    /// whole sample, not zero, as a newly created delay line does. See the
    /// warning on [`Self::new`].
    pub fn reset(&mut self) {
        self.delay = 0.0;
        self.w_index = self.nmax - 1;
        self.f_index = 0;
        self.w.reset();
        self.pfb.reset();
    }

    /// Returns the delay last set with [`Self::set_delay`], in samples.
    ///
    /// The value is the requested delay, before rounding to the nearest
    /// filter phase.
    pub fn delay(&self) -> f32 {
        self.delay
    }

    /// Sets the delay in samples.
    ///
    /// The delay must lie in `[0, maximum_delay]` and is rounded to the nearest
    /// `1 / num_filters` of a sample. The input history is kept.
    ///
    /// The change does not take full effect at once. The interpolating filter
    /// still holds samples taken under the old delay, so the output blends the
    /// two delays until those samples have passed through. It matches the new
    /// delay after `2 * filter_semi_length - 1` more outputs. When both delays
    /// are whole numbers of samples, the output instead keeps the old delay for
    /// `filter_semi_length` outputs and then switches.
    ///
    /// # Errors
    ///
    /// Returns an error when `delay` is negative or greater than
    /// `maximum_delay`. The delay is unchanged on error.
    pub fn set_delay(&mut self, delay: f32) -> Result<()> {
        if delay < 0.0 {
            return Err(Error::Config("delay cannot be negative".into()));
        }
        if delay > self.nmax as f32 {
            return Err(Error::Config(format!("delay ({}) cannot exceed maximum ({})", delay, self.nmax)));
        }

        let offset = self.nmax as f32 - delay;
        let intpart = offset.floor() as i32;
        let fracpart = offset - intpart as f32;

        self.w_index = intpart as usize;
        self.f_index = (self.npfb as f32 * fracpart).round() as usize;
        while self.f_index >= self.npfb {
            self.w_index += 1;
            self.f_index -= self.npfb;
        }

        if self.w_index > self.nmax {
            return Err(Error::Internal("window index exceeds maximum".into()));
        }

        self.delay = delay;
        Ok(())
    }

    /// Adds `delta` samples to the current delay.
    ///
    /// # Errors
    ///
    /// Returns an error when the new delay is negative or greater than
    /// `maximum_delay`. The delay is unchanged on error.
    pub fn adjust_delay(&mut self, delta: f32) -> Result<()> {
        self.set_delay(self.delay + delta)
    }

    /// Returns the largest delay that can be set, in samples.
    pub fn nmax(&self) -> usize {
        self.nmax
    }

    /// Returns the filter semi-length in samples.
    ///
    /// This is also the delay the filter adds on top of the requested delay.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Returns the number of filter phases.
    ///
    /// The delay resolution is one over this value, in samples.
    pub fn npfb(&self) -> usize {
        self.npfb
    }

    /// Pushes one input sample into the delay line.
    ///
    /// Call [`Self::execute`] afterwards to read the matching output.
    pub fn push(&mut self, input: T) {
        self.w.push(input);
        self.pfb.push(self.w.index(self.w_index).unwrap());
    }

    /// Pushes a block of input samples into the delay line.
    ///
    /// This is the same as calling [`Self::push`] for each sample, and no
    /// output is produced.
    pub fn write(&mut self, input: &[T]) {
        for &xi in input {
            self.push(xi);
        }
    }

    /// Returns the delayed output for the most recently pushed sample.
    pub fn execute(&mut self) -> Result<T> {
        self.pfb.execute(self.f_index)
    }

    /// Delays a block of input samples and writes one output per input.
    ///
    /// `output` must hold at least `input.len()` samples. Extra output is left
    /// unchanged. The results match calling [`Self::push`] and
    /// [`Self::execute`] for each sample, apart from floating-point rounding.
    /// Returns `input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`. The delay line
    /// and `output` are unchanged on error.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Config("output block length must be at least the input length".into()));
        }
        let n = input.len();
        let w_index = self.w_index;
        let f_index = self.f_index;
        let pfb = &mut self.pfb;
        let mut result = Ok(());

        self.w.execute_block_contiguous(&input[..n], |indices, samples| {
            if result.is_err() {
                return;
            }

            let delayed = &samples[w_index..w_index + indices.len()];
            result = pfb.execute_block(f_index, delayed, &mut output[indices]).map(|_| ());
        });

        result.map(|()| n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::test_helpers::validate_block_execute;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    fn testbench_fdelay_rrrf(nmax: usize, m: usize, npfb: usize, delay: f32) {
        let tol = 0.01f32;
        let num_samples = nmax + 2 * m; // number of samples to run

        // create delay object and split delay between set and adjust methods
        let mut q = FractionalDelay::<f32, f32>::new(nmax, m, npfb).unwrap();
        q.set_delay(delay * 0.7).unwrap();
        q.adjust_delay(delay * 0.3).unwrap();

        // ensure object is configured properly
        assert_eq!(q.nmax(), nmax);
        assert_eq!(q.m(), m);
        assert_eq!(q.npfb(), npfb);
        assert_abs_diff_eq!(q.delay(), delay, epsilon = 1e-6);

        // generate impulse and propagate through object
        let mut x = vec![0.0f32; num_samples];
        let mut y = vec![0.0f32; num_samples];
        // generate input
        x[0] = 1.0;

        // run filter
        q.execute_block(&x, &mut y).unwrap();

        // estimate delay; assumes input is impulse and uses phase at
        // single point of frequency estimate evaluation
        let fc = 0.1 / num_samples as f32; // sufficiently small
        let mut v = Complex32::new(0.0, 0.0);
        for i in 0..num_samples {
            v += y[i] * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * i as f32);
        }
        let delay_est = v.arg() / (2.0 * std::f32::consts::PI * fc) - m as f32;

        // verify delay
        assert_abs_diff_eq!(delay_est, delay, epsilon = tol);
    }

    // nominal delays
    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_0)]
    fn test_fdelay_rrrf_0() {
        testbench_fdelay_rrrf(200, 12, 64, 0.0);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_1)]
    fn test_fdelay_rrrf_1() {
        testbench_fdelay_rrrf(200, 12, 64, 0.0001);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_2)]
    fn test_fdelay_rrrf_2() {
        testbench_fdelay_rrrf(200, 12, 64, 0.1);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_3)]
    fn test_fdelay_rrrf_3() {
        testbench_fdelay_rrrf(200, 12, 64, 0.9);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_4)]
    fn test_fdelay_rrrf_4() {
        testbench_fdelay_rrrf(200, 12, 64, 0.9999);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_5)]
    fn test_fdelay_rrrf_5() {
        testbench_fdelay_rrrf(200, 12, 64, 16.99);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_6)]
    fn test_fdelay_rrrf_6() {
        testbench_fdelay_rrrf(200, 12, 64, 17.00);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_7)]
    fn test_fdelay_rrrf_7() {
        testbench_fdelay_rrrf(200, 12, 64, 17.01);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_8)]
    fn test_fdelay_rrrf_8() {
        testbench_fdelay_rrrf(200, 12, 64, 199.9);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_9)]
    fn test_fdelay_rrrf_9() {
        testbench_fdelay_rrrf(200, 12, 64, 200.0);
    }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_config)]
    fn test_fdelay_rrrf_config() {
        // default configurations
        let nmax: usize = 200;
        let m: usize = 12;
        let npfb: usize = 64;

        // test invalid configurations, normal construction
        assert!(FractionalDelay::<f32, f32>::new(0, m, npfb).is_err());
        assert!(FractionalDelay::<f32, f32>::new(nmax, 0, npfb).is_err());
        assert!(FractionalDelay::<f32, f32>::new(nmax, m, 0).is_err());

        // test invalid configurations, default construction
        assert!(FractionalDelay::<f32, f32>::from_max_delay(0).is_err());

        // create proper object but test invalid internal configurations
        let mut q = FractionalDelay::<f32, f32>::from_max_delay(nmax).unwrap();

        assert!(q.set_delay(-1.0).is_err());
        assert!(q.set_delay(nmax as f32 + 1.0).is_err());
        assert!(q.adjust_delay(-1.0).is_err());

        // test valid configurations, methods
        // assert!(q.print().is_ok());
    }

    // compare push vs write methods
    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_push_write)]
    fn test_fdelay_rrrf_push_write() {
        // create two identical objects
        let mut q0 = FractionalDelay::<f32, f32>::from_max_delay(200).unwrap();
        let mut q1 = FractionalDelay::<f32, f32>::from_max_delay(200).unwrap();

        // set identical delays
        q0.set_delay(7.2280).unwrap();
        q1.set_delay(7.2280).unwrap();

        // generate pseudo-random inputs, and compare outputs
        let buf = [-1.0, 3.0, 5.0, -3.0, 5.0, 1.0, -3.0, -4.0];
        for trial in 0..20 {
            let n = trial % 8;

            // push/write samples
            for i in 0..n {
                q0.push(buf[i]);
            }
            q1.write(&buf[..n]);

            // compute outputs and compare
            let v0 = q0.execute();
            let v1 = q1.execute();
            assert_eq!(v0, v1);
        }
    }

    #[test]
    fn test_fdelay_crcf_execute_block_matches_execute() {
        let mut reference = FractionalDelay::<Complex32, f32>::new(9, 4, 16).unwrap();
        let mut block = reference.clone();
        reference.set_delay(5.375).unwrap();
        block.set_delay(5.375).unwrap();

        let x: Vec<_> = (0..257).map(|i| Complex32::new((0.13 * i as f32).sin(), (0.07 * i as f32).cos())).collect();
        let mut expected = vec![Complex32::new(0.0, 0.0); x.len()];
        let mut actual = vec![Complex32::new(0.0, 0.0); x.len()];

        for (i, &xi) in x.iter().enumerate() {
            reference.push(xi);
            expected[i] = reference.execute().unwrap();
        }

        let mut offset = 0;
        for &len in &[1, 7, 31, 3, 64, 151] {
            block.execute_block(&x[offset..offset + len], &mut actual[offset..offset + len]).unwrap();
            offset += len;
        }

        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-5);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_fdelay_execute_block_contract() {
        let q = FractionalDelay::<f32, f32>::new(20, 5, 64).unwrap();
        validate_block_execute(&q, &[1.0, -0.5, 0.25], -9.0, FractionalDelay::execute_block);
    }
}
