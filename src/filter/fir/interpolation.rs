use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::{self, Sample};
use std::f32::consts::PI;

/// Finite impulse response filter that interpolates by a fixed factor.
///
/// The filter produces a group of `interpolation_factor` output samples for
/// each input sample. The coefficients are split into that many polyphase
/// branches, and output `p` of each group comes from branch `p`. A new filter
/// starts with zero-valued history.
///
/// Use [`Self::execute`] to process one input sample at a time, or
/// [`Self::execute_block`] to process many.
#[derive(Clone, Debug)]
pub struct FirInterpolationFilter<T, Coeff = T> {
    h_sub_len: usize,
    interpolation_factor: usize,
    w: Window<T>,
    bank: filter::FirPolyphaseFilterBank<T, Coeff>,
    block_scratch: Vec<T>,
}

impl<T, Coeff> FirInterpolationFilter<T, Coeff>
where
    Coeff: Sample,
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates an interpolation filter from existing coefficients.
    ///
    /// The coefficients are in time order at the output rate, with
    /// `coefficients[0]` applied to the newest sample. When their length is
    /// not a multiple of `interpolation_factor`, they are padded with zeros at
    /// the end. The output scale starts at one.
    ///
    /// # Errors
    ///
    /// Returns an error when `interpolation_factor` is less than two or
    /// `coefficients` has fewer than `interpolation_factor` entries.
    pub fn new(interpolation_factor: usize, coefficients: &[Coeff]) -> Result<Self> {
        if interpolation_factor < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }
        if coefficients.len() < interpolation_factor {
            return Err(Error::Config("filter length cannot be less than interp factor".into()));
        }

        let mut h_sub_len = 0;
        while interpolation_factor * h_sub_len < coefficients.len() {
            h_sub_len += 1;
        }

        let h_len_padded = interpolation_factor * h_sub_len;
        let mut h_padded = vec![Coeff::zero(); h_len_padded];
        h_padded[..coefficients.len()].clone_from_slice(coefficients);

        let w = Window::new(h_sub_len)?;
        let bank = filter::FirPolyphaseFilterBank::new(interpolation_factor, &h_padded)?;

        Ok(Self { h_sub_len, interpolation_factor, w, bank, block_scratch: Vec::new() })
    }

    /// Creates a Kaiser-windowed interpolation filter.
    ///
    /// The filter spans `filter_delay` input symbols on each side of its
    /// center tap, for `2 * interpolation_factor * filter_delay + 1` taps in
    /// total before padding. The cutoff is `0.5 / interpolation_factor` cycles
    /// per output sample, which is the Nyquist frequency of the input.
    /// Stopband attenuation is given in dB.
    ///
    /// A constant input produces outputs of about the same level, so no scale
    /// correction is needed for unity gain.
    ///
    /// # Errors
    ///
    /// Returns an error when `interpolation_factor` is less than two,
    /// `filter_delay` is zero, or `stopband_attenuation` is nonfinite, zero or
    /// negative.
    ///
    /// # Panics
    ///
    /// Panics when the tap count overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the count wraps and the design is
    /// invalid.
    pub fn new_kaiser(interpolation_factor: usize, filter_delay: usize, stopband_attenuation: f32) -> Result<Self> {
        if interpolation_factor < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }
        if filter_delay == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if !stopband_attenuation.is_finite() || stopband_attenuation <= 0.0 {
            return Err(Error::Config("stop-band attenuation must be finite and greater than zero".into()));
        }

        let h_len = 2 * interpolation_factor * filter_delay + 1;
        let cutoff_frequency = 0.5 / interpolation_factor as f32;
        let hf = filter::fir::design::fir_design_kaiser(h_len, cutoff_frequency, stopband_attenuation, 0.0)?;

        let hc: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();
        Self::new(interpolation_factor, &hc)
    }

    /// Creates an interpolation filter from a pulse-shaping prototype.
    ///
    /// # Arguments
    ///
    /// - `interpolation_factor`: at least two.
    /// - `filter_delay`: filter span on each side of the center tap, in input
    ///   symbols. Must be nonzero.
    /// - `excess_bandwidth`: rolloff factor, in `[0, 1]`.
    /// - `fractional_delay`: shift of the impulse response, in output samples,
    ///   in `[-1, 1]`. A positive value moves it earlier. Some prototypes
    ///   ignore it.
    ///
    /// For most prototypes, a constant input produces outputs of about the
    /// same level. The GMSK prototypes are an exception, so check their level
    /// before relying on it.
    ///
    /// # Errors
    ///
    /// Returns an error when `interpolation_factor` is less than two,
    /// `filter_delay` is zero, either `excess_bandwidth` or `fractional_delay`
    /// is nonfinite or outside its range, or the selected prototype rejects the
    /// configuration.
    pub fn new_prototype(
        filter_shape: filter::FirFilterShape,
        interpolation_factor: usize,
        filter_delay: usize,
        excess_bandwidth: f32,
        fractional_delay: f32,
    ) -> Result<Self> {
        if interpolation_factor < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }
        if filter_delay == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if !(0.0..=1.0).contains(&excess_bandwidth) {
            return Err(Error::Config("filter excess bandwidth factor must be in [0,1]".into()));
        }
        if !(-1.0..=1.0).contains(&fractional_delay) {
            return Err(Error::Config("filter fractional sample delay must be in [-1,1]".into()));
        }

        let h = filter::fir::design::fir_design_prototype(
            filter_shape,
            interpolation_factor,
            filter_delay,
            excess_bandwidth,
            fractional_delay,
        )?;

        let hc: Vec<Coeff> = h.iter().map(|&x| x.into()).collect();
        Self::new(interpolation_factor, &hc)
    }

    /// Creates a linear interpolation filter.
    ///
    /// The outputs trace straight lines between consecutive input samples,
    /// one input sample late. With `L = interpolation_factor`, output `p` of
    /// the group for input `x[n]` is `(1 - p / L) * x[n - 1] + (p / L) * x[n]`.
    ///
    /// # Errors
    ///
    /// Returns an error when `interpolation_factor` is less than two.
    pub fn new_linear(interpolation_factor: usize) -> Result<Self> {
        if interpolation_factor < 1 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }

        let mut hc = vec![Coeff::zero(); 2 * interpolation_factor];
        for i in 0..interpolation_factor {
            hc[i] = (i as f32 / interpolation_factor as f32).into();
            hc[interpolation_factor + i] = (1.0 - i as f32 / interpolation_factor as f32).into();
        }

        Self::new(interpolation_factor, &hc)
    }

    /// Creates an interpolation filter with a sine-squared impulse response.
    ///
    /// The response spans `filter_semi_length` input samples on each side of
    /// its peak, for `2 * filter_semi_length * interpolation_factor` taps in
    /// total. A constant input produces outputs of `filter_semi_length` times
    /// its level. Set the scale to `1 / filter_semi_length` with
    /// [`Self::set_scale`] for unity gain.
    ///
    /// # Errors
    ///
    /// Returns an error when `interpolation_factor` is less than two or
    /// `filter_semi_length` is zero.
    ///
    /// # Panics
    ///
    /// Panics when the tap count overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the count wraps and the design is
    /// invalid.
    pub fn new_window(interpolation_factor: usize, filter_semi_length: usize) -> Result<Self> {
        if interpolation_factor < 1 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }
        if filter_semi_length < 1 {
            return Err(Error::Config("filter semi-length must be greater than 0".into()));
        }

        let h_len = 2 * filter_semi_length * interpolation_factor;
        let mut hc = vec![Coeff::zero(); h_len];
        for (i, hci) in hc.iter_mut().enumerate() {
            *hci = (PI * i as f32 / (2 * filter_semi_length * interpolation_factor) as f32).sin().powi(2).into();
        }

        Self::new(interpolation_factor, &hc)
    }

    /// Clears the sample history to zeros.
    ///
    /// The scale is unchanged.
    pub fn reset(&mut self) {
        self.w.reset()
    }

    /// Returns the interpolation factor.
    pub fn interp_rate(&self) -> usize {
        self.interpolation_factor
    }

    /// Returns the number of taps in each polyphase branch.
    ///
    /// This is also the number of input samples each output depends on.
    pub fn sub_len(&self) -> usize {
        self.h_sub_len
    }

    /// Sets the factor applied to every output sample.
    pub fn set_scale(&mut self, scale: Coeff) {
        self.bank.set_scale(scale)
    }

    /// Returns the factor applied to every output sample.
    pub fn scale(&self) -> Coeff {
        self.bank.scale()
    }

    /// Consumes one input sample and writes one group of output samples.
    ///
    /// A group is [`Self::interp_rate`] samples, and `output` must hold
    /// exactly that many.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` does not have exactly
    /// [`Self::interp_rate`] entries. The filter and `output` are unchanged on
    /// error.
    pub fn execute(&mut self, input: T, output: &mut [T]) -> Result<()> {
        if output.len() != self.interpolation_factor {
            return Err(Error::Config("output length must equal the interpolation factor".into()));
        }
        self.w.push(input);
        self.bank.execute_all(self.w.read(), output);
        Ok(())
    }

    /// Consumes every sample of `input` and writes one group of outputs for each.
    ///
    /// `output` must hold at least `input.len() * interp_rate()` samples.
    /// Extra output is left unchanged. Returns the number of outputs written.
    ///
    /// The result is the same as calling [`Self::execute`] for each input
    /// sample, up to rounding. Block processing can sum the products in a
    /// different order, so the last bits can differ.
    ///
    /// # Errors
    ///
    /// Returns an error when `input.len() * interp_rate()` overflows `usize`
    /// or `output` is too short. The filter and `output` are unchanged on
    /// error.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        let num_output = input
            .len()
            .checked_mul(self.interpolation_factor)
            .ok_or_else(|| Error::Range("interpolator output length overflow".into()))?;
        if output.len() < num_output {
            return Err(Error::Config(format!("output length ({}) must be at least {}", output.len(), num_output,)));
        }

        let block_len = input.len().saturating_sub(self.h_sub_len - 1);
        if block_len > self.block_scratch.len() {
            self.block_scratch.resize(block_len, T::default());
        }

        let interpolation_factor = self.interpolation_factor;
        let bank = &self.bank;
        let scratch = &mut self.block_scratch;

        self.w.execute_block_contiguous(input, |indices, history| {
            let output_start = indices.start * interpolation_factor;
            let output_end = indices.end * interpolation_factor;
            bank.execute_block_all(history, &mut output[output_start..output_end], scratch);
        });

        Ok(num_output)
    }

    /// Consumes one zero-valued input sample and writes one group of outputs.
    ///
    /// Call this after the last input to drain the samples still held in the
    /// filter history.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` does not have exactly
    /// [`Self::interp_rate`] entries. The filter and `output` are unchanged on
    /// error.
    pub fn flush(&mut self, output: &mut [T]) -> Result<()> {
        self.execute(T::zero(), output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    #[test]
    fn test_firinterp_prototype_parameter_ranges() {
        for shape in [filter::FirFilterShape::Pm, filter::FirFilterShape::Hm3, filter::FirFilterShape::Rcos] {
            for delay in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -1.0001, 1.0001] {
                assert!(matches!(
                    FirInterpolationFilter::<Complex32, f32>::new_prototype(shape, 2, 3, 0.5, delay),
                    Err(Error::Config(_))
                ));
            }
            for beta in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.0001, 1.0001] {
                assert!(matches!(
                    FirInterpolationFilter::<Complex32, f32>::new_prototype(shape, 2, 3, beta, 0.0),
                    Err(Error::Config(_))
                ));
            }
        }

        for beta in [0.0, 1.0] {
            for delay in [-1.0, 0.0, 1.0] {
                assert!(FirInterpolationFilter::<Complex32, f32>::new_prototype(
                    filter::FirFilterShape::Rcos,
                    2,
                    3,
                    beta,
                    delay
                )
                .is_ok());
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_rrrf_common)]
    fn test_firinterp_rrrf_common() {
        let interp = FirInterpolationFilter::<f32, f32>::new_kaiser(17, 4, 60.0).unwrap();
        assert_eq!(interp.interp_rate(), 17);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_crcf_common)]
    fn test_firinterp_crcf_common() {
        let interp = FirInterpolationFilter::<Complex32, f32>::new_kaiser(7, 4, 60.0).unwrap();
        assert_eq!(interp.interp_rate(), 7);
    }

    #[test]
    fn test_firinterp_kaiser_retains_endpoint() {
        let interpolation_factor = 3;
        let filter_delay = 4;
        let attenuation = 60.0;
        let h_len = 2 * interpolation_factor * filter_delay + 1;
        let h =
            filter::fir::design::fir_design_kaiser(h_len, 0.5 / interpolation_factor as f32, attenuation, 0.0).unwrap();
        let mut interp =
            FirInterpolationFilter::<f32, f32>::new_kaiser(interpolation_factor, filter_delay, attenuation).unwrap();

        assert_eq!(interp.sub_len(), 2 * filter_delay + 1);

        let mut output = [0.0f32; 3];
        interp.execute(1.0, &mut output).unwrap();
        for _ in 1..=2 * filter_delay {
            interp.execute(0.0, &mut output).unwrap();
        }

        assert_abs_diff_eq!(output[0], h[h_len - 1], epsilon = 1e-7);
        assert_abs_diff_eq!(output[1], 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(output[2], 0.0, epsilon = 1e-7);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_rrrf_generic)]
    fn test_firinterp_rrrf_generic() {
        #[rustfmt::skip]
        let h: [f32; 9] = [
            -0.2762293319046737,
             1.4757679031218007,
             0.1432569489572376,
            -0.2142368750177835,
             1.3471241294836864,
             0.1166010284926269,
             0.0536534505390281,
             0.1412672462812405,
            -0.0991854372394269
        ];

        let m = 4; // firinterp factor
        let mut interp = FirInterpolationFilter::<f32, f32>::new(m, &h).unwrap();

        let x = [1.0, -1.0, 1.0, 1.0];
        let mut y = [0.0; 16];
        #[rustfmt::skip]
        let test: [f32; 16] = [
            -0.2762293319046737,
             1.4757679031218007,
             0.1432569489572376,
            -0.2142368750177835,
             1.6233534613883602,
            -1.3591668746291738,
            -0.0896034984182095,
             0.3555041212990241,
            -1.7225388986277870,
             1.3591668746291738,
             0.0896034984182095,
            -0.3555041212990241,
             1.1700802348184398,
             1.5923689316144276,
             0.1969103994962658,
            -0.0729696287365430
        ];

        let tol = 1e-6;

        for i in 0..4 {
            interp.execute(x[i], &mut y[i * m..(i + 1) * m]).unwrap();
        }

        for i in 0..16 {
            assert_abs_diff_eq!(y[i], test[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_crcf_generic)]
    fn test_firinterp_crcf_generic() {
        // h = [0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0];
        #[rustfmt::skip]
        let h: [f32; 9] = [
            -0.7393353832652201,
             0.1909821993029451,
            -1.7013834621383086,
            -0.6157406339062349,
             0.5806218191269317,
             0.0576963976148674,
            -1.0958217797368455,
            -0.6379821629743743,
             0.7019489165905530
        ];

        let m = 4; // firinterp factor
        let mut interp = FirInterpolationFilter::<Complex32, f32>::new(m, &h).unwrap();

        //  x = [1+j*0.2, -0.2+j*1.3, 0.5+j*0.3, 1.1-j*0.2]
        #[rustfmt::skip]
        let x: [Complex32; 4] = [
            Complex32::new( 1.0000e+00,  2.0000e-01),
            Complex32::new(-2.0000e-01,  1.3000e+00),
            Complex32::new( 5.0000e-01,  3.0000e-01),
            Complex32::new( 1.1000e+00, -2.0000e-01)
        ];

        let mut y = [Complex32::new(0.0, 0.0); 16];

        // z = [x(1) 0 0 0 x(2) 0 0 0 x(3) 0 0 0 x(4) 0 0 0];
        // test = filter(h,1,z)
        #[rustfmt::skip]
        let test: [Complex32; 16] = [
            Complex32::new(-0.7393353832652201, -0.1478670766530440),
            Complex32::new( 0.1909821993029451,  0.0381964398605890),
            Complex32::new(-1.7013834621383086, -0.3402766924276617),
            Complex32::new(-0.6157406339062349, -0.1231481267812470),
            Complex32::new( 0.7284888957799757, -0.8450116344193997),
            Complex32::new( 0.0194999577542784,  0.2598161386168021),
            Complex32::new(-0.7555450873091838, -2.4309628567271702),
            Complex32::new(-0.5148340361931273, -0.9280592566729803),
            Complex32::new( 0.2161568611325566,  0.6733975332035558),
            Complex32::new( 0.0839518201284991,  0.1322999766902112),
            Complex32::new(-0.6315273751217851, -1.9349833522993918),
            Complex32::new(-0.1802738843582426, -1.0140990020385570),
            Complex32::new(-0.6633477953463869,  1.2345872139588425),
            Complex32::new( 0.2389286180406733, -0.0208875205761288),
            Complex32::new(-2.4194326982205623,  0.0115301585066081),
            Complex32::new(-0.9963057787840456, -0.0682465221110653)
        ];

        let tol = 1e-6;

        for i in 0..4 {
            interp.execute(x[i], &mut y[i * m..(i + 1) * m]).unwrap();
        }

        for i in 0..16 {
            assert_abs_diff_eq!(y[i].re, test[i].re, epsilon = tol);
            assert_abs_diff_eq!(y[i].im, test[i].im, epsilon = tol);
        }
    }

    fn testbench_firinterp_crcf_nyquist(ftype: filter::FirFilterShape, m: usize, k: usize, beta: f32) {
        let tol = 1e-6;
        // create interpolator object
        let mut interp = FirInterpolationFilter::<Complex32, f32>::new_prototype(ftype, m, k, beta, 0.0).unwrap();

        // create input buffer of symbols to interpolate
        let num_symbols = k + 16;
        let mut x = vec![Complex32::new(0.0, 0.0); num_symbols]; // input symbols
        let mut y = vec![Complex32::new(0.0, 0.0); m]; // output interp buffer

        for i in 0..num_symbols {
            x[i] = Complex32::from_polar(1.0, 0.7 * i as f32);
        }

        for i in 0..num_symbols {
            // interpolate and store into output buffer
            interp.execute(x[i], &mut y).unwrap();

            // for a Nyquist filter, output should match input at
            // proper sampling time (compensating for delay)
            if i >= k {
                assert_abs_diff_eq!(x[i - k].re, y[0].re, epsilon = tol);
                assert_abs_diff_eq!(x[i - k].im, y[0].im, epsilon = tol);
            }
        }
    }

    // add specific tests
    #[test]
    #[autotest_annotate(autotest_firinterp_crcf_rnyquist_0)]
    fn test_firinterp_crcf_rnyquist_0() {
        testbench_firinterp_crcf_nyquist(filter::FirFilterShape::Kaiser, 2, 9, 0.3);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_crcf_rnyquist_1)]
    fn test_firinterp_crcf_rnyquist_1() {
        testbench_firinterp_crcf_nyquist(filter::FirFilterShape::Kaiser, 3, 9, 0.3);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_crcf_rnyquist_2)]
    fn test_firinterp_crcf_rnyquist_2() {
        testbench_firinterp_crcf_nyquist(filter::FirFilterShape::Kaiser, 7, 9, 0.3);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_crcf_rnyquist_3)]
    fn test_firinterp_crcf_rnyquist_3() {
        testbench_firinterp_crcf_nyquist(filter::FirFilterShape::Rcos, 2, 9, 0.3);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_copy)]
    fn test_firinterp_copy() {
        // create base object
        let mut q0 = FirInterpolationFilter::<Complex32, f32>::new_kaiser(3, 7, 60.0).unwrap();
        q0.set_scale(0.12345);

        // run samples through filter
        let mut buf_0 = [Complex32::new(0.0, 0.0); 3];
        for _ in 0..20 {
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.execute(v, &mut buf_0).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters in parallel
        let mut buf_1 = [Complex32::new(0.0, 0.0); 3];
        for _ in 0..60 {
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.execute(v, &mut buf_0).unwrap();
            q1.execute(v, &mut buf_1).unwrap();

            assert_eq!(buf_0, buf_1);
        }

        // objects are automatically destroyed when they go out of scope
    }

    #[test]
    fn test_firinterp_crcf_execute_block_matches_execute() {
        let mut reference = FirInterpolationFilter::<Complex32, f32>::new_kaiser(5, 7, 60.0).unwrap();
        reference.set_scale(0.37);
        let mut block = reference.clone();

        let x: Vec<_> = (0..257).map(|i| Complex32::new((0.13 * i as f32).sin(), (0.07 * i as f32).cos())).collect();
        let mut expected = vec![Complex32::new(0.0, 0.0); 5 * x.len()];
        let mut actual = vec![Complex32::new(0.0, 0.0); 5 * x.len()];

        for (i, &xi) in x.iter().enumerate() {
            reference.execute(xi, &mut expected[5 * i..5 * (i + 1)]).unwrap();
        }

        let mut offset = 0;
        for &len in &[1, 7, 31, 3, 64, 151] {
            let written =
                block.execute_block(&x[offset..offset + len], &mut actual[5 * offset..5 * (offset + len)]).unwrap();
            assert_eq!(written, 5 * len);
            offset += len;
        }

        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-5);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_firinterp_execute_block_rejects_short_output() {
        let mut interp = FirInterpolationFilter::<f32, f32>::new_kaiser(3, 4, 60.0).unwrap();
        let mut y = [0.0; 5];
        assert!(interp.execute_block(&[1.0, 2.0], &mut y).is_err());

        let mut actual = [0.0; 3];
        let mut reference = interp.clone();
        interp.execute(1.0, &mut actual).unwrap();
        let mut expected = [0.0; 3];
        reference.execute(1.0, &mut expected).unwrap();
        assert_eq!(actual, expected);

        let mut oversized = [-9.0; 8];
        assert_eq!(interp.execute_block(&[0.25, -0.5], &mut oversized).unwrap(), 6);
        assert_eq!(&oversized[6..], &[-9.0, -9.0]);
        assert_eq!(interp.execute_block(&[], &mut oversized).unwrap(), 0);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_flush)]
    fn test_firinterp_flush() {
        // create base object
        let m = 7;
        let mut q = FirInterpolationFilter::<Complex32, f32>::new_kaiser(3, m, 60.0).unwrap();

        // run samples through filter
        let mut buf = [Complex32::new(0.0, 0.0); 3];
        for _ in 0..20 {
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q.execute(v, &mut buf).unwrap();
        }

        // ensure buffer does not contain zeros
        assert!(crate::dotprod::sumsqcf(&buf) > 0.0);

        // flush buffer
        for _ in 0..q.sub_len() {
            q.flush(&mut buf).unwrap();
        }

        // ensure buffer contains only zeros
        assert_abs_diff_eq!(crate::dotprod::sumsqcf(&buf), 0.0);

        // objects are automatically destroyed when they go out of scope
    }
}
