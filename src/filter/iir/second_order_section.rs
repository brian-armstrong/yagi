use crate::error::{Error, Result};
use crate::filter::iir::design::iir_group_delay;
use crate::filter::Sample;

/// One second-order IIR filter section.
///
/// A section holds three numerator and three denominator coefficients in
/// ascending powers of `z^-1`. [`Self::execute`] runs direct form II, and
/// [`Self::execute_df1`] runs direct form I.
///
/// The two forms keep separate histories. Switching between them midstream
/// without calling [`Self::reset`] produces wrong output. To build a
/// higher-order filter from a chain of sections, use
/// [`IirFilter::new_sos`](crate::filter::IirFilter::new_sos).
#[derive(Debug, Clone)]
#[doc(alias = "IirFilterSos")]
pub struct IirSecondOrderSection<T, Coeff = T> {
    b: [Coeff; 3], // feed-forward coefficients
    a: [Coeff; 3], // feed-back coefficients

    // internal buffering
    x: [T; 3], // Direct form I  buffer (input)
    y: [T; 3], // Direct form I  buffer (output)
    v: [T; 3], // Direct form II buffer
}

impl<T, Coeff> IirSecondOrderSection<T, Coeff>
where
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    Coeff: Sample + std::ops::Mul<T, Output = T>,
{
    /// Creates a section from numerator and denominator coefficients.
    ///
    /// Both are normalized by the first denominator coefficient. A new section
    /// starts with zero-valued history.
    ///
    /// # Errors
    ///
    /// Returns an error when a coefficient is nonfinite, the first denominator
    /// coefficient is zero, or normalization produces a nonfinite coefficient.
    pub fn new(numerator_coefficients: &[Coeff; 3], denominator_coefficients: &[Coeff; 3]) -> Result<Self> {
        let mut filter = IirSecondOrderSection {
            b: [Coeff::default(); 3],
            a: [Coeff::default(); 3],
            x: [T::default(); 3],
            y: [T::default(); 3],
            v: [T::default(); 3],
        };

        // set the internal coefficients
        filter.set_coefficients(numerator_coefficients, denominator_coefficients)?;

        // clear filter state and return object
        filter.reset();
        Ok(filter)
    }

    /// Replaces the coefficients without clearing the section history.
    ///
    /// Both are normalized by the first denominator coefficient.
    ///
    /// # Errors
    ///
    /// Returns an error when a coefficient is nonfinite, the first denominator
    /// coefficient is zero, or normalization produces a nonfinite coefficient.
    /// The section and its history are unchanged on error.
    pub fn set_coefficients(
        &mut self,
        numerator_coefficients: &[Coeff; 3],
        denominator_coefficients: &[Coeff; 3],
    ) -> Result<()> {
        let b = numerator_coefficients;
        let a = denominator_coefficients;

        if b.iter().chain(a.iter()).any(|coefficient| !coefficient.abs().is_finite()) || a[0].abs() == 0.0 {
            return Err(Error::Config("IIR section coefficients must be finite with nonzero a0".into()));
        }

        // retain a0 coefficient for normalization
        let a0 = a[0];

        let normalized_b = [b[0] / a0, b[1] / a0, b[2] / a0];
        let normalized_a = [a[0] / a0, a[1] / a0, a[2] / a0];
        if normalized_b.iter().chain(normalized_a.iter()).any(|coefficient| !coefficient.abs().is_finite()) {
            return Err(Error::Config("normalized IIR section coefficients must be finite".into()));
        }

        // copy feed-forward coefficients (numerator)
        self.b = normalized_b;

        // copy feed-back coefficients (denominator)
        self.a = normalized_a;

        Ok(())
    }

    /// Clears the direct-form-I and direct-form-II histories to zeros.
    pub fn reset(&mut self) {
        self.v[0] = T::default();
        self.v[1] = T::default();
        self.v[2] = T::default();

        self.x[0] = T::default();
        self.x[1] = T::default();
        self.x[2] = T::default();

        self.y[0] = T::default();
        self.y[1] = T::default();
        self.y[2] = T::default();
    }

    /// Filters one sample with direct form II.
    #[inline]
    pub fn execute(&mut self, input: T) -> T {
        self.execute_df2(input)
    }

    /// Filters one sample with direct form I.
    ///
    /// Only the direct-form-I history advances.
    pub fn execute_df1(&mut self, input: T) -> T {
        // advance buffer input
        self.x[2] = self.x[1];
        self.x[1] = self.x[0];
        self.x[0] = input;

        // advance buffer y
        self.y[2] = self.y[1];
        self.y[1] = self.y[0];

        // compute new v
        #[rustfmt::skip]
        let v = self.x[0] * self.b[0] +
                self.x[1] * self.b[1] +
                self.x[2] * self.b[2];

        // compute new y[0]
        self.y[0] = v - self.y[1] * self.a[1] - self.y[2] * self.a[2];

        self.y[0]
    }

    /// Filters one sample with direct form II.
    ///
    /// Only the direct-form-II history advances.
    #[inline]
    pub fn execute_df2(&mut self, input: T) -> T {
        // advance buffer
        self.v[2] = self.v[1];
        self.v[1] = self.v[0];

        // compute new v[0]
        self.v[0] = input - self.a[1] * self.v[1] - self.a[2] * self.v[2];

        // compute output y
        self.b[0] * self.v[0] + self.b[1] * self.v[1] + self.b[2] * self.v[2]
    }

    /// Estimates the group delay at `fc` cycles per sample, plus a fixed
    /// offset of two samples.
    ///
    /// The offset is always added, even for a section that passes its input
    /// straight through, so the result is not the section's true group delay.
    /// For the estimate without the offset, pass the section's real
    /// coefficients to [`iir_group_delay`]. Only the real parts of the
    /// coefficients are used.
    ///
    /// # Errors
    ///
    /// Returns an error when `fc` is outside `[-0.5, 0.5]`.
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        // copy coefficients
        let b: [f32; 3] = [self.b[0].re(), self.b[1].re(), self.b[2].re()];
        let a: [f32; 3] = [self.a[0].re(), self.a[1].re(), self.a[2].re()];

        Ok(iir_group_delay(&b, &a, fc)? + 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_iirfiltsos_impulse_n2)]
    fn test_iirfiltsos_impulse_n2() {
        // initialize filter with 2nd-order low-pass butterworth filter
        let a = [1.000000000000000, -0.942809041582063, 0.333333333333333];

        let b = [0.0976310729378175, 0.1952621458756350, 0.0976310729378175];

        // create identical objects
        let mut q0 = IirSecondOrderSection::<f32, f32>::new(&b, &a).unwrap();
        let mut q1 = IirSecondOrderSection::<f32, f32>::new(&b, &a).unwrap();

        // initialize oracle; expected output (generated with octave)
        #[rustfmt::skip]
        let test = [
             9.76310729378175e-02,
             2.87309604180767e-01,
             3.35965474513536e-01,
             2.20981418970514e-01,
             9.63547883225231e-02,
             1.71836926400291e-02,
            -1.59173219853878e-02,
            -2.07348926322729e-02,
            -1.42432702548109e-02,
            -6.51705310050832e-03,
            -1.39657983602602e-03,
             8.55642936806248e-04,
             1.27223450919543e-03,
             9.14259886013424e-04,
             4.37894317157432e-04,
        ];

        let tol = 1e-4f32;

        // hit filter with impulse, compare output
        for i in 0..15 {
            // generate input
            let v = if i == 0 { 1.0f32 } else { 0.0f32 };

            // run direct-form I
            let y = q0.execute_df1(v);
            assert_abs_diff_eq!(test[i], y, epsilon = tol);

            // run direct-form II
            let y = q1.execute_df2(v);
            assert_abs_diff_eq!(test[i], y, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfiltsos_step_n2)]
    fn test_iirfiltsos_step_n2() {
        // initialize filter with 2nd-order low-pass butterworth filter
        let a = [1.000000000000000, -0.942809041582063, 0.333333333333333];

        let b = [0.0976310729378175, 0.1952621458756350, 0.0976310729378175];

        // create identical objects
        let mut q0 = IirSecondOrderSection::<f32, f32>::new(&b, &a).unwrap();
        let mut q1 = IirSecondOrderSection::<f32, f32>::new(&b, &a).unwrap();

        let test = [
            0.0976310729378175,
            0.3849406771185847,
            0.7209061516321208,
            0.9418875706026352,
            1.0382423589251584,
            1.0554260515651877,
            1.0395087295798000,
            1.0187738369475272,
            1.0045305666927162,
            0.9980135135922078,
            0.9966169337561817,
            0.9974725766929878,
            0.9987448112021832,
            0.9996590710881966,
            1.0000969654053542,
        ];

        let tol = 1e-4f32;

        // hit filter with step, compare output
        for i in 0..15 {
            // run direct-form I
            let y = q0.execute_df1(1.0);
            assert_abs_diff_eq!(test[i], y, epsilon = tol);

            // run direct-form II
            let y = q1.execute_df2(1.0);
            assert_abs_diff_eq!(test[i], y, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfiltsos_copy)]
    fn test_iirfiltsos_copy() {
        // initialize filter with 2nd-order low-pass butterworth filter
        #[rustfmt::skip]
        let a = [1.0000000000000000f32, -0.942809041582063f32, 0.3333333333333333f32];
        #[rustfmt::skip]
        let b = [0.0976310729378175f32,  0.195262145875635f32, 0.0976310729378175f32];

        // create base object
        let mut q0 = IirSecondOrderSection::<Complex32, f32>::new(&b, &a).unwrap();

        // start running input through filter
        let num_samples = 80;
        for _ in 0..num_samples {
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.execute(v);
        }

        // copy filter
        let mut q1 = q0.clone();

        // continue running through both filters
        for _ in 0..num_samples {
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            let y0 = q0.execute(v);
            let y1 = q1.execute(v);

            // compare result
            assert_eq!(y0, y1);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirfiltsos_config)]
    fn test_iirfiltsos_config() {
        // test copying/creating invalid objects
        // assert!(IirFiltSos::<f32, f32, f32>::clone(None).is_none());
        // nothing to be done - you can't clone an invalid object
    }
}
