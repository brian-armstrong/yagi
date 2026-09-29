//
// firfarrow.rs
//
// Finite impulse response Farrow filter
//

use num_complex::Complex32;

use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::fir::design::fir_group_delay;
use crate::filter::fir::design::freqresponse;
use crate::filter::fir::design::kaiser_beta_stopband_attenuation;
use crate::filter::Sample;
use crate::math::kaiser;
use crate::math::sincd;
use crate::math::{poly_val, vandermonde_workspace};

/// FIR filter with an adjustable fractional delay.
///
/// Each tap is a polynomial in the fractional delay, so
/// [`Self::set_delay`] can retune the filter without redesigning it. The total
/// delay is about `(filter_length - 1) / 2` samples plus the fractional
/// delay. A positive fractional delay makes the output later, which is the
/// opposite sign from the `mu` offset used by
/// [`crate::filter::fir::design::fir_design_kaiser`].
///
/// The fractional delay is approximate, and it falls further short of the
/// request as the cutoff drops. A low cutoff leaves the prototype nearly flat
/// across the fitted range, so changing the delay moves the filter less than
/// requested. At a cutoff of zero the delay has no effect. With 19 taps,
/// polynomial order five and 60 dB of attenuation, a delay of 0.75 gives a DC
/// group delay of 9.01 samples at a cutoff of 0.01 and 9.26 at 0.05, against a
/// nominal 9.75. [`Self::groupdelay`] reports the delay the filter actually
/// gives.
///
/// The taps are always real, even for complex `Coeff`. A new filter starts
/// with zero-valued history and zero fractional delay. Changing the delay
/// keeps the history.
#[derive(Clone, Debug)]
pub struct FirFarrowFilter<T, Coeff = T> {
    /// filter coefficients for the current delay
    h: Vec<Coeff>,
    /// `h` reversed, to convolve against the oldest-first window
    h_rev: Vec<Coeff>,
    /// scratch space for the unnormalized taps during `set_delay`
    htmp: Vec<f32>,
    /// filter length
    h_len: usize,
    /// filter cutoff frequency
    fc: f32,
    /// stop-band attenuation [dB]
    as_: f32,
    /// polynomial order
    q: usize,
    /// polynomial coefficient matrix [h_len x q+1]
    p: Vec<f32>,
    /// inverse of DC response (normalization factor)
    gamma: f32,
    /// input buffer
    w: Window<T>,
}

impl<T, Coeff> FirFarrowFilter<T, Coeff>
where
    Coeff: Sample,
    T: Sample + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Creates a fractional-delay filter.
    ///
    /// # Arguments
    ///
    /// - `filter_length`: number of taps. Must be at least two.
    /// - `polynomial_order`: degree of the polynomial in the fractional delay
    ///   that generates each tap. Must be nonzero.
    /// - `cutoff_frequency`: low-pass cutoff, in cycles per sample, in
    ///   `[0, 0.5]`.
    /// - `stopband_attenuation`: in dB. Must not be negative.
    ///
    /// # Errors
    ///
    /// Returns an error when `filter_length` is less than two,
    /// `polynomial_order` is zero, `cutoff_frequency` is outside `[0, 0.5]`,
    /// `stopband_attenuation` is negative, or the polynomial fit fails.
    ///
    /// # Panics
    ///
    /// Panics when `filter_length * (polynomial_order + 1)` overflows `usize`
    /// and overflow checks are enabled. Without overflow checks, the size
    /// wraps and the coefficient storage is too short.
    pub fn new(
        filter_length: usize,
        polynomial_order: usize,
        cutoff_frequency: f32,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        if filter_length < 2 {
            return Err(Error::Config("filter length must be greater than 1".into()));
        }
        if polynomial_order < 1 {
            return Err(Error::Config("polynomial order must be at least 1".into()));
        }
        if cutoff_frequency < 0.0 || cutoff_frequency > 0.5 {
            return Err(Error::Config("filter cutoff must be in [0,0.5]".into()));
        }
        if stopband_attenuation < 0.0 {
            return Err(Error::Config("filter stop-band attenuation must be non-negative".into()));
        }

        let mut q = Self {
            h: vec![<Coeff as From<f32>>::from(0.0); filter_length],
            h_rev: vec![<Coeff as From<f32>>::from(0.0); filter_length],
            htmp: vec![0.0; filter_length],
            h_len: filter_length,
            fc: cutoff_frequency,
            as_: stopband_attenuation,
            q: polynomial_order,
            p: vec![0.0; filter_length * (polynomial_order + 1)],
            gamma: 1.0,
            w: Window::new(filter_length)?,
        };

        q.reset();
        q.genpoly()?;
        q.set_delay(0.0)?;

        Ok(q)
    }

    /// Clears the sample history to zeros.
    ///
    /// The fractional delay is unchanged.
    pub fn reset(&mut self) {
        self.w.reset();
    }

    /// Returns the number of taps.
    pub fn length(&self) -> usize {
        self.h_len
    }

    /// Returns the degree of the polynomial behind each tap.
    pub fn order(&self) -> usize {
        self.q
    }

    /// Returns the coefficients for the current delay.
    ///
    /// They are in time order, with the first applied to the newest sample.
    pub fn coefficients(&self) -> &[Coeff] {
        &self.h
    }

    /// Sets the fractional delay in samples.
    ///
    /// Positive values make the output later. The sample history is kept.
    ///
    /// The new taps are scaled to unity DC gain. If they sum to nearly zero,
    /// a fixed scale chosen at construction is used instead, and the DC gain
    /// is not unity.
    ///
    /// # Errors
    ///
    /// Returns an error when `fractional_delay` is outside `[-1, 1]`. The
    /// filter is unchanged on error.
    pub fn set_delay(&mut self, fractional_delay: f32) -> Result<()> {
        if fractional_delay < -1.0 || fractional_delay > 1.0 {
            return Err(Error::Config("delay must be in [-1,1]".into()));
        }

        // evaluate the taps, then normalize by their own sum so the DC response
        // is unity at this delay.
        let mut sum = 0.0f32;
        for i in 0..self.h_len {
            let n = i * (self.q + 1);
            // the polynomials are fit against fractional_delay, but evaluated at -fractional_delay: a
            // positive delay shifts the sinc the other way.
            let v = poly_val(&self.p[n..n + self.q + 1], -fractional_delay);
            self.htmp[i] = v;
            sum += v;
        }

        // normalize with fallback
        let scale = if sum.abs() > f32::EPSILON { 1.0 / sum } else { self.gamma };
        for i in 0..self.h_len {
            self.h[i] = <Coeff as From<f32>>::from(self.htmp[i] * scale);
        }

        // store coeffs in reversed order for dotprod
        for (dst, &src) in self.h_rev.iter_mut().zip(self.h.iter().rev()) {
            *dst = src;
        }

        Ok(())
    }

    /// Adds one sample to the history without computing an output.
    pub fn push(&mut self, input: T) {
        self.w.push(input);
    }

    /// Adds a block of samples to the history without computing outputs.
    pub fn write(&mut self, input: &[T]) {
        self.w.write(input);
    }

    /// Returns the output for the current sample history.
    ///
    /// The history is not changed, so repeated calls return the same value.
    pub fn execute(&self) -> T {
        self.w.read().dotprod(&self.h_rev)
    }

    /// Filters a block of samples at the current delay.
    ///
    /// Each input sample is added to the history and followed by one output.
    /// Extra output is left unchanged. Returns the number of outputs written,
    /// which is `input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`. The filter and
    /// `output` are unchanged on error.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Config("output block length must be at least the input length".into()));
        }
        for (&xi, yi) in input.iter().zip(output.iter_mut()) {
            self.push(xi);
            *yi = self.execute();
        }
        Ok(input.len())
    }

    /// Evaluates the frequency response at `fc` cycles per sample.
    ///
    /// The response is for the current delay. `fc` is not range-checked.
    pub fn freqresponse(&self, fc: f32) -> Complex32 {
        let h: Vec<f32> = self.h.iter().map(|&h_i| h_i.re()).collect();
        freqresponse(&h, fc).unwrap()
    }

    /// Estimates the group delay in samples at `fc` cycles per sample.
    ///
    /// The estimate comes from the current taps. It includes the
    /// `(filter_length - 1) / 2` center delay and the fractional delay the taps
    /// actually give, which can be less than the delay set with
    /// [`Self::set_delay`] when the cutoff is low.
    ///
    /// # Errors
    ///
    /// Returns an error when `fc` is outside `[-0.5, 0.5]`.
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        let h: Vec<f32> = self.h.iter().map(|&h_i| h_i.re()).collect();
        fir_group_delay(&h, fc)
    }

    /// generate the polynomials representing each filter tap
    fn genpoly(&mut self) -> Result<()> {
        // Design in f64 because the Vandermonde fit can be ill-conditioned in
        // f32. The resulting error is static in the stored f32 coefficients.
        let beta = kaiser_beta_stopband_attenuation(self.as_);
        let mu_vect: Vec<_> = (0..=self.q).map(|j| 2.0 * j as f64 / self.q as f64 - 1.0).collect();
        let mut workspace = vandermonde_workspace(&mu_vect, self.q + 1)?;
        let mut hp_vect = vec![0.0f64; self.q + 1];
        let mut p = vec![0.0f64; self.q + 1];

        for i in 0..self.h_len {
            let x = i as f64 - (self.h_len - 1) as f64 / 2.0;
            let h1 = kaiser(i, self.h_len, beta)? as f64;
            for (&mu, hp) in mu_vect.iter().zip(&mut hp_vect) {
                // here we diverge from liquid. we use the full [-1.0, 1.0] range so that
                // set_delay can use the full range
                let h0 = sincd(2.0 * self.fc as f64 * (x + mu));
                *hp = h0 * h1;
            }

            // fit a polynomial in mu through those samples
            workspace.solve(&hp_vect, &mut p)?;

            let n = i * (self.q + 1);
            for (dst, &src) in self.p[n..n + self.q + 1].iter_mut().zip(p.iter()) {
                *dst = src as f32;
            }
        }

        // normalize DC gain
        let mut sum = 0.0f32;
        for i in 0..self.h_len {
            let n = i * (self.q + 1);
            sum += poly_val(&self.p[n..n + self.q + 1], 0.0);
        }
        self.gamma = if sum.abs() > f32::EPSILON { 1.0 / sum } else { 1.0 };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::test_helpers::validate_block_execute;
    use approx::assert_abs_diff_eq;

    /// estimate the group delay of an impulse response from the phase at a
    /// single low frequency, as the fdelay testbench does
    fn measure_delay(y: &[f32]) -> f32 {
        let fc = 0.1 / y.len() as f32;
        let mut v = Complex32::new(0.0, 0.0);
        for (i, &yi) in y.iter().enumerate() {
            v += yi * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * fc * i as f32);
        }
        v.arg() / (2.0 * std::f32::consts::PI * fc)
    }

    fn impulse_response(h_len: usize, p: usize, fc: f32, as_: f32, mu: f32) -> Vec<f32> {
        let mut q = FirFarrowFilter::<f32, f32>::new(h_len, p, fc, as_).unwrap();
        q.set_delay(mu).unwrap();
        let n = h_len + 8;
        let mut x = vec![0.0f32; n];
        x[0] = 1.0;
        let mut y = vec![0.0f32; n];
        q.execute_block(&x, &mut y).unwrap();
        y
    }

    #[test]
    fn test_firfarrow_delay_is_continuous() {
        let h_len = 19;
        let nominal = (h_len - 1) as f32 / 2.0;
        for k in -63..=63 {
            let mu = k as f32 / 63.0;
            let y = impulse_response(h_len, 5, 0.45, 60.0, mu);
            assert_abs_diff_eq!(measure_delay(&y), nominal + mu, epsilon = 0.01);
        }
    }

    #[test]
    fn test_firfarrow_reported_delay_matches_realized() {
        let h_len = 19;
        let mut q = FirFarrowFilter::<f32, f32>::new(h_len, 5, 0.45, 60.0).unwrap();
        for &mu in &[-0.5f32, -0.25, 0.0, 0.25, 0.5] {
            q.set_delay(mu).unwrap();
            let reported = q.groupdelay(0.0).unwrap();
            let realized = measure_delay(&impulse_response(h_len, 5, 0.45, 60.0, mu));
            assert_abs_diff_eq!(reported, realized, epsilon = 0.01);
        }
    }

    #[test]
    fn test_firfarrow_groupdelay() {
        let h_len = 19;
        let nominal = (h_len - 1) as f32 / 2.0;
        let mut q = FirFarrowFilter::<f32, f32>::new(h_len, 5, 0.45, 60.0).unwrap();
        for &mu in &[-0.5f32, 0.0, 0.25, 0.5] {
            q.set_delay(mu).unwrap();
            assert_abs_diff_eq!(q.groupdelay(0.0).unwrap(), nominal + mu, epsilon = 0.01);
            assert_abs_diff_eq!(q.groupdelay(0.2).unwrap(), nominal + mu, epsilon = 0.05);
        }
    }

    #[test]
    fn test_firfarrow_unit_dc_response() {
        for &(h_len, p, fc, as_) in &[(19usize, 5usize, 0.45f32, 60.0f32), (8, 3, 0.25, 40.0), (33, 7, 0.45, 80.0)] {
            let mut q = FirFarrowFilter::<f32, f32>::new(h_len, p, fc, as_).unwrap();
            for k in -16..=16 {
                let mu = k as f32 / 16.0;
                q.set_delay(mu).unwrap();
                let sum: f32 = q.coefficients().iter().sum();
                assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_firfarrow_phase_ramp() {
        let h_len = 19;
        let nominal = (h_len - 1) as f32 / 2.0;
        let mut q = FirFarrowFilter::<f32, f32>::new(h_len, 5, 0.45, 60.0).unwrap();
        for &mu in &[-0.5f32, -0.2, 0.0, 0.2, 0.5] {
            q.set_delay(mu).unwrap();
            let h = q.coefficients();
            for &f in &[0.05f32, 0.1, 0.2] {
                let mut acc = Complex32::new(0.0, 0.0);
                for (i, &h_i) in h.iter().enumerate() {
                    acc +=
                        h_i * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * f * (i as f32 - nominal - mu));
                }
                // with the expected ramp removed the response is real and
                // positive across the passband
                assert!(acc.re > 0.0, "mu {mu} f {f} re {}", acc.re);
                assert_abs_diff_eq!(acc.arg(), 0.0, epsilon = 0.02);
            }
        }
    }

    #[test]
    fn test_firfarrow_complex() {
        let h_len = 19;
        let nominal = (h_len - 1) as f32 / 2.0;
        let mu = 0.3f32;
        let mut q = FirFarrowFilter::<Complex32, f32>::new(h_len, 5, 0.45, 60.0).unwrap();
        q.set_delay(mu).unwrap();

        let n = h_len + 8;
        let mut x = vec![Complex32::new(0.0, 0.0); n];
        x[0] = Complex32::new(1.0, -1.0);
        let mut y = vec![Complex32::new(0.0, 0.0); n];
        q.execute_block(&x, &mut y).unwrap();

        let re: Vec<f32> = y.iter().map(|v| v.re).collect();
        let im: Vec<f32> = y.iter().map(|v| -v.im).collect();
        assert_abs_diff_eq!(measure_delay(&re), nominal + mu, epsilon = 0.01);
        assert_abs_diff_eq!(measure_delay(&im), nominal + mu, epsilon = 0.01);
    }

    #[test]
    fn test_firfarrow_config() {
        // invalid constructions
        assert!(FirFarrowFilter::<f32, f32>::new(1, 5, 0.45, 60.0).is_err());
        assert!(FirFarrowFilter::<f32, f32>::new(19, 0, 0.45, 60.0).is_err());
        assert!(FirFarrowFilter::<f32, f32>::new(19, 5, -0.1, 60.0).is_err());
        assert!(FirFarrowFilter::<f32, f32>::new(19, 5, 0.6, 60.0).is_err());
        assert!(FirFarrowFilter::<f32, f32>::new(19, 5, 0.45, -1.0).is_err());

        let mut q = FirFarrowFilter::<f32, f32>::new(19, 5, 0.45, 60.0).unwrap();
        assert_eq!(q.length(), 19);
        assert_eq!(q.order(), 5);
        assert_eq!(q.coefficients().len(), 19);

        // delay outside [-1,1]
        assert!(q.set_delay(1.01).is_err());
        assert!(q.set_delay(-1.01).is_err());
        assert!(q.set_delay(1.0).is_ok());
        assert!(q.set_delay(-1.0).is_ok());
    }

    #[test]
    fn test_firfarrow_execute_block_contract() {
        let q = FirFarrowFilter::<f32, f32>::new(9, 3, 0.45, 60.0).unwrap();
        validate_block_execute(&q, &[1.0, -0.5, 0.25], -9.0, FirFarrowFilter::execute_block);
    }
}
