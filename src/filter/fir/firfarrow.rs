//
// firfarrow.rs
//
// Finite impulse response Farrow filter
//

use num_complex::{Complex32, ComplexFloat};

use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::filter::freqresponse;
use crate::filter::fir_group_delay;
use crate::filter::kaiser_beta_stopband_attenuation;
use crate::math::poly::{poly_fit, poly_val};
use crate::math::sincd;
use crate::math::windows::kaiser;

/// Finite impulse response (FIR) Farrow filter for timing delay
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
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Copy + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + Default,
    [T]: DotProd<Coeff, Output = T>,
{
    /// create a Farrow filter object
    ///
    /// # Arguments
    ///
    /// * `h_len` - filter length, `h_len >= 2`
    /// * `p` - polynomial order, `p >= 1`
    /// * `fc` - filter cutoff frequency, `0 <= fc <= 0.5`
    /// * `as_` - stop-band attenuation [dB], `as_ > 0`
    pub fn new(h_len: usize, p: usize, fc: f32, as_: f32) -> Result<Self> {
        if h_len < 2 {
            return Err(Error::Config("filter length must be greater than 1".into()));
        }
        if p < 1 {
            return Err(Error::Config("polynomial order must be at least 1".into()));
        }
        if fc < 0.0 || fc > 0.5 {
            return Err(Error::Config("filter cutoff must be in [0,0.5]".into()));
        }
        if as_ < 0.0 {
            return Err(Error::Config("filter stop-band attenuation must be greater than zero".into()));
        }

        let mut q = Self {
            h: vec![<Coeff as From<f32>>::from(0.0); h_len],
            h_rev: vec![<Coeff as From<f32>>::from(0.0); h_len],
            htmp: vec![0.0; h_len],
            h_len,
            fc,
            as_,
            q: p,
            p: vec![0.0; h_len * (p + 1)],
            gamma: 1.0,
            w: Window::new(h_len)?,
        };

        q.reset();
        q.genpoly()?;
        q.set_delay(0.0)?;

        Ok(q)
    }

    /// reset the filter's internal state
    pub fn reset(&mut self) {
        self.w.reset();
    }

    /// get the filter length (number of taps)
    pub fn get_length(&self) -> usize {
        self.h_len
    }

    /// get the polynomial order
    pub fn get_order(&self) -> usize {
        self.q
    }

    /// get the coefficients for the current delay
    pub fn get_coefficients(&self) -> &[Coeff] {
        &self.h
    }

    /// set the fractional delay of the filter
    ///
    /// # Arguments
    ///
    /// * `mu` - fractional sample delay, `-1 <= mu <= 1`
    pub fn set_delay(&mut self, mu: f32) -> Result<()> {
        if mu < -1.0 || mu > 1.0 {
            return Err(Error::Config("delay must be in [-1,1]".into()));
        }

        // evaluate the taps, then normalize by their own sum so the DC response
        // is unity at this delay.
        let mut sum = 0.0f32;
        for i in 0..self.h_len {
            let n = i * (self.q + 1);
            // the polynomials are fit against mu, but evaluated at -mu: a
            // positive delay shifts the sinc the other way.
            let v = poly_val(&self.p[n..n + self.q + 1], self.q + 1, -mu);
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

    /// push a sample into the filter's internal buffer
    pub fn push(&mut self, x: T) {
        self.w.push(x);
    }

    /// write a block of samples into the filter's internal buffer
    pub fn write(&mut self, x: &[T]) {
        self.w.write(x);
    }

    /// execute the dot product on the filter's internal buffer
    pub fn execute(&self) -> T {
        self.w.read().dotprod(&self.h_rev)
    }

    /// execute the filter on a block of samples
    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) {
        for (&xi, yi) in x.iter().zip(y.iter_mut()) {
            self.push(xi);
            *yi = self.execute();
        }
    }

    /// compute the complex frequency response at a given frequency
    ///
    /// # Arguments
    ///
    /// * `fc` - normalized frequency for evaluation
    pub fn freqresponse(&self, fc: f32) -> Complex32 {
        let h: Vec<f32> = self.h.iter().map(|&h_i| h_i.re()).collect();
        freqresponse(&h, fc).unwrap()
    }

    /// compute the group delay [samples] at a given frequency
    ///
    /// # Arguments
    ///
    /// * `fc` - normalized frequency for evaluation
    pub fn groupdelay(&self, fc: f32) -> Result<f32> {
        let h: Vec<f32> = self.h.iter().map(|&h_i| h_i.re()).collect();
        fir_group_delay(&h, fc)
    }

    /// generate the polynomials representing each filter tap
    fn genpoly(&mut self) -> Result<()> {
        // run this in f64. we are soling the normal equations here and are sensitive
        // to stability issues in f32. this only runs once at construction, so the
        // extra precision is worth it
        let beta = kaiser_beta_stopband_attenuation(self.as_);
        let mut mu_vect = vec![0.0f64; self.q + 1];
        let mut hp_vect = vec![0.0f64; self.q + 1];
        let mut p = vec![0.0f64; self.q + 1];

        for i in 0..self.h_len {
            let x = i as f64 - (self.h_len - 1) as f64 / 2.0;
            let h1 = kaiser(i, self.h_len, beta)? as f64;
            for j in 0..=self.q {
                // here we diverge from liquid. we use the full [-1.0, 1.0] range so that
                // set_delay can use the full range
                let mu = 2.0 * j as f64 / self.q as f64 - 1.0;
                let h0 = sincd(2.0 * self.fc as f64 * (x + mu));

                mu_vect[j] = mu;
                hp_vect[j] = h0 * h1;
            }

            // fit a polynomial in mu through those samples
            poly_fit(&mu_vect, &hp_vect, self.q + 1, &mut p, self.q + 1)?;

            let n = i * (self.q + 1);
            for (dst, &src) in self.p[n..n + self.q + 1].iter_mut().zip(p.iter()) {
                *dst = src as f32;
            }
        }

        // normalize DC gain
        let mut sum = 0.0f32;
        for i in 0..self.h_len {
            let n = i * (self.q + 1);
            sum += poly_val(&self.p[n..n + self.q + 1], self.q + 1, 0.0);
        }
        self.gamma = if sum.abs() > f32::EPSILON { 1.0 / sum } else { 1.0 };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        q.execute_block(&x, &mut y);
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
                let sum: f32 = q.get_coefficients().iter().sum();
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
            let h = q.get_coefficients();
            for &f in &[0.05f32, 0.1, 0.2] {
                let mut acc = Complex32::new(0.0, 0.0);
                for (i, &h_i) in h.iter().enumerate() {
                    acc += h_i * Complex32::from_polar(1.0, -2.0 * std::f32::consts::PI * f * (i as f32 - nominal - mu));
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
        q.execute_block(&x, &mut y);

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
        assert_eq!(q.get_length(), 19);
        assert_eq!(q.get_order(), 5);
        assert_eq!(q.get_coefficients().len(), 19);

        // delay outside [-1,1]
        assert!(q.set_delay(1.01).is_err());
        assert!(q.set_delay(-1.01).is_err());
        assert!(q.set_delay(1.0).is_ok());
        assert!(q.set_delay(-1.0).is_ok());
    }
}
