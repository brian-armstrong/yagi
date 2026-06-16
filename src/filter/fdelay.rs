use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter::FirPfbFilter;
use crate::buffer::Window;

use num_complex::ComplexFloat;

#[derive(Clone, Debug)]
pub struct Fdelay<T, Coeff = T> {
    nmax: usize,
    m: usize,
    npfb: usize,
    delay: f32,
    w: Window<T>,
    pfb: FirPfbFilter<T, Coeff>,
    w_index: usize,
    f_index: usize,
}

impl<T, Coeff> Fdelay<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Copy + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + Default,
    [Coeff]: DotProd<T, Output = T>,
{
    pub fn new(nmax: usize, m: usize, npfb: usize) -> Result<Self> {
        if nmax == 0 {
            return Err(Error::Config("maximum delay must be greater than zero".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter semi-length must be greater than zero".into()));
        }
        if npfb == 0 {
            return Err(Error::Config("number of filters must be greater than zero".into()));
        }

        let w = Window::new(nmax + 1)?;
        let pfb = FirPfbFilter::default(npfb, m)?;

        let mut q = Self {
            nmax,
            m,
            npfb,
            delay: 0.0,
            w,
            pfb,
            w_index: nmax - 1,
            f_index: 0,
        };

        q.reset();
        Ok(q)
    }

    pub fn new_default(nmax: usize) -> Result<Self> {
        Self::new(nmax, 8, 64)
    }

    pub fn reset(&mut self) {
        self.delay = 0.0;
        self.w_index = self.nmax - 1;
        self.f_index = 0;
        self.w.reset();
        self.pfb.reset();
    }

    pub fn get_delay(&self) -> f32 {
        self.delay
    }

    pub fn set_delay(&mut self, delay: f32) -> Result<()> {
        if delay < 0.0 {
            return Err(Error::Config("delay cannot be negative".into()));
        }
        if delay > self.nmax as f32 {
            return Err(Error::Config(format!("delay ({}) cannot exceed maximum ({})",
                                             delay, self.nmax)));
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

    pub fn adjust_delay(&mut self, delta: f32) -> Result<()> {
        self.set_delay(self.delay + delta)
    }

    pub fn get_nmax(&self) -> usize {
        self.nmax
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn get_npfb(&self) -> usize {
        self.npfb
    }

    pub fn push(&mut self, x: T) -> () {
        self.w.push(x);
        self.pfb.push(self.w.index(self.w_index).unwrap());
    }

    pub fn write(&mut self, x: &[T]) -> () {
        for &xi in x {
            self.push(xi);
        }
    }

    pub fn execute(&mut self) -> Result<T> {
        self.pfb.execute(self.f_index)
    }

    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        for (&xi, yi) in x.iter().zip(y.iter_mut()) {
            self.push(xi);
            *yi = self.execute()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use num_complex::Complex32;
    use approx::assert_abs_diff_eq;

    fn testbench_fdelay_rrrf(nmax: usize, m: usize, npfb: usize, delay: f32) {
        let tol = 0.01f32;
        let num_samples = nmax + 2 * m;  // number of samples to run

        // create delay object and split delay between set and adjust methods
        let mut q = Fdelay::<f32, f32>::new(nmax, m, npfb).unwrap();
        q.set_delay(delay * 0.7).unwrap();
        q.adjust_delay(delay * 0.3).unwrap();

        // ensure object is configured properly
        assert_eq!(q.get_nmax(), nmax);
        assert_eq!(q.get_m(), m);
        assert_eq!(q.get_npfb(), npfb);
        assert_abs_diff_eq!(q.get_delay(), delay, epsilon = 1e-6);

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
    fn test_fdelay_rrrf_0() { testbench_fdelay_rrrf(200, 12, 64,   0.0    ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_1)]
    fn test_fdelay_rrrf_1() { testbench_fdelay_rrrf(200, 12, 64,   0.0001 ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_2)]
    fn test_fdelay_rrrf_2() { testbench_fdelay_rrrf(200, 12, 64,   0.1    ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_3)]
    fn test_fdelay_rrrf_3() { testbench_fdelay_rrrf(200, 12, 64,   0.9    ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_4)]
    fn test_fdelay_rrrf_4() { testbench_fdelay_rrrf(200, 12, 64,   0.9999 ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_5)]
    fn test_fdelay_rrrf_5() { testbench_fdelay_rrrf(200, 12, 64,  16.99   ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_6)]
    fn test_fdelay_rrrf_6() { testbench_fdelay_rrrf(200, 12, 64,  17.00   ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_7)]
    fn test_fdelay_rrrf_7() { testbench_fdelay_rrrf(200, 12, 64,  17.01   ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_8)]
    fn test_fdelay_rrrf_8() { testbench_fdelay_rrrf(200, 12, 64, 199.9    ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_9)]
    fn test_fdelay_rrrf_9() { testbench_fdelay_rrrf(200, 12, 64, 200.0    ); }

    #[test]
    #[autotest_annotate(autotest_fdelay_rrrf_config)]
    fn test_fdelay_rrrf_config() {
        // default configurations
        let nmax: usize = 200;
        let m: usize = 12;
        let npfb: usize = 64;

        // test invalid configurations, normal construction
        assert!(Fdelay::<f32, f32>::new(0, m, npfb).is_err());
        assert!(Fdelay::<f32, f32>::new(nmax, 0, npfb).is_err());
        assert!(Fdelay::<f32, f32>::new(nmax, m, 0).is_err());

        // test invalid configurations, default construction
        assert!(Fdelay::<f32, f32>::new_default(0).is_err());

        // create proper object but test invalid internal configurations
        let mut q = Fdelay::<f32, f32>::new_default(nmax).unwrap();

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
        let mut q0 = Fdelay::<f32, f32>::new_default(200).unwrap();
        let mut q1 = Fdelay::<f32, f32>::new_default(200).unwrap();

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
}