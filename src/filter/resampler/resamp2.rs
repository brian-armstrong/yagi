use crate::error::{Error, Result};
use crate::dotprod::{DotProd, DotProduct};
use crate::filter;
use crate::buffer::Window;
use std::f32::consts::PI;

use num_complex::{ComplexFloat, Complex32};

pub trait Resamp2Coeff {
    fn for_halfband(halfband: f32, t: f32, f0: f32) -> Self;
}

impl Resamp2Coeff for f32 {
    fn for_halfband(halfband: f32, t: f32, f0: f32) -> Self {
        2.0 * halfband * (2.0 * PI * t * f0).cos()
    }
}

impl Resamp2Coeff for Complex32 {
    fn for_halfband(halfband: f32, t: f32, f0: f32) -> Self {
        2.0 * halfband * Complex32::new((2.0 * PI * t * f0).cos(), (2.0 * PI * t * f0).sin())
    }
}

#[derive(Clone, Debug)]
pub struct Resamp2<T, Coeff = T> {
    m: usize,

    dp: DotProduct<T, Coeff>,

    w0: Window<T>,
    w1: Window<T>,
    scale: Coeff,

    toggle: bool,
}

impl<T, Coeff> Resamp2<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32> + Resamp2Coeff,
    T: Clone + Copy + ComplexFloat<Real = f32> + Default + From<f32> + std::ops::Mul<Coeff, Output = T>,
    [T]: DotProd<Coeff, Output = T>,
{
    pub fn new(m: usize, f0: f32, as_: f32) -> Result<Self> {
        if m < 2 {
            return Err(Error::Config("filter semi-length must be at least 2".into()));
        }
        if f0 < -0.5 || f0 > 0.5 {
            return Err(Error::Config(format!("f0 ({}) must be in [-0.5,0.5]", f0)));
        }
        if as_ < 0.0 {
            return Err(Error::Config(format!("as ({}) must be greater than zero", as_)));
        }

        let h_len = 4 * m + 1;
        let mut h = vec![Coeff::zero(); h_len];
        let hf = filter::fir_design_pm_halfband_stopband_attenuation(m, as_)?;

        for (i, hi) in h.iter_mut().enumerate() {
            let t = i as f32 - (h_len - 1) as f32 / 2.0;
            *hi = Coeff::for_halfband(hf[i], t, f0);
        }

        let h1_len = 2 * m;
        let mut h1 = vec![Coeff::zero(); h1_len];
        for (i, h1i) in h1.iter_mut().enumerate() {
            *h1i = h[h_len - 2*i - 2];
        }

        let w0 = Window::new(2 * m)?;
        let w1 = Window::new(2 * m)?;

        let mut q = Self {
            m,
            dp: DotProduct::new(&h1)?,
            w0,
            w1,
            scale: Coeff::one(),
            toggle: false,
        };

        q.reset();
        Ok(q)
    }

    pub fn reset(&mut self) {
        self.w0.reset();
        self.w1.reset();
        self.toggle = false;
    }

    pub fn set_scale(&mut self, scale: Coeff) {
        self.scale = scale;
    }

    pub fn get_scale(&self) -> Coeff {
        self.scale
    }

    pub fn get_delay(&self) -> usize {
        2 * self.m - 1
    }

    pub fn filter_execute(&mut self, x: T) -> Result<(T, T)> {
        let (yi, yq) = if !self.toggle {
            self.w0.push(x);
            let yi = self.w0.index(self.m - 1)?;
            let r = self.w1.read();
            let yq = self.dp.execute(r);
            (yi, yq)
        } else {
            self.w1.push(x);
            let yi = self.w1.index(self.m - 1)?;
            let r = self.w0.read();
            let yq = self.dp.execute(r);
            (yi, yq)
        };

        self.toggle = !self.toggle;

        let y0 = Into::<T>::into(0.5) * (yi + yq) * self.scale;
        let y1 = Into::<T>::into(0.5) * (yi - yq) * self.scale;
        Ok((y0, y1))
    }

    pub fn analyzer_execute(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        self.w1.push(Into::<T>::into(0.5) * x[0]);
        let r = self.w1.read();
        let y1 = self.dp.execute(r);

        self.w0.push(Into::<T>::into(0.5) * x[1]);
        let y0 = self.w0.index(self.m - 1)?.into();

        y[0] = (y1 + y0) * self.scale;
        y[1] = (y1 - y0) * self.scale;
        Ok(())
    }

    pub fn synthesizer_execute(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        let x0 = x[0] + x[1];
        let x1 = x[0] - x[1];

        self.w0.push(x0);
        y[0] = self.w0.index(self.m - 1)? * self.scale;

        self.w1.push(x1);
        let r = self.w1.read();
        y[1] = self.dp.execute(r) * self.scale;

        Ok(())
    }

    pub fn decim_execute(&mut self, x: &[T]) -> Result<T> {
        self.w1.push(x[0]);
        let r = self.w1.read();
        let y1 = self.dp.execute(r);

        self.w0.push(x[1]);
        let y0 = self.w0.index(self.m - 1)?;

        let y = (y0 + y1) * self.scale;
        Ok(y)
    }

    pub fn interp_execute(&mut self, x: T, y: &mut [T]) -> Result<()> {
        self.w0.push(x);
        y[0] = self.w0.index(self.m - 1)? * self.scale;

        self.w1.push(x);
        let r = self.w1.read();
        y[1] = self.dp.execute(r) * self.scale;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_abs_diff_eq;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_signal};
    use crate::random::randnf;

    use num_complex::Complex32;

    #[test]
    #[autotest_annotate(autotest_resamp2_analysis)]
    fn test_resamp2_analysis() {
        let m = 5;       // filter semi-length (actual length: 4*m+1)
        let n = 37;      // number of input samples
        let as_ = 60.0;  // stop-band attenuation [dB]
        let f0 = 0.0739; // low frequency signal
        let f1 = -0.1387; // high frequency signal (+pi)
        let tol = 1e-3;  // error tolerance

        // allocate memory for data arrays
        let mut x = vec![Complex32::new(0.0, 0.0); 2*n+2*m+1]; // input signal (with delay)
        let mut y0 = vec![Complex32::new(0.0, 0.0); n];        // low-pass output
        let mut y1 = vec![Complex32::new(0.0, 0.0); n];        // high-pass output

        // generate the baseband signal
        for i in 0..(2*n+2*m+1) {
            x[i] = if i < 2*n {
                Complex32::new(0.0, f0 * i as f32).exp() + Complex32::new(0.0, (std::f32::consts::PI + f1) * i as f32).exp()
            } else {
                Complex32::new(0.0, 0.0)
            };
        }

        // create the half-band resampler, with a specified stopband attenuation level
        let mut q = Resamp2::<Complex32>::new(m as usize, 0.0, as_).unwrap();

        // run half-band decimation
        let mut y_hat = [Complex32::new(0.0, 0.0); 2];
        for i in 0..n {
            q.analyzer_execute(&x[2*i..2*i+2], &mut y_hat).unwrap();
            y0[i] = y_hat[0];
            y1[i] = y_hat[1];
        }

        // validate output
        for i in m..(n-m) {
            assert_abs_diff_eq!(y0[i+m].re, (2.0 * f0 * (i as f32 + 0.5)).cos(), epsilon = tol);
            assert_abs_diff_eq!(y0[i+m].im, (2.0 * f0 * (i as f32 + 0.5)).sin(), epsilon = tol);

            assert_abs_diff_eq!(y1[i+m].re, (2.0 * f1 * (i as f32 + 0.5)).cos(), epsilon = tol);
            assert_abs_diff_eq!(y1[i+m].im, (2.0 * f1 * (i as f32 + 0.5)).sin(), epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_resamp2_synthesis)]
    fn test_resamp2_synthesis() {
        let m = 5;       // filter semi-length (actual length: 4*m+1)
        let n = 37;      // number of input samples
        let as_ = 60.0;  // stop-band attenuation [dB]
        let f0 = 0.0739; // low frequency signal
        let f1 = -0.1387; // high frequency signal (+pi)
        let tol = 3e-3;  // error tolerance

        // allocate memory for data arrays
        let mut x0 = vec![Complex32::new(0.0, 0.0); n+2*m+1]; // input signal (with delay)
        let mut x1 = vec![Complex32::new(0.0, 0.0); n+2*m+1]; // input signal (with delay)
        let mut y = vec![Complex32::new(0.0, 0.0); 2*n];      // synthesized output

        // generate the baseband signals
        for i in 0..(n+2*m+1) {
            x0[i] = if i < 2*n { Complex32::new(0.0, f0 * i as f32).exp() } else { Complex32::new(0.0, 0.0) };
            x1[i] = if i < 2*n { Complex32::new(0.0, f1 * i as f32).exp() } else { Complex32::new(0.0, 0.0) };
        }

        // create the half-band resampler, with a specified stopband attenuation level
        let mut q = Resamp2::<Complex32>::new(m as usize, 0.0, as_).unwrap();

        // run synthesis
        let mut x_hat = [Complex32::new(0.0, 0.0); 2];
        for i in 0..n {
            x_hat[0] = x0[i];
            x_hat[1] = x1[i];
            q.synthesizer_execute(&x_hat, &mut y[2*i..2*i+2]).unwrap();
        }

        // validate output
        for i in m..(n-2*m) {
            assert_abs_diff_eq!(y[i+2*m].re, (0.5 * f0 * i as f32).cos() + ((std::f32::consts::PI + 0.5 * f1) * i as f32).cos(), epsilon = tol);
            assert_abs_diff_eq!(y[i+2*m].im, (0.5 * f0 * i as f32).sin() + ((std::f32::consts::PI + 0.5 * f1) * i as f32).sin(), epsilon = tol);
        }
    }

    fn testbench_resamp2_crcf_filter(m: usize, as_: f32) {
        // error tolerance [dB]
        let tol = 0.5f32;

        // create the half-band resampler
        let mut q = Resamp2::<Complex32>::new(m, 0.0, as_).unwrap();

        // get impulse response
        let h_len = 4 * m + 1;
        let mut h_0 = vec![Complex32::new(0.0, 0.0); h_len];   // low-frequency response
        let mut h_1 = vec![Complex32::new(0.0, 0.0); h_len];   // high-frequency response
        
        for i in 0..h_len {
            let input = if i == 0 { Complex32::new(1.0, 0.0) } else { Complex32::new(0.0, 0.0) };
            let (y0, y1) = q.filter_execute(input).unwrap();
            h_0[i] = y0;
            h_1[i] = y1;
        }

        // compute expected transition band (extend slightly for relaxed constraints)
        let ft = filter::estimate_req_filter_transition_bandwidth(as_, h_len).unwrap() * 1.1;

        // verify low-pass frequency response
        let regions_h0 = vec![
            PsdRegion { fmin: -0.5,           fmax: -0.25 - ft/2.0, pmin: 0.0,  pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.25 + ft/2.0, fmax:  0.25 - ft/2.0, pmin: -1.0, pmax: 1.0,        test_lo: true,  test_hi: true },
            PsdRegion { fmin:  0.25 + ft/2.0, fmax:  0.5,           pmin: 0.0,  pmax: -as_ + tol, test_lo: false, test_hi: true },
        ];
        
        assert!(validate_psd_signal(&h_0, &regions_h0).unwrap());

        // verify high-pass frequency response
        let regions_h1 = vec![
            PsdRegion { fmin: -0.5,           fmax: -0.25 - ft/2.0, pmin: -1.0, pmax: 1.0,        test_lo: true,  test_hi: true },
            PsdRegion { fmin: -0.25 + ft/2.0, fmax:  0.25 - ft/2.0, pmin: 0.0,  pmax: -as_ + tol, test_lo: false, test_hi: true },
            PsdRegion { fmin:  0.25 + ft/2.0, fmax:  0.5,           pmin: -1.0, pmax: 1.0,        test_lo: true,  test_hi: true },
        ];
        
        assert!(validate_psd_signal(&h_1, &regions_h1).unwrap());
    }

    // test different configurations
    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_0)]
    fn test_resamp2_crcf_filter_0() { testbench_resamp2_crcf_filter(4, 60.0); }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_1)]
    fn test_resamp2_crcf_filter_1() { testbench_resamp2_crcf_filter(7, 60.0); }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_2)]
    fn test_resamp2_crcf_filter_2() { testbench_resamp2_crcf_filter(12, 60.0); }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_3)]
    fn test_resamp2_crcf_filter_3() { testbench_resamp2_crcf_filter(15, 80.0); }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_4)]
    fn test_resamp2_crcf_filter_4() { testbench_resamp2_crcf_filter(15, 100.0); }

    #[test]
    #[autotest_annotate(autotest_resamp2_crcf_filter_5)]
    fn test_resamp2_crcf_filter_5() { testbench_resamp2_crcf_filter(15, 120.0); }

    #[test]
    #[autotest_annotate(autotest_resamp2_config)]
    fn test_resamp2_config() {
        // check that object returns None for invalid configurations
        assert!(Resamp2::<Complex32, f32>::new(0, 0.0, 60.0).is_err()); // m out of range
        assert!(Resamp2::<Complex32, f32>::new(1, 0.0, 60.0).is_err()); // m out of range
        assert!(Resamp2::<Complex32, f32>::new(2, 0.7, 60.0).is_err()); // f0 out of range
        assert!(Resamp2::<Complex32, f32>::new(2, -0.7, 60.0).is_err()); // f0 out of range
        assert!(Resamp2::<Complex32, f32>::new(2, 0.0, -1.0).is_err()); // as out of range

        // create proper object and test configurations
        let q = Resamp2::<Complex32, f32>::new(4, 0.0, 60.0).unwrap();
        assert_eq!(q.get_delay(), 2 * 4 - 1);
        // q.print();

        // redesign filter with new length
        // nb there's no recreate
        // q = q.recreate(8, 0.0, 60.0);
        let q = Resamp2::<Complex32, f32>::new(8, 0.0, 60.0).unwrap();
        assert_eq!(q.get_delay(), 2 * 8 - 1);

        // redesign filter with same length, but new stop-band suppression
        // q = q.recreate(8, 0.0, 80.0);
        let mut q = Resamp2::<Complex32, f32>::new(8, 0.0, 80.0).unwrap();
        assert_eq!(q.get_delay(), 2 * 8 - 1);

        // test setting/getting properties
        q.set_scale(7.22);
        let scale = q.get_scale();
        assert_eq!(scale, 7.22);
    }

    // test copy method
    #[test]
    #[autotest_annotate(autotest_resamp2_copy)]
    fn test_resamp2_copy() {
        // create original half-band resampler
        let mut qa = Resamp2::<Complex32>::new(12, 0.0, 60.0).unwrap();

        // run random samples through filter
        let num_samples = 80;
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let _ = qa.filter_execute(v);
        }

        // copy object
        let mut qb = qa.clone();

        // run random samples through both filters and compare
        for _ in 0..num_samples {
            let v = Complex32::new(randnf(), randnf());
            let (ya0, ya1) = qa.filter_execute(v).unwrap();
            let (yb0, yb1) = qb.filter_execute(v).unwrap();

            assert_eq!(ya0, yb0);
            assert_eq!(ya1, yb1);
        }
    }
}