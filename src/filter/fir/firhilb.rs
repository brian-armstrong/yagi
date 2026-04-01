use crate::error::{Error, Result};
use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::filter;
use num_traits::Zero;
use std::f32::consts::PI;

use num_complex::Complex32;

/// Finite impulse response (FIR) Hilbert transform
/// 
/// 2:1 real-to-complex decimator
/// 
/// 1:2 complex-to-real interpolator
#[derive(Clone, Debug)]
pub struct FirHilbertFilter {
    m: usize,           // filter semi-length, h_len = 4*m+1
    hq: Vec<f32>,       // quadrature filter coefficients
    w0: Window<f32>,    // input buffer (even samples)
    w1: Window<f32>,    // input buffer (odd samples)
    w2: Window<f32>,    // additional buffers needed exclusively for real-to-complex filter operations
    w3: Window<f32>,    // additional buffers needed exclusively for real-to-complex filter operations
    toggle: bool,       // toggle for real-to-complex/complex-to-real operation
}

impl FirHilbertFilter {
    /// Create a new FIR Hilbert transform object with a particular filter
    /// semi-length and desired stop-band attenuation.
    /// 
    /// # Arguments
    /// 
    /// * `m` - filter semi-length, delay is 2*m+1
    /// * `as_` - filter stop-band attenuation \[dB\]
    /// 
    /// # Returns
    ///
    /// A new FIR Hilbert transform object
    pub fn new(m: usize, as_: f32) -> Result<Self> {
        if m < 2 {
            return Err(Error::Config("filter semi-length (m) must be at least 2".into()));
        }

        let h_len = 4 * m + 1;
        let mut hc = vec![Complex32::zero(); h_len];
        let hq_len = 2 * m;
        let mut hq = vec![0.0; hq_len];
        let as_ = as_.abs();

        // compute filter coefficients for half-band filter
        let mut h = filter::fir_design_kaiser(h_len, 0.25, as_, 0.0)?;

        // alternate sign of non-zero elements
        for i in 0..h_len {
            let t = i as f32 - (h_len - 1) as f32 / 2.0;
            hc[i] = h[i] * Complex32::from_polar(1.0, 0.5 * PI * t);
            h[i] = hc[i].im;
        }

        // resample, reverse direction
        let mut j = 0;
        for i in (1..h_len).step_by(2) {
            hq[j] = h[h_len - i - 1].clone();
            j += 1;
        }

        // create windows for upper and lower polyphase filter branches
        let w0 = Window::new(2 * m)?;
        let w1 = Window::new(2 * m)?;
        let w2 = Window::new(2 * m)?;
        let w3 = Window::new(2 * m)?;

        let mut q = Self {
            m,
            hq,
            w0,
            w1,
            w2,
            w3,
            toggle: false,
        };

        q.reset();
        Ok(q)
    }

    /// Reset the internal state of the filter
    pub fn reset(&mut self) {
        self.w0.reset();
        self.w1.reset();
        self.w2.reset();
        self.w3.reset();
        self.toggle = false;
    }

    /// Execute the Hilbert transform (real-to-complex)
    /// 
    /// # Arguments
    /// 
    /// * `x` - real-valued input sample
    /// 
    /// # Returns
    /// 
    /// A complex-valued output sample
    pub fn r2c_execute(&mut self, x: f32) -> Result<Complex32> {
        let yi;  // in-phase component
        let yq;  // quadrature component

        if !self.toggle {
            // push sample into upper branch
            self.w0.push(x);

            // upper branch (delay)
            yi = self.w0.index(self.m - 1)?;

            // lower branch (filter)
            let r = self.w1.read();

            // execute dot product
            yq = self.hq.dotprod(r);
        } else {
            // push sample into lower branch
            self.w1.push(x);

            // upper branch (delay)
            yi = self.w1.index(self.m - 1)?;

            // lower branch (filter)
            let r = self.w0.read();

            // execute dot product
            yq = self.hq.dotprod(r);
        }

        self.toggle = !self.toggle;

        Ok(Complex32::new(yi, yq))
    }

    /// Execute the Hilbert transform (complex-to-real)
    /// 
    /// # Arguments
    /// 
    /// * `x` - complex-valued input sample
    /// 
    /// # Returns
    /// 
    /// A tuple of two real-valued output samples
    ///    (lower side-band retained, upper side-band retained)
    pub fn c2r_execute(&mut self, x: Complex32) -> Result<(f32, f32)> {
        let yi;  // in-phase component
        let yq;

        if !self.toggle {
            // push samples into appropriate buffers
            self.w0.push(x.re);
            self.w1.push(x.im);

            // delay branch
            yi = self.w0.index(self.m - 1)?;

            // filter branch
            let r = self.w3.read();
            yq = self.hq.dotprod(r);
        } else {
            // push samples into appropriate buffers
            self.w2.push(x.re);
            self.w3.push(x.im);

            // delay branch
            yi = self.w2.index(self.m - 1)?;

            // filter branch
            let r = self.w1.read();
            yq = self.hq.dotprod(r);
        }

        self.toggle = !self.toggle;

        Ok((yi + yq, yi - yq))
    }

    /// Execute the Hilbert transform decimator (real-to-complex)
    /// 
    /// # Arguments
    /// 
    /// * `x` - real-valued input array, [size: 2 x 1]
    /// 
    /// # Returns
    /// 
    /// A complex-valued output sample
    pub fn decim_execute(&mut self, x: &[f32]) -> Result<Complex32> {
        let yi;  // in-phase component
        let yq;  // quadrature component

        // compute quadrature component (filter branch)
        self.w1.push(x[0]);
        let r = self.w1.read();
        yq = self.hq.dotprod(r);

        // delay branch
        self.w0.push(x[1]);
        yi = self.w0.index(self.m - 1)?;

        // set return value
        let v = Complex32::new(yi, yq);
        let y = if self.toggle { -v } else { v };

        // toggle flag
        self.toggle = !self.toggle;
        Ok(y)
    }

    /// Execute the Hilbert transform decimator (real-to-complex) on a block of samples
    /// 
    /// # Arguments
    /// 
    /// * `x` - real-valued input array, [size: 2*n x 1]
    /// * `n` - number of output samples
    /// * `y` - complex-valued output array, [size: n x 1]
    pub fn decim_execute_block(&mut self, x: &[f32], n: usize, y: &mut [Complex32]) -> Result<()> {
        for i in 0..n {
            y[i] = self.decim_execute(&x[2*i..2*i+2])?;
        }
        Ok(())
    }

    /// Execute the Hilbert transform interpolator (complex-to-real)
    /// 
    /// # Arguments
    /// 
    /// * `x` - complex-valued input sample
    /// * `y` - real-valued output array, [size: 2 x 1]
    pub fn interp_execute(&mut self, x: Complex32, y: &mut [f32]) -> Result<()> {
        let vi = if self.toggle { -x.re } else { x.re };
        let vq = if self.toggle { -x.im } else { x.im };

        // compute delay branch
        self.w0.push(vq.into());
        y[0] = self.w0.index(self.m - 1)?;

        // compute filter branch
        self.w1.push(vi.into());
        let r = self.w1.read();
        y[1] = self.hq.dotprod(r);

        self.toggle = !self.toggle;
        Ok(())
    }

    /// Execute the Hilbert transform interpolator (complex-to-real) on a block of samples
    /// 
    /// # Arguments
    /// 
    /// * `x` - complex-valued input array, [size: n x 1]
    /// * `n` - number of output samples
    /// * `y` - real-valued output array, [size: 2*n x 1]
    pub fn interp_execute_block(&mut self, x: &[Complex32], n: usize, y: &mut [f32]) -> Result<()> {
        for i in 0..n {
            self.interp_execute(x[i], &mut y[2*i..2*i+2])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use crate::utility::test_helpers::{PsdRegion, validate_psd_signal, validate_psd_signalf};
    use approx::assert_relative_eq;

    #[test]
    #[autotest_annotate(autotest_firhilbf_decim)]
    fn test_firhilbf_decim() {
        let x: [f32; 32] = [
            1.0000,  0.7071,  0.0000, -0.7071, -1.0000, -0.7071, -0.0000,  0.7071,
            1.0000,  0.7071,  0.0000, -0.7071, -1.0000, -0.7071, -0.0000,  0.7071,
            1.0000,  0.7071,  0.0000, -0.7071, -1.0000, -0.7071, -0.0000,  0.7071,
            1.0000,  0.7071, -0.0000, -0.7071, -1.0000, -0.7071, -0.0000,  0.7071
        ];

        let test: [Complex32; 16] = [
            Complex32::new(0.0000, -0.0055), Complex32::new(-0.0000,  0.0231), Complex32::new(0.0000, -0.0605), Complex32::new(-0.0000,  0.1459),
            Complex32::new(0.0000, -0.5604), Complex32::new(-0.7071, -0.7669), Complex32::new(-0.7071,  0.7294), Complex32::new(0.7071,  0.7008),
            Complex32::new(0.7071, -0.7064), Complex32::new(-0.7071, -0.7064), Complex32::new(-0.7071,  0.7064), Complex32::new(0.7071,  0.7064),
            Complex32::new(0.7071, -0.7064), Complex32::new(-0.7071, -0.7064), Complex32::new(-0.7071,  0.7064), Complex32::new(0.7071,  0.7064)
        ];

        let mut y = [Complex32::new(0.0, 0.0); 16];
        let m = 5;   // h_len = 4*m+1 = 21
        let mut ht = FirHilbertFilter::new(m, 60.0).unwrap();
        let tol = 0.005;

        // run decimator
        for i in 0..16 {
            y[i] = ht.decim_execute(&x[2*i..2*i+2]).unwrap();
        }

        // run validation
        for i in 0..16 {
            assert_relative_eq!(y[i].re, test[i].re, epsilon = tol);
            assert_relative_eq!(y[i].im, test[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firhilbf_interp)]
    fn test_firhilbf_interp() {
        let x: [Complex32; 16] = [
            Complex32::new(1.0000, 0.0000), Complex32::new(-0.0000, -1.0000), Complex32::new(-1.0000, 0.0000), Complex32::new(0.0000, 1.0000),
            Complex32::new(1.0000, -0.0000), Complex32::new(-0.0000, -1.0000), Complex32::new(-1.0000, 0.0000), Complex32::new(0.0000, 1.0000),
            Complex32::new(1.0000, -0.0000), Complex32::new(-0.0000, -1.0000), Complex32::new(-1.0000, 0.0000), Complex32::new(0.0000, 1.0000),
            Complex32::new(1.0000, -0.0000), Complex32::new(0.0000, -1.0000), Complex32::new(-1.0000, 0.0000), Complex32::new(0.0000, 1.0000)
        ];

        let test: [f32; 32] = [
            0.0000, -0.0055, -0.0000, -0.0231, -0.0000, -0.0605, -0.0000, -0.1459,
            -0.0000, -0.5604, -0.0000, 0.7669, 1.0000, 0.7294, 0.0000, -0.7008,
            -1.0000, -0.7064, -0.0000, 0.7064, 1.0000, 0.7064, 0.0000, -0.7064,
            -1.0000, -0.7064, -0.0000, 0.7064, 1.0000, 0.7064, 0.0000, -0.7064
        ];

        let mut y = [0.0; 32];
        let m = 5;   // h_len = 4*m+1 = 21
        let mut ht = FirHilbertFilter::new(m, 60.0).unwrap();
        let tol = 0.005;

        // run interpolator
        for i in 0..16 {
            ht.interp_execute(x[i], &mut y[2*i..2*i+2]).unwrap();
        }

        // run validation
        for i in 0..32 {
            assert_relative_eq!(y[i], test[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firhilbf_psd)]
    fn test_firhilbf_psd() {
        let tol: f32 = 1.0;  // error tolerance [dB]
        let bw: f32 = 0.4;   // pulse bandwidth
        let as_: f32 = 60.0; // transform stop-band suppression
        let p: usize = 40;   // pulse semi-length
        let m: usize = 25;   // Transform delay

        // create transform
        let mut q = FirHilbertFilter::new(m, as_).unwrap();

        let h_len: usize = 2 * p + 1; // pulse length
        let num_samples: usize = h_len + 2 * m + 8;

        let mut buf_0 = vec![Complex32::new(0.0, 0.0); num_samples];
        let mut buf_1 = vec![0.0f32; num_samples * 2];
        let mut buf_2 = vec![Complex32::new(0.0, 0.0); num_samples];

        // generate the baseband signal (filter pulse)
        let w: f32 = 0.36 * bw; // pulse bandwidth
        let h = filter::fir_design_kaiser(h_len, w, 80.0, 0.0).unwrap();
        for i in 0..num_samples {
            buf_0[i] = Complex32::new(if i < h_len { 2.0 * w * h[i] } else { 0.0 }, 0.0);
        }

        // run interpolation
        q.interp_execute_block(&buf_0, num_samples, &mut buf_1).unwrap();

        // clear object
        q.reset();

        // run decimation
        q.decim_execute_block(&buf_1, num_samples, &mut buf_2).unwrap();

        // verify input spectrum
        let regions_orig = vec![
            PsdRegion { fmin: -0.5,    fmax: -0.5*bw, pmin: 0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.3*bw, fmax: 0.3*bw,  pmin: -1.0, pmax: 1.0,     test_lo: true,  test_hi: true },
            PsdRegion { fmin: 0.5*bw,  fmax: 0.5,     pmin: 0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signal(&buf_0, &regions_orig).unwrap());

        // verify interpolated spectrum
        let regions_interp = vec![
            PsdRegion { fmin: -0.5,           fmax: -0.25-0.25*bw, pmin: 0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.25-0.15*bw,  fmax: -0.25+0.15*bw, pmin: -1.0, pmax: 1.0,     test_lo: true,  test_hi: true },
            PsdRegion { fmin: -0.25+0.25*bw,  fmax: 0.25-0.25*bw,  pmin: 0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: 0.25-0.15*bw,   fmax: 0.25+0.15*bw,  pmin: -1.0, pmax: 1.0,     test_lo: true,  test_hi: true },
            PsdRegion { fmin: 0.25+0.25*bw,   fmax: 0.5,           pmin: 0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signalf(&buf_1, &regions_interp).unwrap());

        // verify decimated spectrum (using same regions as original)
        assert!(validate_psd_signal(&buf_2, &regions_orig).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_firhilbf_invalid_config)]
    fn test_firhilb_invalid_config() {
        // check that object returns None for invalid configurations
        assert!(FirHilbertFilter::new(0, 60.0).is_err()); // m too small
        assert!(FirHilbertFilter::new(1, 60.0).is_err()); // m too small

        // create proper object but test invalid internal configurations
        // let q = FirHilb::create(12, 60.0).unwrap();
        // q is automatically dropped at the end of scope
    }

    #[test]
    #[autotest_annotate(autotest_firhilbf_copy_interp)]
    fn test_firhilb_copy_interp() {
        let mut q0 = FirHilbertFilter::new(12, 120.0).unwrap();

        // run interpolator on random data
        let mut y0 = [0.0; 2];
        let mut y1 = [0.0; 2];
        for _ in 0..80 {
            let x = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.interp_execute(x, &mut y0).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        for _ in 0..80 {
            let x = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.interp_execute(x, &mut y0).unwrap();
            q1.interp_execute(x, &mut y1).unwrap();
            assert_relative_eq!(y0[0], y1[0]);
            assert_relative_eq!(y0[1], y1[1]);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firhilbf_copy_decim)]
    fn test_firhilb_copy_decim() {
        let mut q0 = FirHilbertFilter::new(12, 120.0).unwrap();

        // run decimator on random data
        let mut x = [0.0; 2];
        let mut y0;
        let mut y1;
        for _ in 0..80 {
            x[0] = crate::random::randnf();
            x[1] = crate::random::randnf();
            q0.decim_execute(&x).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        for _ in 0..80 {
            x[0] = crate::random::randnf();
            x[1] = crate::random::randnf();
            y0 = q0.decim_execute(&x).unwrap();
            y1 = q1.decim_execute(&x).unwrap();
            assert_relative_eq!(y0.re, y1.re);
            assert_relative_eq!(y0.im, y1.im);
        }
    }
}