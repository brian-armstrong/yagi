use crate::error::{Error, Result};
use crate::filter::{IirBandType, IirFilter, IirFilterShape, IirFormat};

use num_complex::Complex32;

#[derive(Clone, Debug)]
pub struct IirHilbertFilter {
    filt_0: IirFilter<f32, f32>,
    filt_1: IirFilter<f32, f32>,
    state: u8,
}

impl IirHilbertFilter {
    pub fn new(
        filter_shape: IirFilterShape,
        order: usize,
        passband_ripple: f32,
        stopband_attenuation: f32,
    ) -> Result<Self> {
        if order == 0 {
            return Err(Error::Config("filter order must be greater than zero".into()));
        }

        let btype = IirBandType::Lowpass;
        let format = IirFormat::SecondOrderSections;
        let fc = 0.25;
        let f0 = 0.0;

        let filt_0 = IirFilter::new_prototype(
            filter_shape,
            btype,
            format,
            order,
            fc,
            f0,
            passband_ripple,
            stopband_attenuation,
        )?;
        let filt_1 = IirFilter::new_prototype(
            filter_shape,
            btype,
            format,
            order,
            fc,
            f0,
            passband_ripple,
            stopband_attenuation,
        )?;

        let mut q = Self { filt_0, filt_1, state: 0 };

        q.reset();
        Ok(q)
    }

    pub fn from_order(order: usize) -> Result<Self> {
        if order == 0 {
            return Err(Error::Config("filter order must be greater than zero".into()));
        }

        let ftype = IirFilterShape::Butter;
        let ap = 0.1;
        let as_ = 60.0;
        Self::new(ftype, order, ap, as_)
    }

    pub fn reset(&mut self) {
        self.filt_0.reset();
        self.filt_1.reset();
        self.state = 0;
    }

    pub fn r2c_execute(&mut self, input: f32) -> Complex32 {
        let y = match self.state {
            0 => {
                let yi = self.filt_0.execute(input);
                let yq = self.filt_1.execute(0.0);
                2.0 * Complex32::new(yi, yq)
            }
            1 => {
                let yi = self.filt_0.execute(0.0);
                let yq = self.filt_1.execute(-input);
                2.0 * Complex32::new(-yq, yi)
            }
            2 => {
                let yi = self.filt_0.execute(-input);
                let yq = self.filt_1.execute(0.0);
                2.0 * Complex32::new(-yi, -yq)
            }
            3 => {
                let yi = self.filt_0.execute(0.0);
                let yq = self.filt_1.execute(input);
                2.0 * Complex32::new(yq, -yi)
            }
            _ => unreachable!(),
        };

        self.state = (self.state + 1) & 0x3;
        y
    }

    pub fn r2c_execute_block(&mut self, input: &[f32], output: &mut [Complex32]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Config("output length must be at least the input length".into()));
        }
        for (&xi, yi) in input.iter().zip(output.iter_mut()) {
            *yi = self.r2c_execute(xi);
        }
        Ok(input.len())
    }

    pub fn c2r_execute(&mut self, input: Complex32) -> f32 {
        let y = match self.state {
            0 => {
                let (yi, _yq) = (self.filt_0.execute(input.re), self.filt_1.execute(input.im));
                yi
            }
            1 => {
                let (_yi, yq) = (self.filt_0.execute(input.im), self.filt_1.execute(-input.re));
                -yq
            }
            2 => {
                let (yi, _yq) = (self.filt_0.execute(-input.re), self.filt_1.execute(-input.im));
                -yi
            }
            3 => {
                let (_yi, yq) = (self.filt_0.execute(-input.im), self.filt_1.execute(input.re));
                yq
            }
            _ => unreachable!(),
        };

        self.state = (self.state + 1) & 0x3;
        y
    }

    pub fn c2r_execute_block(&mut self, input: &[Complex32], output: &mut [f32]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Config("output length must be at least the input length".into()));
        }
        for (&xi, yi) in input.iter().zip(output.iter_mut()) {
            *yi = self.c2r_execute(xi);
        }
        Ok(input.len())
    }

    pub fn decim_execute(&mut self, input: &[f32]) -> Complex32 {
        let xi = if self.state != 0 { -input[0] } else { input[0] };
        let xq = if self.state != 0 { input[1] } else { -input[1] };

        let yi0 = self.filt_0.execute(xi);
        let _yi1 = self.filt_0.execute(0.0);

        let yq0 = self.filt_1.execute(0.0);
        let _yq1 = self.filt_1.execute(xq);

        let y = 2.0 * Complex32::new(yi0, yq0);

        self.state = 1 - self.state;
        y
    }

    pub fn decim_execute_block(&mut self, input: &[f32], output: &mut [Complex32]) -> Result<usize> {
        if !input.len().is_multiple_of(2) {
            return Err(Error::Config("input length must be a multiple of 2".into()));
        }
        let n = input.len() / 2;
        if output.len() < n {
            return Err(Error::Config("output length must be at least input length / 2".into()));
        }
        for (x_chunk, yi) in input.as_chunks::<2>().0.iter().zip(output.iter_mut()) {
            *yi = self.decim_execute(x_chunk);
        }
        Ok(n)
    }

    pub fn interp_execute(&mut self, input: Complex32, output: &mut [f32]) {
        let yi0 = self.filt_0.execute(input.re);
        let _yi1 = self.filt_0.execute(0.0);

        let _yq0 = self.filt_1.execute(input.im);
        let yq1 = self.filt_1.execute(0.0);

        output[0] = if self.state != 0 { -2.0 * yi0 } else { 2.0 * yi0 };
        output[1] = if self.state != 0 { 2.0 * yq1 } else { -2.0 * yq1 };

        self.state = 1 - self.state;
    }

    pub fn interp_execute_block(&mut self, input: &[Complex32], output: &mut [f32]) -> Result<usize> {
        let n_out = input
            .len()
            .checked_mul(2)
            .ok_or_else(|| Error::Range("interpolation output length overflow".into()))?;
        if output.len() < n_out {
            return Err(Error::Config("output length must be at least 2 * input length".into()));
        }
        for (&xi, yi) in input.iter().zip(output.as_chunks_mut::<2>().0) {
            self.interp_execute(xi, yi);
        }
        Ok(n_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter;
    use crate::math::kaiser;
    use crate::utility::test_helpers::{validate_block_execute, validate_psd_signal, validate_psd_signalf, PsdRegion};
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_iirhilbf_interp_decim)]
    fn test_iirhilbf_interp_decim() {
        let tol = 1.0; // error tolerance [dB]
        let bw = 0.4; // pulse bandwidth
        let as_ = 60.0; // transform stop-band suppression
        let p = 40; // pulse semi-length
        let m = 5; // Transform order

        // create transform
        let mut q = IirHilbertFilter::from_order(m).unwrap();
        // q.print();

        let h_len = 2 * p + 1; // pulse length
        let num_samples = h_len + 2 * m + 8;

        let mut buf_0 = vec![Complex32::new(0.0, 0.0); num_samples];
        let mut buf_1 = vec![0.0; num_samples * 2];
        let mut buf_2 = vec![Complex32::new(0.0, 0.0); num_samples];

        // generate the baseband signal (filter pulse)
        let w = 0.36 * bw; // pulse bandwidth
        let h = filter::fir::design::fir_design_kaiser(h_len, w, 80.0, 0.0).unwrap();
        for i in 0..num_samples {
            buf_0[i] = Complex32::new(if i < h_len { 2.0 * w * h[i] } else { 0.0 }, 0.0);
        }

        // run interpolation
        q.interp_execute_block(&buf_0, &mut buf_1).unwrap();

        // clear object
        q.reset();

        // run decimation
        q.decim_execute_block(&buf_1, &mut buf_2).unwrap();

        // verify input spectrum
        #[rustfmt::skip]
        let regions_orig = vec![
            PsdRegion { fmin: -0.5,    fmax: -0.5*bw, pmin: 0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -0.3*bw, fmax: 0.3*bw,  pmin: -1.0, pmax: 1.0,     test_lo: true,  test_hi: true },
            PsdRegion { fmin: 0.5*bw,  fmax: 0.5,     pmin: 0.0, pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signal(&buf_0, &regions_orig).unwrap());

        // verify interpolated spectrum
        #[rustfmt::skip]
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
    #[autotest_annotate(autotest_iirhilbf_filter)]
    fn test_iirhilbf_filter() {
        let tol: f32 = 1.0; // error tolerance [dB]
        let bw: f32 = 0.2; // pulse bandwidth
        let f0: f32 = 0.3; // pulse center frequency
        let ft: f32 = -0.3; // frequency of tone in lower half of band
        let as_: f32 = 60.0; // transform stop-band suppression
        let p: usize = 50; // pulse semi-length
        let m: usize = 7; // Transform order

        // create transform
        let mut q = IirHilbertFilter::from_order(m).unwrap();
        // q.print();

        let h_len: usize = 2 * p + 1; // pulse length
        let num_samples: usize = h_len + 2 * m + 8;

        let mut buf_0 = vec![Complex32::new(0.0, 0.0); num_samples];
        let mut buf_1 = vec![0.0; num_samples];
        let mut buf_2 = vec![Complex32::new(0.0, 0.0); num_samples];

        // generate the baseband signal (filter pulse)
        let w = 0.36 * bw;
        let h = filter::fir::design::fir_design_kaiser(h_len, w, 80.0, 0.0).unwrap();
        for i in 0..num_samples {
            buf_0[i] = if i < h_len {
                2.0 * w * h[i] * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * f0 * i as f32)
            } else {
                Complex32::new(0.0, 0.0)
            };
        }
        for i in 0..num_samples {
            buf_0[i] += if i < h_len {
                1e-3 * kaiser(i, num_samples, 10.0).unwrap()
                    * Complex32::from_polar(1.0, 2.0 * std::f32::consts::PI * ft * i as f32)
            } else {
                Complex32::new(0.0, 0.0)
            };
        }

        // run interpolation
        q.c2r_execute_block(&buf_0, &mut buf_1).unwrap();
        // scale output
        for i in 0..num_samples {
            buf_1[i] *= 2.0;
        }

        // clear object
        q.reset();

        // run decimation
        q.r2c_execute_block(&buf_1, &mut buf_2).unwrap();
        // scale output
        for i in 0..num_samples {
            buf_2[i] *= 0.5;
        }

        // verify input spectrum
        #[rustfmt::skip]
        let regions_orig = vec![
            PsdRegion { fmin: -0.5,       fmax: ft-0.03,   pmin: 0.0,  pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: ft-0.01,    fmax: ft+0.01,   pmin: -40.0,pmax: 0.0,      test_lo: true,  test_hi: false },
            PsdRegion { fmin: ft+0.03,    fmax: f0-0.5*bw, pmin: 0.0,  pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: f0-0.3*bw,  fmax: f0+0.3*bw, pmin: -1.0, pmax: 1.0,      test_lo: true,  test_hi: true },
            PsdRegion { fmin: f0+0.5*bw,  fmax: 0.5,       pmin: 0.0,  pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signal(&buf_0, &regions_orig).unwrap());

        // verify interpolated spectrum
        #[rustfmt::skip]
        let regions_c2r = vec![
            PsdRegion { fmin: -0.5,       fmax: -f0-0.5*bw, pmin: 0.0,  pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: -f0-0.3*bw, fmax: -f0+0.3*bw, pmin: -1.0, pmax: 1.0,      test_lo: true,  test_hi: true },
            PsdRegion { fmin: -f0+0.5*bw, fmax: f0-0.5*bw,  pmin: 0.0,  pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: f0-0.3*bw,  fmax: f0+0.3*bw,  pmin: -1.0, pmax: 1.0,      test_lo: true,  test_hi: true },
            PsdRegion { fmin: f0+0.5*bw,  fmax: 0.5,        pmin: 0.0,  pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signalf(&buf_1, &regions_c2r).unwrap());

        // verify decimated spectrum (using same regions as original)
        #[rustfmt::skip]
        let regions_r2c = vec![
            PsdRegion { fmin: -0.5,       fmax: f0-0.5*bw, pmin: 0.0,  pmax: -as_+tol, test_lo: false, test_hi: true },
            PsdRegion { fmin: f0-0.3*bw,  fmax: f0+0.3*bw, pmin: -1.0, pmax: 1.0,      test_lo: true,  test_hi: true },
            PsdRegion { fmin: f0+0.5*bw,  fmax: 0.5,       pmin: 0.0,  pmax: -as_+tol, test_lo: false, test_hi: true },
        ];
        assert!(validate_psd_signal(&buf_2, &regions_r2c).unwrap());
    }

    #[test]
    #[autotest_annotate(autotest_iirhilbf_invalid_config)]
    fn test_iirhilbf_invalid_config() {
        // check that object returns None for invalid configurations
        assert!(IirHilbertFilter::new(IirFilterShape::Butter, 0, 0.1, 60.0).is_err()); // order out of range
        assert!(IirHilbertFilter::from_order(0).is_err()); // order out of range

        // create proper object and test configuration methods
        // let q = IirHilb::new(IirdesFilterType::Butter, 5, 0.1, 60.0).unwrap();
        // q.print();
    }

    #[test]
    #[autotest_annotate(autotest_iirhilbf_copy_interp)]
    fn test_iirhilbf_copy_interp() {
        // create base object
        let mut q0 = IirHilbertFilter::new(IirFilterShape::Ellip, 7, 0.1, 80.0).unwrap();

        // run interpolator on random data
        let mut y0 = [0.0; 2];
        let mut y1 = [0.0; 2];
        for _ in 0..80 {
            let x = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.interp_execute(x, &mut y0);
        }

        // copy object
        let mut q1 = q0.clone();

        for _ in 0..80 {
            let x = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.interp_execute(x, &mut y0);
            q1.interp_execute(x, &mut y1);
            assert_eq!(y0[0], y1[0]);
            assert_eq!(y0[1], y1[1]);
        }
    }

    #[test]
    #[autotest_annotate(autotest_iirhilbf_copy_decim)]
    fn test_iirhilbf_copy_decim() {
        // create base object
        let mut q0 = IirHilbertFilter::new(IirFilterShape::Ellip, 7, 0.1, 80.0).unwrap();

        // run decimator on random data
        let mut x = [0.0; 2];
        for _ in 0..80 {
            x[0] = crate::random::randnf();
            x[1] = crate::random::randnf();
            let _y0 = q0.decim_execute(&x);
        }

        // copy object and run samples through each in parallel
        let mut q1 = q0.clone();
        for _ in 0..80 {
            x[0] = crate::random::randnf();
            x[1] = crate::random::randnf();
            let y0 = q0.decim_execute(&x);
            let y1 = q1.decim_execute(&x);
            assert_eq!(y0, y1);
        }
    }

    #[test]
    fn test_iirhilbf_execute_block_output_capacity() {
        let mut q = IirHilbertFilter::from_order(5).unwrap();

        // interp: n complex in -> 2n real out
        let x = vec![Complex32::new(1.0, -1.0); 4];
        assert!(q.interp_execute_block(&x, &mut vec![0.0f32; 7]).is_err());
        let mut y = vec![-9.0f32; 10];
        assert_eq!(q.interp_execute_block(&x, &mut y).unwrap(), 8);
        assert_eq!(y[8], -9.0); // tail untouched
        assert_eq!(y[9], -9.0);

        // decim: 2n real in -> n complex out
        let xr = vec![1.0f32; 8];
        assert!(q.decim_execute_block(&xr[..7], &mut vec![Complex32::default(); 4]).is_err());
        assert!(q.decim_execute_block(&xr, &mut vec![Complex32::default(); 3]).is_err());
        let mut yc = vec![Complex32::new(-9.0, -9.0); 6];
        assert_eq!(q.decim_execute_block(&xr, &mut yc).unwrap(), 4);
        assert_eq!(yc[4], Complex32::new(-9.0, -9.0)); // tail untouched

        // one-to-one transforms reject undersized output before advancing state
        let mut q_ref = q.clone();
        assert!(q.r2c_execute_block(&xr, &mut vec![Complex32::default(); 7]).is_err());
        let mut actual = vec![Complex32::default(); xr.len()];
        let mut expected = vec![Complex32::default(); xr.len()];
        assert_eq!(q.r2c_execute_block(&xr, &mut actual).unwrap(), xr.len());
        q_ref.r2c_execute_block(&xr, &mut expected).unwrap();
        assert_eq!(actual, expected);

        let q = IirHilbertFilter::from_order(5).unwrap();
        let input = [Complex32::new(1.0, -0.5), Complex32::new(0.25, 0.75), Complex32::new(-1.0, 0.0)];
        validate_block_execute(&q, &input, -9.0, IirHilbertFilter::c2r_execute_block);
    }
}
