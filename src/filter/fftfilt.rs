use crate::error::{Error, Result};
use crate::fft::{Fft, Direction};
use num_traits::Zero;
use num_complex::{ComplexFloat, Complex32};

pub trait FromComplex32 {
    fn from_complex32(c: Complex32) -> Self;
}

impl FromComplex32 for Complex32 {
    fn from_complex32(c: Complex32) -> Self {
        c
    }
}

impl FromComplex32 for f32 {
    fn from_complex32(c: Complex32) -> Self {
        c.re()
    }
}

#[derive(Debug, Clone)]
pub struct FftFilt<T, Coeff = T> {
    h: Vec<Coeff>,
    h_len: usize,
    n: usize,

    time_buf: Vec<Complex32>,
    freq_buf: Vec<Complex32>,
    h_freq: Vec<Complex32>,
    w: Vec<Complex32>,

    fft: Fft<f32>,
    ifft: Fft<f32>,
    scale: Coeff,

    _phantom: std::marker::PhantomData<T>,
}

impl<T, Coeff> FftFilt<T, Coeff>
where
    Coeff: Clone + Default + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Default + ComplexFloat<Real = f32> + FromComplex32,
    Complex32: From<Coeff> + From<T>,
{
    pub fn create(h: &[Coeff], n: usize) -> Result<Self> {
        let h_len = h.len();
        if h_len == 0 {
            return Err(Error::Config("filter length must be greater than zero".into()));
        }
        if n < h_len - 1 {
            return Err(Error::Config(format!("block length must be greater than h_len-1 ({})", h_len - 1)));
        }

        let mut q = Self {
            h: h.to_vec(),
            h_len,
            n,
            time_buf: vec![Complex32::zero(); 2 * n],
            freq_buf: vec![Complex32::zero(); 2 * n],
            h_freq: vec![Complex32::zero(); 2 * n],
            w: vec![Complex32::zero(); n],
            fft: Fft::new(2 * n, Direction::Forward),
            ifft: Fft::new(2 * n, Direction::Backward),
            scale: Coeff::one(),
            _phantom: std::marker::PhantomData,
        };

        // compute FFT of filter coefficients and copy to internal H array
        for i in 0..2 * q.n {
            q.time_buf[i] = if i < q.h_len {
                Complex32::from(q.h[i].clone())
            } else {
                Complex32::zero()
            };
        }
        q.fft.run(&mut q.time_buf, &mut q.freq_buf);
        q.h_freq.copy_from_slice(&q.freq_buf);

        q.set_scale(Coeff::one());
        q.reset();

        Ok(q)
    }

    pub fn reset(&mut self) {
        self.w.fill(Complex32::zero());
    }

    // pub fn print(&self) {
    //     println!("<liquid.fftfilt_{}, len={}, nfft={}, scale={:?}>",
    //              std::any::type_name::<TC>(), self.h_len, self.n, self.scale);
    // }

    pub fn set_scale(&mut self, scale: Coeff) {
        self.scale = scale / (2.0 * self.n as f32).into();
    }

    pub fn get_scale(&self) -> Coeff {
        self.scale * (2.0 * self.n as f32).into() 
    }

    pub fn execute(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        if x.len() != self.n || y.len() != self.n {
            return Err(Error::Config("input and output lengths must match filter block size".into()));
        }

        // copy input
        for i in 0..self.n {
            self.time_buf[i] = Complex32::from(x[i]);
        }

        // pad end of time-domain buffer with zeros
        for i in self.n..2 * self.n {
            self.time_buf[i] = Complex32::zero();
        }

        // run forward transform
        self.fft.run(&mut self.time_buf, &mut self.freq_buf);

        // compute inner product between FFT{ x } and FFT{ H }
        for i in 0..2 * self.n {
            self.freq_buf[i] *= self.h_freq[i];
        }

        // compute inverse transform
        self.ifft.run(&mut self.freq_buf, &mut self.time_buf);

        // copy output summed with buffer
        for i in 0..self.n {
            y[i] = T::from_complex32((self.time_buf[i] + self.w[i]) * Complex32::from(self.scale));
        }

        // copy buffer
        self.w.copy_from_slice(&self.time_buf[self.n..2 * self.n]);

        Ok(())
    }

    pub fn get_length(&self) -> usize {
        self.h_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_fftfilt_config)]
    fn test_fftfilt_config() {
        // check that object returns error for invalid configurations
        let h_1: [f32; 0] = [];
        let h_2: [f32; 9] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(FftFilt::<f32, f32>::create(&h_1, 64).is_err()); // filter length too small
        assert!(FftFilt::<f32, f32>::create(&h_2, 7).is_err()); // block length too small

        // create proper object and test configurations
        let mut filt = FftFilt::<f32, f32>::create(&h_2, 64).unwrap();

        filt.set_scale(3.0);
        assert_eq!(filt.get_scale(), 3.0);
        assert_eq!(filt.get_length(), 9);
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_copy)]
    fn test_fftfilt_copy() {
        // generate random filter coefficients
        let h_len = 31;
        let h: Vec<f32> = (0..h_len).map(|_| crate::random::randnf()).collect();

        // determine appropriate block size
        // NOTE: this number can be anything at least h_len-1
        let n = 96;

        // create object
        let mut q0 = FftFilt::<Complex32, f32>::create(&h, n).unwrap();

        // compute output in blocks of size 'n'
        let mut buf = vec![Complex32::zero(); n];
        let mut buf_0 = vec![Complex32::zero(); n];
        let mut buf_1 = vec![Complex32::zero(); n];

        for _ in 0..10 {
            for j in 0..n {
                buf[j] = Complex32::new(crate::random::randnf(), crate::random::randnf());
            }
            q0.execute(&buf, &mut buf_0).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run filters in parallel and compare results
        for _ in 0..10 {
            for j in 0..n {
                buf[j] = Complex32::new(crate::random::randnf(), crate::random::randnf());
            }
            q0.execute(&buf, &mut buf_0).unwrap();
            q1.execute(&buf, &mut buf_1).unwrap();

            assert_eq!(buf_0, buf_1);
        }
    }

    include!("test_data.rs");

    fn fftfilt_rrrf_test(h: &[f32], x: &[f32], y: &[f32]) {
        let tol = 0.001f32;

        // determine appropriate block size
        // NOTE: this number can be anything at least h.len()-1
        let n = 1 << crate::math::nextpow2(h.len() as u32 - 1).unwrap();

        // determine number of blocks
        let (quot, rem) = (x.len() / n, x.len() % n);
        let num_blocks = quot + if rem > 0 { 1 } else { 0 };

        // load filter coefficients externally
        let mut q = FftFilt::<f32, f32>::create(h, n).unwrap();

        // allocate memory for output
        let mut y_test = vec![0.0; n * num_blocks];

        // compute output in blocks of size 'n'
        for i in 0..num_blocks {
            q.execute(&x[i*n..(i+1)*n], &mut y_test[i*n..(i+1)*n]).unwrap();
        }

        // compare results
        for i in 0..y.len() {
            assert_abs_diff_eq!(y_test[i], y[i], epsilon = tol);
        }
    }

    fn fftfilt_crcf_test(h: &[f32], x: &[Complex32], y: &[Complex32]) {
        let tol = 0.001f32;

        // determine appropriate block size
        // NOTE: this number can be anything at least h.len()-1
        let n = 1 << crate::math::nextpow2(h.len() as u32 - 1).unwrap();

        // determine number of blocks
        let (quot, rem) = (x.len() / n, x.len() % n);
        let num_blocks = quot + if rem > 0 { 1 } else { 0 };

        // load filter coefficients externally
        let mut q = FftFilt::<Complex32, f32>::create(h, n).unwrap();

        // allocate memory for output
        let mut y_test = vec![Complex32::zero(); n * num_blocks];

        // compute output in blocks of size 'n'
        for i in 0..num_blocks {
            q.execute(&x[i*n..(i+1)*n], &mut y_test[i*n..(i+1)*n]).unwrap();
        }

        // compare results
        for i in 0..y.len() {
            assert_abs_diff_eq!(y_test[i].re, y[i].re, epsilon = tol);
            assert_abs_diff_eq!(y_test[i].im, y[i].im, epsilon = tol);
        }
    }

    fn fftfilt_cccf_test(h: &[Complex32], x: &[Complex32], y: &[Complex32]) {
        let tol = 0.001f32;

        // determine appropriate block size
        // NOTE: this number can be anything at least h.len()-1
        let n = 1 << crate::math::nextpow2(h.len() as u32 - 1).unwrap();

        // determine number of blocks
        let (quot, rem) = (x.len() / n, x.len() % n);
        let num_blocks = quot + if rem > 0 { 1 } else { 0 };

        // load filter coefficients externally
        let mut q = FftFilt::<Complex32, Complex32>::create(h, n).unwrap();

        // allocate memory for output
        let mut y_test = vec![Complex32::zero(); n * num_blocks];

        // compute output in blocks of size 'n'
        for i in 0..num_blocks {
            q.execute(&x[i*n..(i+1)*n], &mut y_test[i*n..(i+1)*n]).unwrap();
        }

        // compare results
        for i in 0..y.len() {
            assert_abs_diff_eq!(y_test[i].re, y[i].re, epsilon = tol);
            assert_abs_diff_eq!(y_test[i].im, y[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_rrrf_data_h4x256)]
    fn test_fftfilt_rrrf_data_h4x256() {
        fftfilt_rrrf_test(
            &FFTFILT_RRRF_DATA_H4X256_H,
            &FFTFILT_RRRF_DATA_H4X256_X,
            &FFTFILT_RRRF_DATA_H4X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_rrrf_data_h7x256)]
    fn test_fftfilt_rrrf_data_h7x256() {
        fftfilt_rrrf_test(
            &FFTFILT_RRRF_DATA_H7X256_H,
            &FFTFILT_RRRF_DATA_H7X256_X,
            &FFTFILT_RRRF_DATA_H7X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_rrrf_data_h13x256)]
    fn test_fftfilt_rrrf_data_h13x256() {
        fftfilt_rrrf_test(
            &FFTFILT_RRRF_DATA_H13X256_H,
            &FFTFILT_RRRF_DATA_H13X256_X,
            &FFTFILT_RRRF_DATA_H13X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_rrrf_data_h23x256)]
    fn test_fftfilt_rrrf_data_h23x256() {
        fftfilt_rrrf_test(
            &FFTFILT_RRRF_DATA_H23X256_H,
            &FFTFILT_RRRF_DATA_H23X256_X,
            &FFTFILT_RRRF_DATA_H23X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_crcf_data_h4x256)]
    fn test_fftfilt_crcf_data_h4x256() {
        fftfilt_crcf_test(
            &FFTFILT_CRCF_DATA_H4X256_H,
            &FFTFILT_CRCF_DATA_H4X256_X,
            &FFTFILT_CRCF_DATA_H4X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_crcf_data_h7x256)]
    fn test_fftfilt_crcf_data_h7x256() {
        fftfilt_crcf_test(
            &FFTFILT_CRCF_DATA_H7X256_H,
            &FFTFILT_CRCF_DATA_H7X256_X,
            &FFTFILT_CRCF_DATA_H7X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_crcf_data_h13x256)]
    fn test_fftfilt_crcf_data_h13x256() {
        fftfilt_crcf_test(
            &FFTFILT_CRCF_DATA_H13X256_H,
            &FFTFILT_CRCF_DATA_H13X256_X,
            &FFTFILT_CRCF_DATA_H13X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_crcf_data_h23x256)]
    fn test_fftfilt_crcf_data_h23x256() {
        fftfilt_crcf_test(
            &FFTFILT_CRCF_DATA_H23X256_H,
            &FFTFILT_CRCF_DATA_H23X256_X,
            &FFTFILT_CRCF_DATA_H23X256_Y
        );
    }
    
    #[test]
    #[autotest_annotate(autotest_fftfilt_cccf_data_h4x256)]
    fn test_fftfilt_cccf_data_h4x256() {
        fftfilt_cccf_test(
            &FFTFILT_CCCF_DATA_H4X256_H,
            &FFTFILT_CCCF_DATA_H4X256_X,
            &FFTFILT_CCCF_DATA_H4X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_cccf_data_h7x256)]
    fn test_fftfilt_cccf_data_h7x256() {
        fftfilt_cccf_test(
            &FFTFILT_CCCF_DATA_H7X256_H,
            &FFTFILT_CCCF_DATA_H7X256_X,
            &FFTFILT_CCCF_DATA_H7X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_cccf_data_h13x256)]
    fn test_fftfilt_cccf_data_h13x256() {
        fftfilt_cccf_test(
            &FFTFILT_CCCF_DATA_H13X256_H,
            &FFTFILT_CCCF_DATA_H13X256_X,
            &FFTFILT_CCCF_DATA_H13X256_Y
        );
    }

    #[test]
    #[autotest_annotate(autotest_fftfilt_cccf_data_h23x256)]
    fn test_fftfilt_cccf_data_h23x256() {
        fftfilt_cccf_test(
            &FFTFILT_CCCF_DATA_H23X256_H,
            &FFTFILT_CCCF_DATA_H23X256_X,
            &FFTFILT_CCCF_DATA_H23X256_Y
        );
    }
}