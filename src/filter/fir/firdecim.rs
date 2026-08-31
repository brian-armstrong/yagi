use crate::error::{Error, Result};
use crate::buffer::Window;
use super::design;
use crate::dotprod::{DotProd, DotProduct};
use crate::matrix::FloatComplex;

use num_complex::Complex32;


/// Finite impulse response (FIR) decimation filter
#[derive(Clone, Debug)]
pub struct FirDecimationFilter<T, Coeff = T> {
    h: Vec<Coeff>,
    dp: DotProduct<T, Coeff>,
    decimation_factor: usize,
    w: Window<T>,
    scale: Coeff,
}

impl<T, Coeff> FirDecimationFilter<T, Coeff>
where
    T: Clone + Copy + FloatComplex<Real = f32>,
    Coeff: Clone + Copy + FloatComplex<Real = f32>,
    T: Clone + Copy + FloatComplex<Real = f32> + std::ops::Mul<Coeff, Output = T>,
    Complex32: From<Coeff>,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Create a new decimation filter from external coefficients
    /// 
    /// # Arguments
    /// 
    /// * `decimation_factor` - The decimation factor
    /// * `h` - The filter coefficients
    /// * `h_len` - The length of the filter coefficients
    /// 
    /// # Returns
    /// 
    /// A new decimation filter
    pub fn new(decimation_factor: usize, h: &[Coeff], h_len: usize) -> Result<Self> {
        if h_len == 0 {
            return Err(Error::Config("filter length must be greater than zero".into()));
        }
        if decimation_factor == 0 {
            return Err(Error::Config("decimation factor must be greater than zero".into()));
        }

        let mut q = Self {
            h: h[..h_len].to_vec(),
            dp: DotProduct::new_rev(&h[..h_len])?,
            decimation_factor,
            w: Window::new(h_len)?,
            scale: Coeff::zero(),
        };

        q.set_scale(Coeff::one());
        q.reset();

        Ok(q)
    }

    /// Create a new decimation filter from a Kaiser-Bessel filter prototype
    /// 
    /// # Arguments
    /// 
    /// * `decimation_factor` - The decimation factor
    /// * `m` - The filter delay
    /// * `as_` - The stop-band attenuation
    /// 
    /// # Returns
    /// 
    /// A new decimation filter
    pub fn new_kaiser(decimation_factor: usize, m: usize, as_: f32) -> Result<Self> {
        if decimation_factor < 2 {
            return Err(Error::Config("decim factor must be greater than 1".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if as_ < 0.0 {
            return Err(Error::Config("stop-band attenuation must be positive".into()));
        }

        let h_len = 2 * decimation_factor * m + 1;
        let fc = 0.5 / decimation_factor as f32;
        let hf = design::fir_design_kaiser(h_len, fc, as_, 0.0)?;

        let hc: Vec<Coeff> = hf.iter().map(|&x| Coeff::from(x).unwrap()).collect();
        Self::new(decimation_factor, &hc, h_len)
    }

    /// Create a new decimation filter from a filter prototype
    /// 
    /// # Arguments
    /// 
    /// * `filter_type` - The filter type
    /// * `decimation_factor` - The decimation factor
    /// * `m` - The filter delay
    /// * `beta` - The excess bandwidth factor
    /// * `dt` - The fractional sample delay
    /// 
    /// # Returns
    /// 
    /// A new decimation filter
    pub fn new_prototype(filter_type: design::FirFilterShape, decimation_factor: usize, m: usize, beta: f32, dt: f32) -> Result<Self> {
        if decimation_factor < 2 {
            return Err(Error::Config("decimation factor must be greater than 1".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(Error::Config("filter excess bandwidth factor must be in [0,1]".into()));
        }
        if dt < -1.0 || dt > 1.0 {
            return Err(Error::Config("filter fractional sample delay must be in [-1,1]".into()));
        }

        let h_len = 2 * decimation_factor * m + 1;
        let h = design::fir_design_prototype(filter_type, decimation_factor, m, beta, dt)?;

        let hc: Vec<Coeff> = h.iter().map(|&x| Coeff::from(x).unwrap()).collect();
        Self::new(decimation_factor, &hc, h_len)
    }

    /// Reset the filter state
    pub fn reset(&mut self) {
        self.w.reset();
    }

    /// Get the decimation rate
    /// 
    /// # Returns
    /// 
    /// The decimation rate
    pub fn get_decim_rate(&self) -> usize {
        self.decimation_factor
    }

    /// Set the output scaling for the filter
    /// 
    /// # Arguments
    /// 
    /// * `scale` - The scaling factor
    pub fn set_scale(&mut self, scale: Coeff) {
        self.scale = scale;
    }

    /// Get the output scaling for the filter
    /// 
    /// # Returns
    /// 
    /// The scaling factor
    pub fn get_scale(&self) -> Coeff {
        self.scale
    }

    /// Compute the frequency response of the filter at a given frequency
    /// 
    /// # Arguments
    /// 
    /// * `fc` - The normalized frequency
    /// 
    /// # Returns
    /// 
    /// The frequency response
    pub fn freqresp(&self, fc: f32) -> Result<Complex32> {
        let mut h_freq = design::freqresponse(&self.h, fc)?;
        h_freq *= Complex32::from(self.scale);
        Ok(h_freq)
    }

    /// Execute the filter on `decimation_factor` input samples
    /// 
    /// # Arguments
    /// 
    /// * `x` - The input samples
    /// 
    /// # Returns
    /// 
    /// The output sample
    pub fn execute(&mut self, x: &[T]) -> Result<T> {
        let mut y = T::zero();
        for i in 0..self.decimation_factor {
            self.w.push(x[i]);

            if i == 0 {
                let r = self.w.read();
                y = self.dp.execute(r);
                y = y * self.scale;
            }
        }
        Ok(y)
    }

    /// Execute the filter on a block of input samples
    ///
    /// # Arguments
    ///
    /// * `x` - The input samples (size: `n * decimation_factor`)
    /// * `n` - The number of output samples
    /// * `y` - The output samples (destination) (size: `n`)
    ///
    /// Returns the number of output samples written, `n`.
    pub fn execute_block(&mut self, x: &[T], n: usize, y: &mut [T]) -> Result<usize> {
        for i in 0..n {
            y[i] = self.execute(&x[i * self.decimation_factor..(i + 1) * self.decimation_factor])?;
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_abs_diff_eq;
    use crate::math::WindowType;
    use crate::filter::fir::design::FirFilterShape;

    #[test]
    fn test_firdecim_freqresp_matches_firfilt() {
        use crate::filter::FirFilter;

        let h: Vec<f32> = (0..21).map(|i| (i as f32 * 0.31).sin() * (1.0 - i as f32 / 40.0)).collect();

        let q = FirDecimationFilter::<Complex32, f32>::new(3, &h, h.len()).unwrap();
        let r = FirFilter::<Complex32, f32>::new(&h).unwrap();

        for k in -8..=8 {
            let fc = k as f32 / 20.0;
            let y = q.freqresp(fc).unwrap();
            let y_test = r.freqresponse(fc);

            assert_abs_diff_eq!(y.re, y_test.re, epsilon = 1e-5);
            assert_abs_diff_eq!(y.im, y_test.im, epsilon = 1e-5);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_config)]
    fn test_firdecim_config() {
        // design filter
        let m = 4;
        let n = 12;
        let h_len = 2 * m * n + 1;
        let wtype = WindowType::Hamming;
        let h = design::fir_design_windowf(wtype, h_len, 0.2, 0.0).unwrap();

        // check that estimate methods return None for invalid configs
        assert!(FirDecimationFilter::<Complex32, f32>::new(0, &h, h_len).is_err()); // M cannot be 0
        assert!(FirDecimationFilter::<Complex32, f32>::new(m, &h, 0).is_err()); // h_len cannot be 0

        assert!(FirDecimationFilter::<Complex32, f32>::new_kaiser(1, 12, 60.0).is_err()); // M too small
        assert!(FirDecimationFilter::<Complex32, f32>::new_kaiser(4, 0, 60.0).is_err()); // m too small
        assert!(FirDecimationFilter::<Complex32, f32>::new_kaiser(4, 12, -2.0).is_err()); // As too small

        // assert!(FirDecim::<Complex32, f32>::new_prototype(FirdesFilterType::Unknown, 4, 12, 0.3, 0.0).is_err());
        assert!(FirDecimationFilter::<Complex32, f32>::new_prototype(FirFilterShape::Rcos, 1, 12, 0.3, 0.0).is_err());
        assert!(FirDecimationFilter::<Complex32, f32>::new_prototype(FirFilterShape::Rcos, 4, 0, 0.3, 0.0).is_err());
        assert!(FirDecimationFilter::<Complex32, f32>::new_prototype(FirFilterShape::Rcos, 4, 12, 7.2, 0.0).is_err());
        assert!(FirDecimationFilter::<Complex32, f32>::new_prototype(FirFilterShape::Rcos, 4, 12, 0.3, 4.0).is_err());

        // create valid object and test configuration
        let mut decim = FirDecimationFilter::<Complex32, f32>::new_kaiser(m, n, 60.0).unwrap();
        decim.set_scale(8.0);
        assert_eq!(decim.get_scale(), 8.0);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_block)]
    fn test_firdecim_block() {
        let m = 4;
        let n = 12;
        let beta = 0.3;

        let num_blocks = 10 + n;
        let mut buf_0 = vec![Complex32::new(0.0, 0.0); m * num_blocks]; // input
        let mut buf_1 = vec![Complex32::new(0.0, 0.0); num_blocks]; // output (regular)
        let mut buf_2 = vec![Complex32::new(0.0, 0.0); num_blocks]; // output (block)

        let mut decim = FirDecimationFilter::<Complex32, f32>::new_prototype(
            FirFilterShape::Arkaiser, m, n, beta, 0.0).unwrap();

        // create random-ish input (does not really matter what the input is
        // so long as the outputs match, but systematic for repeatability)
        for i in 0..m * num_blocks {
            buf_0[i] = Complex32::from_polar(1.0, 0.2 * i as f32 + 1e-5 * (i * i) as f32 + 0.1 * (i as f32).cos());
        }

        // regular execute
        decim.reset();
        for i in 0..num_blocks {
            buf_1[i] = decim.execute(&buf_0[i * m..(i + 1) * m]).unwrap();
        }

        // block execute
        decim.reset();
        decim.execute_block(&buf_0, num_blocks, &mut buf_2).unwrap();

        // check results
        assert_eq!(buf_1, buf_2);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_rrrf_common)]
    fn test_firdecim_rrrf_common() {
        let decim = FirDecimationFilter::<f32, f32>::new_kaiser(17, 4, 60.0).unwrap();
        assert_eq!(decim.get_decim_rate(), 17);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_crcf_common)]
    fn test_firdecim_crcf_common() {
        let decim = FirDecimationFilter::<Complex32, f32>::new_kaiser(7, 4, 60.0).unwrap();
        assert_eq!(decim.get_decim_rate(), 7);
    }

    include!("firdecim_test_data.rs");

    fn firdecim_rrrf_test(
        m: usize,
        h: &[f32],
        x: &[f32],
        y: &[f32],
    ) {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = FirDecimationFilter::<f32, f32>::new(m, h, h.len()).unwrap();

        // allocate memory for output
        let mut y_test = vec![0.0; y.len()];

        // compute output
        for i in 0..y.len() {
            y_test[i] = q.execute(&x[m*i..m*(i+1)]).unwrap();
            
            assert_abs_diff_eq!(y_test[i], y[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_rrrf_data_M2h4x20)]
    fn test_firdecim_rrrf_data_m2h4x20() {
        firdecim_rrrf_test(2,
                       &FIRDECIM_RRRF_DATA_M2H4X20_H,
                       &FIRDECIM_RRRF_DATA_M2H4X20_X,
                       &FIRDECIM_RRRF_DATA_M2H4X20_Y);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_rrrf_data_M3h7x30)]
    fn test_firdecim_rrrf_data_m3h7x30() {
        firdecim_rrrf_test(3,
                        &FIRDECIM_RRRF_DATA_M3H7X30_H,
                        &FIRDECIM_RRRF_DATA_M3H7X30_X,
                        &FIRDECIM_RRRF_DATA_M3H7X30_Y);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_rrrf_data_M4h13x40)]
    fn test_firdecim_rrrf_data_m4h13x40() {
        firdecim_rrrf_test(4,
                        &FIRDECIM_RRRF_DATA_M4H13X40_H,
                        &FIRDECIM_RRRF_DATA_M4H13X40_X,
                        &FIRDECIM_RRRF_DATA_M4H13X40_Y);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_rrrf_data_M5h23x50)]
    fn test_firdecim_rrrf_data_m5h23x50() {
        firdecim_rrrf_test(5,
                        &FIRDECIM_RRRF_DATA_M5H23X50_H,
                        &FIRDECIM_RRRF_DATA_M5H23X50_X,
                        &FIRDECIM_RRRF_DATA_M5H23X50_Y);
    }

    fn firdecim_crcf_test(m: usize,
                          h: &[f32],
                          x: &[Complex32],
                          y: &[Complex32])
    {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = FirDecimationFilter::<Complex32, f32>::new(m, h, h.len()).unwrap();

        // allocate memory for output
        let mut y_test = vec![Complex32::new(0.0, 0.0); y.len()];

        // compute output
        for i in 0..y.len() {
            y_test[i] = q.execute(&x[m*i..m*(i+1)]).unwrap();
            
            assert_abs_diff_eq!(y_test[i].re, y[i].re, epsilon = tol);
            assert_abs_diff_eq!(y_test[i].im, y[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_crcf_data_M2h4x20)]
    fn test_firdecim_crcf_data_m2h4x20() {
        firdecim_crcf_test(2,
                        &FIRDECIM_CRCF_DATA_M2H4X20_H,
                        &FIRDECIM_CRCF_DATA_M2H4X20_X,
                        &FIRDECIM_CRCF_DATA_M2H4X20_Y);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_crcf_data_M3h7x30)]
    fn test_firdecim_crcf_data_m3h7x30()
    {
        firdecim_crcf_test(3,
                        &FIRDECIM_CRCF_DATA_M3H7X30_H,
                        &FIRDECIM_CRCF_DATA_M3H7X30_X,
                        &FIRDECIM_CRCF_DATA_M3H7X30_Y);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_crcf_data_M4h13x40)]
    fn test_firdecim_crcf_data_m4h13x40()
    {
        firdecim_crcf_test(4,
                        &FIRDECIM_CRCF_DATA_M4H13X40_H,
                        &FIRDECIM_CRCF_DATA_M4H13X40_X,
                        &FIRDECIM_CRCF_DATA_M4H13X40_Y);
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_crcf_data_M5h23x50)]
    fn test_firdecim_crcf_data_m5h23x50()
    {
        firdecim_crcf_test(5,
                        &FIRDECIM_CRCF_DATA_M5H23X50_H,
                        &FIRDECIM_CRCF_DATA_M5H23X50_X,
                        &FIRDECIM_CRCF_DATA_M5H23X50_Y);
    }

    fn firdecim_cccf_test(m: usize,
                          h: &[Complex32],
                          x: &[Complex32],
                          y: &[Complex32])
    {
        let tol = 0.001f32;

        // load filter coefficients externally
        let mut q = FirDecimationFilter::<Complex32, Complex32>::new(m, h, h.len()).unwrap();

        // allocate memory for output
        let mut y_test = vec![Complex32::new(0.0, 0.0); y.len()];

        // compute output
        for i in 0..y.len() {
            y_test[i] = q.execute(&x[m*i..m*(i+1)]).unwrap();
            
            assert_abs_diff_eq!(y_test[i].re, y[i].re, epsilon = tol);
            assert_abs_diff_eq!(y_test[i].im, y[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firdecim_cccf_data_M2h4x20)]
    fn test_firdecim_cccf_data_m2h4x20()
    {
        firdecim_cccf_test(2,
                        &FIRDECIM_CCCF_DATA_M2H4X20_H,
                        &FIRDECIM_CCCF_DATA_M2H4X20_X,
                        &FIRDECIM_CCCF_DATA_M2H4X20_Y);
    }
    #[test]
    #[autotest_annotate(autotest_firdecim_cccf_data_M3h7x30)]
    fn test_firdecim_cccf_data_m3h7x30()
    {
        firdecim_cccf_test(3,
                        &FIRDECIM_CCCF_DATA_M3H7X30_H,
                        &FIRDECIM_CCCF_DATA_M3H7X30_X,
                        &FIRDECIM_CCCF_DATA_M3H7X30_Y);
    }
    #[test]
    #[autotest_annotate(autotest_firdecim_cccf_data_M4h13x40)]
    fn test_firdecim_cccf_data_m4h13x40()
    {
        firdecim_cccf_test(4,
                        &FIRDECIM_CCCF_DATA_M4H13X40_H,
                        &FIRDECIM_CCCF_DATA_M4H13X40_X,
                        &FIRDECIM_CCCF_DATA_M4H13X40_Y);
    }
    #[test]
    #[autotest_annotate(autotest_firdecim_cccf_data_M5h23x50)]
    fn test_firdecim_cccf_data_m5h23x50()
    {
        firdecim_cccf_test(5,
                        &FIRDECIM_CCCF_DATA_M5H23X50_H,
                        &FIRDECIM_CCCF_DATA_M5H23X50_X,
                        &FIRDECIM_CCCF_DATA_M5H23X50_Y);
    }
}
