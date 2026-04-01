use crate::error::{Error, Result};
use crate::dotprod::DotProd;
use crate::filter;
use std::f32::consts::PI;
use num_complex::ComplexFloat;

/// Finite impulse response (FIR) interpolator
#[derive(Clone, Debug)]
pub struct FirInterpolationFilter<T, Coeff = T> {
    h_sub_len: usize,
    interpolation_factor: usize,
    filterbank: filter::FirPfbFilter<T, Coeff>,
}

impl<T, Coeff> FirInterpolationFilter<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Copy + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + Default,
    [Coeff]: DotProd<T, Output = T>,
{
    /// Create a new interpolator from external coefficients
    /// 
    /// If the input filter length is not a multiple of the interpolation
    /// factor, the object internally pads the coefficients with zeros
    /// to compensate.
    /// 
    /// # Arguments
    /// 
    /// * `interp` - interpolation factor
    /// * `h` - filter coefficients
    /// * `h_len` - filter length
    /// 
    /// # Returns
    /// 
    /// A new interpolator
    pub fn new(interp: usize, h: &[Coeff], h_len: usize) -> Result<Self> {
        if interp < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }
        if h_len < interp {
            return Err(Error::Config("filter length cannot be less than interp factor".into()));
        }

        let mut h_sub_len = 0;
        while interp * h_sub_len < h_len {
            h_sub_len += 1;
        }

        let h_len_padded = interp * h_sub_len;
        let mut h_padded = vec![Coeff::zero(); h_len_padded];
        h_padded[..h_len].clone_from_slice(&h[..h_len]);

        let filterbank = filter::FirPfbFilter::new(interp, &h_padded, h_len_padded)?;

        Ok(Self {
            h_sub_len,
            interpolation_factor: interp,
            filterbank,
        })
    }

    /// Create a new interpolator from a Kaiser prototype
    /// 
    /// # Arguments
    /// 
    /// * `interp` - interpolation factor
    /// * `m` - filter delay
    /// * `as_` - stop-band attenuation \[dB\]
    /// 
    /// # Returns
    /// 
    /// A new interpolator
    pub fn new_kaiser(interp: usize, m: usize, as_: f32) -> Result<Self> {
        if interp < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if as_ < 0.0 {
            return Err(Error::Config("stop-band attenuation must be positive".into()));
        }

        let h_len = 2 * interp * m + 1;
        let fc = 0.5 / interp as f32;
        let hf = filter::fir_design_kaiser(h_len, fc, as_, 0.0)?;

        let hc: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();
        Self::new(interp, &hc, h_len - 1)
    }

    /// Create a new interpolator from a filter prototype
    /// 
    /// # Arguments
    /// 
    /// * `filter_type` - filter type
    /// * `interp` - interpolation factor
    /// * `m` - filter delay (symbols)
    /// * `beta` - excess bandwidth factor
    /// * `dt` - fractional sample delay
    /// 
    /// # Returns
    /// 
    /// A new interpolator
    pub fn new_prototype(filter_type: filter::FirFilterShape, interp: usize, m: usize, beta: f32, dt: f32) -> Result<Self> {
        if interp < 2 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
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

        let h_len = 2 * interp * m + 1;
        let h = filter::fir_design_prototype(filter_type, interp, m, beta, dt)?;

        let hc: Vec<Coeff> = h.iter().map(|&x| x.into()).collect();
        Self::new(interp, &hc, h_len)
    }

    /// Create a new linear interpolator
    /// 
    /// # Arguments
    /// 
    /// * `interp` - interpolation factor
    /// 
    /// # Returns
    /// 
    /// A new linear interpolator
    pub fn new_linear(interp: usize) -> Result<Self> {
        if interp < 1 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }

        let mut hc = vec![Coeff::zero(); 2 * interp];
        for i in 0..interp {
            hc[i] = (i as f32 / interp as f32).into();
            hc[interp + i] = (1.0 - i as f32 / interp as f32).into();
        }

        Self::new(interp, &hc, 2 * interp)
    }

    /// Create a new window interpolator
    /// 
    /// # Arguments
    /// 
    /// * `interp` - interpolation factor
    /// * `m` - filter semi-length
    /// 
    /// # Returns
    ///
    /// A new window interpolator
    pub fn new_window(interp: usize, m: usize) -> Result<Self> {
        if interp < 1 {
            return Err(Error::Config("interp factor must be greater than 1".into()));
        }
        if m < 1 {
            return Err(Error::Config("filter semi-length must be greater than 0".into()));
        }

        let h_len = 2 * m * interp;
        let mut hc = vec![Coeff::zero(); h_len];
        for i in 0..h_len {
            hc[i] = (PI * i as f32 / (2 * m * interp) as f32).sin().powi(2).into();
        }

        Self::new(interp, &hc, h_len)
    }

    /// Reset the interpolator
    pub fn reset(&mut self) -> () {
        self.filterbank.reset()
    }

    /// Get the interpolation rate
    /// 
    /// # Returns
    /// 
    /// The interpolation rate
    pub fn get_interp_rate(&self) -> usize {
        self.interpolation_factor
    }

    /// Get the sub-filter length (length of each poly-phase filter)
    /// 
    /// # Returns
    /// 
    /// The sub-filter length
    pub fn get_sub_len(&self) -> usize {
        self.h_sub_len
    }

    /// Set the output scaling for interpolator
    /// 
    /// # Arguments
    /// 
    /// * `scale` - scaling factor to apply to each output sample
    pub fn set_scale(&mut self, scale: Coeff) -> () {
        self.filterbank.set_scale(scale)
    }

    /// Get the output scaling for interpolator
    /// 
    /// # Returns
    /// 
    /// The output scaling factor
    pub fn get_scale(&self) -> Coeff {
        self.filterbank.get_scale()
    }

    /// Execute the interpolator on a single input sample and write the
    /// corresponding output samples
    /// 
    /// # Arguments
    /// 
    /// * `x` - input sample
    /// * `y` - output samples (size: `interp` x 1)
    pub fn execute(&mut self, x: T, y: &mut [T]) -> Result<()> {
        self.filterbank.push(x);

        for i in 0..self.interpolation_factor {
            y[i] = self.filterbank.execute(i)?;
        }
        Ok(())
    }

    /// Execute the interpolator on a block of input samples
    /// 
    /// # Arguments
    /// 
    /// * `x` - input samples (size: `n` x 1)
    /// * `y` - output samples (size: `n * interp` x 1)
    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        for (i, &xi) in x.iter().enumerate() {
            self.execute(xi, &mut y[i * self.interpolation_factor..(i + 1) * self.interpolation_factor])?;
        }
        Ok(())
    }

    /// Execute the interpolator with zero-valued input (e.g. flush internal state)
    /// 
    /// # Arguments
    /// 
    /// * `y` - output samples (size: `interp` x 1)
    pub fn flush(&mut self, y: &mut [T]) -> Result<()> {
        self.execute(T::zero(), y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;
    use num_complex::Complex32;

    #[test]
    #[autotest_annotate(autotest_firinterp_rrrf_common)]
    fn test_firinterp_rrrf_common() {
        let interp = FirInterpolationFilter::<f32, f32>::new_kaiser(17, 4, 60.0).unwrap();
        assert_eq!(interp.get_interp_rate(), 17);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_crcf_common)]
    fn test_firinterp_crcf_common() {
        let interp = FirInterpolationFilter::<Complex32, f32>::new_kaiser(7, 4, 60.0).unwrap();
        assert_eq!(interp.get_interp_rate(), 7);
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_rrrf_generic)]
    fn test_firinterp_rrrf_generic() {
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
        let mut interp = FirInterpolationFilter::<f32, f32>::new(m, &h, h.len()).unwrap();

        let x = [1.0, -1.0, 1.0, 1.0];
        let mut y = [0.0; 16];
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
            interp.execute(x[i], &mut y[i*m..(i+1)*m]).unwrap();
        }

        for i in 0..16 {
            assert_relative_eq!(y[i], test[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firinterp_crcf_generic)]
    fn test_firinterp_crcf_generic() {
        // h = [0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0];
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
        let mut interp = FirInterpolationFilter::<Complex32, f32>::new(m, &h, h.len()).unwrap();

        //  x = [1+j*0.2, -0.2+j*1.3, 0.5+j*0.3, 1.1-j*0.2]
        let x: [Complex32; 4] = [
            Complex32::new( 1.0000e+00,  2.0000e-01),
            Complex32::new(-2.0000e-01,  1.3000e+00),
            Complex32::new( 5.0000e-01,  3.0000e-01),
            Complex32::new( 1.1000e+00, -2.0000e-01)
        ];
            
        let mut y = [Complex32::new(0.0, 0.0); 16];

        // z = [x(1) 0 0 0 x(2) 0 0 0 x(3) 0 0 0 x(4) 0 0 0];
        // test = filter(h,1,z)
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
            interp.execute(x[i], &mut y[i*m..(i+1)*m]).unwrap();
        }

        for i in 0..16 {
            assert_relative_eq!(y[i].re, test[i].re, epsilon = tol);
            assert_relative_eq!(y[i].im, test[i].im, epsilon = tol);
        }
    }

    fn testbench_firinterp_crcf_nyquist(ftype: filter::FirFilterShape, m: usize, k: usize, beta: f32) {
        let tol = 1e-6;
        // create interpolator object
        let mut interp = FirInterpolationFilter::<Complex32, f32>::new_prototype(ftype, m, k, beta, 0.0).unwrap();

        // create input buffer of symbols to interpolate
        let num_symbols = k + 16;
        let mut x = vec![Complex32::new(0.0, 0.0); num_symbols]; // input symbols
        let mut y = vec![Complex32::new(0.0, 0.0); m];           // output interp buffer

        for i in 0..num_symbols {
            x[i] = Complex32::from_polar(1.0, 0.7 * i as f32);
        }

        for i in 0..num_symbols {
            // interpolate and store into output buffer
            interp.execute(x[i], &mut y).unwrap();

            // for a Nyquist filter, output should match input at
            // proper sampling time (compensating for delay)
            if i >= k {
                assert_relative_eq!(x[i - k].re, y[0].re, epsilon = tol);
                assert_relative_eq!(x[i - k].im, y[0].im, epsilon = tol);
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
        // TODO replace with sumsqcf
        assert!(buf.iter().map(|x| x.norm_sqr()).sum::<f32>() > 0.0);

        // flush buffer
        for _ in 0..(2 * m) {
            q.flush(&mut buf).unwrap();
        }

        // ensure buffer contains only zeros
        assert_relative_eq!(buf.iter().map(|x| x.norm_sqr()).sum::<f32>(), 0.0);

        // objects are automatically destroyed when they go out of scope
    }
}