use crate::error::{Error, Result};
use crate::buffer::Window;
use crate::dotprod::{DotProd, DotProdKernel};
use crate::filter;
use std::marker::PhantomData;

use num_complex::ComplexFloat;

/// Coefficients and executors for an FIR polyphase filter bank (PFB)
/// Unlike FirPfbFilter, this stores no sample history
#[derive(Clone, Debug)]
pub struct FirPfbBank<T, Coeff = T> {
    num_filters: usize,
    filter_len: usize,
    coefficients: Vec<Coeff>,
    executor: DotProdKernel<T, Coeff, T>,
    scale: Coeff,
    _input: PhantomData<fn(&[T])>,
}

/// Finite impulse response (FIR) polyphase filter bank with internal history
#[derive(Clone, Debug)]
pub struct FirPfbFilter<T, Coeff = T> {
    w: Window<T>,
    bank: FirPfbBank<T, Coeff>,
}

impl<T, Coeff> FirPfbBank<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Copy + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + Default,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Create a new FIR PFB coefficient bank
    /// 
    /// # Arguments
    /// 
    /// * `num_filters` - number of filters in the bank
    /// * `h` - filter coefficients
    /// * `h_len` - filter length
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn new(num_filters: usize, h: &[Coeff], h_len: usize) -> Result<Self> {
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than zero".into()));
        }
        if h_len == 0 {
            return Err(Error::Config("filter length must be greater than zero".into()));
        }

        let filter_len = h_len / num_filters;
        if filter_len == 0 {
            return Err(Error::Config("filter length must be at least the number of filters".into()));
        }

        // store the coefficients in a flat array
        let mut coefficients = Vec::with_capacity(num_filters * filter_len);

        for i in 0..num_filters {
            let start = coefficients.len();
            coefficients.resize(start + filter_len, Coeff::zero());
            let h_sub = &mut coefficients[start..];
            for n in 0..filter_len {
                // load filter in reverse order
                h_sub[filter_len - n - 1] = h[i + n * num_filters];
            }
        }

        // create a single DotProd kernel to be reused for all phases
        let executor = <[T] as DotProd<Coeff>>::plan(filter_len);

        Ok(Self {
            num_filters,
            filter_len,
            coefficients,
            executor,
            scale: Coeff::one(),
            _input: PhantomData,
        })
    }

    /// Create a new FIR PFB filter bank with default parameters
    /// 
    /// This is equivalent to FirPfbBank::new_kaiser(num_filters, m, 0.5, 60.0)
    /// 
    /// # Arguments
    /// 
    /// * `num_filters` - number of filters in the bank
    /// * `m` - filter delay
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn default(num_filters: usize, m: usize) -> Result<Self> {
        Self::new_kaiser(num_filters, m, 0.5, 60.0)
    }

    /// Create a new FIR PFB filter bank using Kaiser-Bessel windowed sinc filter design
    /// 
    /// # Arguments
    /// 
    /// * `num_filters` - number of filters in the bank
    /// * `m` - filter delay
    /// * `fc` - filter normalized cut-off frequency
    /// * `as_` - filter stop-band suppression \[dB\]
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn new_kaiser(num_filters: usize, m: usize, fc: f32, as_: f32) -> Result<Self> {
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than zero".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if fc <= 0.0 || fc > 0.5 {
            return Err(Error::Config("filter cut-off frequency must be in (0,0.5)".into()));
        }
        if as_ < 0.0 {
            return Err(Error::Config("filter excess bandwidth factor must be in [0,1]".into()));
        }

        let h_len = 2 * num_filters * m + 1;
        let hf = filter::fir_design_kaiser(h_len, fc / num_filters as f32, as_, 0.0)?;

        let hc: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();
        Self::new(num_filters, &hc, h_len)
    }

    /// Create a new FIR PFB filter bank using square-root Nyquist prototype filter design
    /// 
    /// # Arguments
    /// 
    /// * `filter_type` - filter type
    /// * `num_filters` - number of filters in the bank
    /// * `k` - samples/symbol
    /// * `m` - filter delay
    /// * `beta` - excess bandwidth factor
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn new_rnyquist(filter_type: filter::FirFilterShape, num_filters: usize, k: usize, m: usize, beta: f32) -> Result<Self> {
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than zero".into()));
        }
        if k < 2 {
            return Err(Error::Config("filter samples/symbol must be greater than 1".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(Error::Config("filter excess bandwidth factor must be in [0,1]".into()));
        }

        let h_len = 2 * num_filters * k * m + 1;
        let hf = filter::fir_design_prototype(filter_type, num_filters * k, m, beta, 0.0)?;

        let hc: Vec<Coeff> = hf.iter().map(|&x| x.into()).collect();
        Self::new(num_filters, &hc, h_len)
    }

    /// Create a new FIR PFB filter bank using square-root derivative Nyquist prototype filter design
    /// 
    /// # Arguments
    /// 
    /// * `filter_type` - filter type
    /// * `num_filters` - number of filters in the bank
    /// * `k` - samples/symbol
    /// * `m` - filter delay
    /// * `beta` - excess bandwidth factor
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn new_drnyquist(filter_type: filter::FirFilterShape, num_filters: usize, k: usize, m: usize, beta: f32) -> Result<Self> {
        if num_filters == 0 {
            return Err(Error::Config("number of filters must be greater than zero".into()));
        }
        if k < 2 {
            return Err(Error::Config("filter samples/symbol must be greater than 1".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(Error::Config("filter excess bandwidth factor must be in [0,1]".into()));
        }

        let h_len = 2 * num_filters * k * m + 1;
        let hf = filter::fir_design_prototype(filter_type, num_filters * k, m, beta, 0.0)?;

        let mut dhf = vec![0.0; h_len];
        let mut hdh_max: f32 = 0.0;
        for i in 0..h_len {
            dhf[i] = if i == 0 {
                hf[i + 1] - hf[h_len - 1]
            } else if i == h_len - 1 {
                hf[0] - hf[i - 1]
            } else {
                hf[i + 1] - hf[i - 1]
            };

            hdh_max = hdh_max.max((hf[i] * dhf[i]).abs());
        }

        let hc: Vec<Coeff> = dhf.iter().map(|&x| (x * 0.06 / hdh_max).into()).collect();
        Self::new(num_filters, &hc, h_len)
    }

    /// Returns the number of coefficient phases in the bank
    pub fn num_filters(&self) -> usize {
        self.num_filters
    }

    /// Returns the number of input samples consumed by each phase
    pub fn filter_len(&self) -> usize {
        self.filter_len
    }

    /// Set the output scaling for the filter bank
    /// 
    /// # Arguments
    /// 
    /// * `scale` - scaling factor to apply to each output sample
    pub fn set_scale(&mut self, scale: Coeff) {
        self.scale = scale;
    }

    /// Get the output scaling for the filter bank
    /// 
    /// # Returns
    /// 
    /// The scaling factor applied to each output sample
    pub fn get_scale(&self) -> Coeff {
        self.scale
    }

    /// Execute one phase against externally managed history
    /// 
    /// # Arguments
    /// 
    /// * `i` - index of filter to use
    /// * `history` - input history, with length [`filter_len`](Self::filter_len)
    pub fn execute(&self, i: usize, history: &[T]) -> Result<T> {
        if i >= self.num_filters {
            return Err(Error::Config(format!("filterbank index ({}) exceeds maximum ({})", i, self.num_filters)));
        }
        assert_eq!(history.len(), self.filter_len, "Invalid filterbank history length");

        Ok(unsafe { self.execute_unchecked(i, history) })
    }

    /// Execute a phase after the caller has validated its index and history
    ///
    /// # Safety
    ///
    /// `i` must be less than [`num_filters`](Self::num_filters), and `history`
    /// must contain exactly [`filter_len`](Self::filter_len) samples.
    #[inline]
    pub(crate) unsafe fn execute_unchecked(&self, i: usize, history: &[T]) -> T {
        debug_assert!(i < self.num_filters);
        debug_assert_eq!(history.len(), self.filter_len);
        unsafe {
            std::hint::assert_unchecked(i < self.num_filters);
            std::hint::assert_unchecked(history.len() == self.filter_len);
        }

        // fetch this phase's coefficients from the flat array
        let start = i * self.filter_len;
        let h = unsafe { self.coefficients.get_unchecked(start..start + self.filter_len) };
        // execute dotprod against our planned kernel
        let y = unsafe { (self.executor)(history, h) };
        y * self.scale
    }
}

impl<T, Coeff> FirPfbFilter<T, Coeff>
where
    Coeff: Clone + Copy + ComplexFloat<Real = f32> + From<f32>,
    T: Clone + Copy + ComplexFloat<Real = f32> + std::ops::Mul<Coeff, Output = T> + Default,
    [T]: DotProd<Coeff, Output = T>,
{
    /// Create a new FIR PFB filter bank
    /// 
    /// # Arguments
    /// 
    /// * `num_filters` - number of filters in the bank
    /// * `h` - filter coefficients
    /// * `h_len` - filter length
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn new(num_filters: usize, h: &[Coeff], h_len: usize) -> Result<Self> {
        Self::from_bank(FirPfbBank::new(num_filters, h, h_len)?)
    }

    /// Create a new FIR PFB filter bank with default parameters
    /// 
    /// This is equivalent to FirPfbFilter::new_kaiser(num_filters, m, 0.5, 60.0)
    /// 
    /// # Arguments
    /// 
    /// * `num_filters` - number of filters in the bank
    /// * `m` - filter delay
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn default(num_filters: usize, m: usize) -> Result<Self> {
        Self::from_bank(FirPfbBank::default(num_filters, m)?)
    }

    /// Create a new FIR PFB filter bank using Kaiser-Bessel windowed sinc filter design
    /// 
    /// # Arguments
    /// 
    /// * `num_filters` - number of filters in the bank
    /// * `m` - filter delay
    /// * `fc` - filter normalized cut-off frequency
    /// * `as_` - filter stop-band suppression \[dB\]
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn new_kaiser(num_filters: usize, m: usize, fc: f32, as_: f32) -> Result<Self> {
        Self::from_bank(FirPfbBank::new_kaiser(num_filters, m, fc, as_)?)
    }

    /// Create a new FIR PFB filter bank using square-root Nyquist prototype filter design
    /// 
    /// # Arguments
    /// 
    /// * `filter_type` - filter type
    /// * `num_filters` - number of filters in the bank
    /// * `k` - samples/symbol
    /// * `m` - filter delay
    /// * `beta` - excess bandwidth factor
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn new_rnyquist(filter_type: filter::FirFilterShape, num_filters: usize, k: usize, m: usize, beta: f32) -> Result<Self> {
        Self::from_bank(FirPfbBank::new_rnyquist(filter_type, num_filters, k, m, beta)?)
    }

    /// Create a new FIR PFB filter bank using square-root derivative Nyquist prototype filter design
    /// 
    /// # Arguments
    /// 
    /// * `filter_type` - filter type
    /// * `num_filters` - number of filters in the bank
    /// * `k` - samples/symbol
    /// * `m` - filter delay
    /// * `beta` - excess bandwidth factor
    /// 
    /// # Returns
    /// 
    /// A new FIR PFB filter bank
    pub fn new_drnyquist(filter_type: filter::FirFilterShape, num_filters: usize, k: usize, m: usize, beta: f32) -> Result<Self> {
        Self::from_bank(FirPfbBank::new_drnyquist(filter_type, num_filters, k, m, beta)?)
    }

    /// Wrap a coefficient bank with newly reset internal history
    pub fn from_bank(bank: FirPfbBank<T, Coeff>) -> Result<Self> {
        let w = Window::new(bank.filter_len())?;
        Ok(Self { w, bank })
    }

    /// Reset the internal history
    pub fn reset(&mut self) -> () {
        self.w.reset();
    }

    /// Returns the underlying coefficient bank
    pub fn bank(&self) -> &FirPfbBank<T, Coeff> {
        &self.bank
    }

    /// Returns the underlying coefficient bank mutably
    pub fn bank_mut(&mut self) -> &mut FirPfbBank<T, Coeff> {
        &mut self.bank
    }

    /// Returns the number of coefficient phases in the bank
    pub fn num_filters(&self) -> usize {
        self.bank.num_filters()
    }

    /// Returns the number of samples retained in the internal history
    pub fn filter_len(&self) -> usize {
        self.bank.filter_len()
    }

    /// Set the output scaling for the filter bank
    /// 
    /// # Arguments
    /// 
    /// * `scale` - scaling factor to apply to each output sample
    pub fn set_scale(&mut self, scale: Coeff) {
        self.bank.set_scale(scale);
    }

    /// Get the output scaling for the filter bank
    /// 
    /// # Returns
    /// 
    /// The scaling factor applied to each output sample
    pub fn get_scale(&self) -> Coeff {
        self.bank.get_scale()
    }

    /// Push a sample into the filter bank
    /// 
    /// # Arguments
    /// 
    /// * `x` - input sample
    pub fn push(&mut self, x: T) -> () {
        self.w.push(x)
    }

    /// Write a block of samples into the filter bank
    /// 
    /// # Arguments
    /// 
    /// * `x` - input samples
    pub fn write(&mut self, x: &[T]) -> () {
        self.w.write(x)
    }

    /// Execute the filter bank on a single input sample
    /// 
    /// # Arguments
    /// 
    /// * `i` - index of filter to use
    /// 
    /// # Returns
    /// 
    /// The output sample
    pub fn execute(&mut self, i: usize) -> Result<T> {
        self.bank.execute(i, self.w.read())
    }

    /// Execute the filter bank on a block of input samples
    /// 
    /// # Arguments
    /// 
    /// * `i` - index of filter to use
    /// * `x` - input samples
    /// * `y` - output samples
    pub fn execute_block(&mut self, i: usize, x: &[T], y: &mut [T]) -> Result<()> {
        for (&xi, yi) in x.iter().zip(y.iter_mut()) {
            self.push(xi);
            *yi = self.execute(i)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_abs_diff_eq;

    #[test]
    #[autotest_annotate(autotest_firpfb_impulse_response)]
    fn test_firpfb_impulse_response() {
        // Initialize variables
        let tol = 1e-4f32;

        // k=2, m=3, beta=0.3, npfb=4;
        // h=rrcos(k*npfb,m,beta);
        let h: [f32; 48] = [
            -0.033116, -0.024181, -0.006284,  0.018261, 
             0.045016,  0.068033,  0.080919,  0.078177, 
             0.056597,  0.016403, -0.038106, -0.098610, 
            -0.153600, -0.189940, -0.194900, -0.158390, 
            -0.075002,  0.054511,  0.222690,  0.415800, 
             0.615340,  0.800390,  0.950380,  1.048100, 
             1.082000,  1.048100,  0.950380,  0.800390, 
             0.615340,  0.415800,  0.222690,  0.054511, 
            -0.075002, -0.158390, -0.194900, -0.189940, 
            -0.153600, -0.098610, -0.038106,  0.016403, 
             0.056597,  0.078177,  0.080919,  0.068033, 
             0.045016,  0.018261, -0.006284, -0.024181
        ];

        // filter input
        let noise: [f32; 12] = [
             0.438310,  1.001900,  0.200600,  0.790040, 
             1.134200,  1.592200, -0.702980, -0.937560, 
            -0.511270, -1.684700,  0.328940, -0.387780
        ];

        // expected filter outputs
        let test: [f32; 4] = [
            2.05558467194397,
            1.56922189602661,
            0.998479744645138,
            0.386125857849177
        ];

        // Load filter coefficients externally
        let mut f = FirPfbFilter::<f32, f32>::new(4, &h, 48).unwrap();
        
        for &n in noise.iter() {
            f.push(n);
        }

        for (i, &expected) in test.iter().enumerate() {
            let y = f.execute(i).unwrap();
            assert_abs_diff_eq!(expected, y, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_firpfb_crcf_copy)]
    fn test_firpfb_crcf_copy() {
        use num_complex::Complex32;

        // create base object with irregular parameters
        let m = 13;
        let h = 7;
        let mut q0 = FirPfbFilter::<Complex32, f32>::default(m, h).unwrap();

        // run random samples through filter
        let num_samples = 80;
        for _ in 0..num_samples {
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            q0.push(v);
        }

        // copy object
        let mut q1 = q0.clone();

        // run random samples through filter
        for _ in 0..num_samples {
            // random input channel and index
            let v = Complex32::new(crate::random::randnf(), crate::random::randnf());
            let idx = rand::random::<usize>() % m;

            // push sample through each filter
            q0.push(v);
            q1.push(v);

            // compare outputs
            let y0 = q0.execute(idx).unwrap();
            let y1 = q1.execute(idx).unwrap();
            assert_eq!(y0, y1);
        }
    }

    #[test]
    fn test_firpfb_bank_matches_filter() {
        use num_complex::Complex32;

        let num_filters = 4;
        let filter_len = 5;
        let h: Vec<f32> = (0..num_filters * filter_len)
            .map(|i| ((i + 1) as f32 * 0.17).sin())
            .collect();
        let mut bank = FirPfbBank::<Complex32, f32>::new(num_filters, &h, h.len()).unwrap();
        bank.set_scale(0.73);

        assert_eq!(bank.num_filters(), num_filters);
        assert_eq!(bank.filter_len(), filter_len);

        let mut filter = FirPfbFilter::from_bank(bank.clone()).unwrap();
        let mut history = Window::new(filter_len).unwrap();

        for i in 0..37 {
            let sample = Complex32::new(
                ((i + 3) as f32 * 0.11).cos(),
                ((i + 5) as f32 * 0.07).sin(),
            );
            history.push(sample);
            filter.push(sample);

            for phase in 0..num_filters {
                let actual = bank.execute(phase, history.read()).unwrap();
                let expected = filter.execute(phase).unwrap();
                assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-6);
                assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_firpfb_bank_invalid_phase() {
        let bank = FirPfbBank::<f32, f32>::new(2, &[1.0, 2.0, 3.0, 4.0], 4).unwrap();
        assert!(bank.execute(2, &[0.0, 0.0]).is_err());
    }

    #[test]
    #[should_panic(expected = "Invalid filterbank history length")]
    fn test_firpfb_bank_invalid_history_length() {
        let bank = FirPfbBank::<f32, f32>::new(2, &[1.0, 2.0, 3.0, 4.0], 4).unwrap();
        let _ = bank.execute(0, &[0.0]);
    }
}
