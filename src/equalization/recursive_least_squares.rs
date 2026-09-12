use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::matrix::{matrix_access, matrix_access_mut, matrix_mul, FloatComplex};

#[derive(Clone, Debug)]
#[doc(alias = "Eqrls")]
pub struct RecursiveLeastSquaresEqualizer<T> {
    p: usize,      // filter order
    lambda: f32,   // RLS forgetting factor
    delta: f32,    // RLS initialization factor
    h0: Vec<T>,    // initial coefficients
    w0: Vec<T>,    // weights [px1]
    w1: Vec<T>,    // weights [px1]
    p0: Vec<T>,    // recursion matrix [pxp]
    p1: Vec<T>,    // recursion matrix [pxp]
    g: Vec<T>,     // gain vector [px1]
    xp0: Vec<T>,   // [1xp]
    zeta: T,       // constant
    gxl: Vec<T>,   // [pxp]
    gxlp0: Vec<T>, // [pxp]
    n: usize,      // input counter
    buffer: Window<T>,
}

impl<T> RecursiveLeastSquaresEqualizer<T>
where
    T: Clone + Copy + From<f32> + Default + FloatComplex,
    [T]: DotProd<T, Output = T>,
{
    /// Create an equalizer of the specified length with a unit coefficient at
    /// the final tap
    pub fn new(filter_length: usize) -> Result<Self> {
        let mut q = Self::new_base(filter_length)?;
        q.h0[filter_length - 1] = 1.0.into();
        q.reset();
        Ok(q)
    }

    /// Create an equalizer with the specified initial coefficients
    pub fn from_coefficients(coefficients: &[T]) -> Result<Self> {
        let mut q = Self::new_base(coefficients.len())?;
        q.h0.copy_from_slice(coefficients);
        q.reset();
        Ok(q)
    }

    fn new_base(filter_length: usize) -> Result<Self> {
        if filter_length == 0 {
            return Err(Error::Config("equalizer length must be greater than 0".into()));
        }

        Ok(Self {
            p: filter_length,
            lambda: 0.99,
            delta: 0.1,
            h0: vec![0.0.into(); filter_length],
            w0: vec![0.0.into(); filter_length],
            w1: vec![0.0.into(); filter_length],
            p0: vec![0.0.into(); filter_length * filter_length],
            p1: vec![0.0.into(); filter_length * filter_length],
            g: vec![0.0.into(); filter_length],
            xp0: vec![0.0.into(); filter_length],
            zeta: 0.0.into(),
            gxl: vec![0.0.into(); filter_length * filter_length],
            gxlp0: vec![0.0.into(); filter_length * filter_length],
            n: 0,
            buffer: Window::new(filter_length)?,
        })
    }

    /// Replace the equalizer coefficients without clearing its buffered input
    /// or adaptive state
    pub fn set_coefficients(&mut self, coefficients: &[T]) -> Result<()> {
        if coefficients.len() != self.p {
            return Err(Error::Config(format!(
                "coefficient length must match equalizer length: {} != {}",
                coefficients.len(),
                self.p
            )));
        }

        self.h0.copy_from_slice(coefficients);
        self.w0.copy_from_slice(coefficients);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.n = 0;

        for i in 0..self.p {
            for j in 0..self.p {
                let v = if i == j { 1.0 / self.delta } else { 0.0 };
                matrix_access_mut(&mut self.p0, self.p, self.p, i, j, v.into());
            }
        }

        self.w0.copy_from_slice(&self.h0);
        self.buffer.reset();
    }

    pub fn bw(&self) -> f32 {
        self.lambda
    }

    pub fn set_bw(&mut self, forgetting_factor: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&forgetting_factor) {
            return Err(Error::Config("forgetting factor must be in (0,1)".into()));
        }
        self.lambda = forgetting_factor;
        Ok(())
    }

    pub fn push(&mut self, input: T) {
        self.buffer.push(input)
    }

    pub fn execute(&self) -> Result<T> {
        let r = self.buffer.read();
        let y = self.w0.dotprod(r);
        Ok(y)
    }

    pub fn step(&mut self, d: T, d_hat: T) -> Result<()> {
        let alpha = d - d_hat;
        let x = self.buffer.read();

        for c in 0..self.p {
            self.xp0[c] = (0..self.p).map(|r| x[r] * matrix_access(&self.p0, self.p, self.p, r, c)).sum();
        }

        self.zeta = self.xp0.iter().zip(x).map(|(&xp, &xi)| xp * xi.conj()).sum::<T>() + self.lambda.into();

        for r in 0..self.p {
            self.g[r] =
                (0..self.p).map(|c| matrix_access(&self.p0, self.p, self.p, r, c) * x[c].conj()).sum::<T>() / self.zeta;
        }

        for r in 0..self.p {
            for c in 0..self.p {
                let v = self.g[r] * x[c] / self.lambda.into();
                matrix_access_mut(&mut self.gxl, self.p, self.p, r, c, v);
            }
        }

        matrix_mul(&self.gxl, self.p, self.p, &self.p0, self.p, self.p, &mut self.gxlp0, self.p, self.p)?;

        for i in 0..self.p * self.p {
            self.p1[i] = self.p0[i] / self.lambda.into() - self.gxlp0[i];
        }

        for i in 0..self.p {
            self.w1[i] = self.w0[i] + alpha * self.g[i];
        }

        self.w0.copy_from_slice(&self.w1);
        self.p0.copy_from_slice(&self.p1);
        Ok(())
    }

    pub fn weights(&self, w: &mut [T]) -> Result<()> {
        if w.len() != self.p {
            return Err(Error::Config("output weights array length must match filter order".into()));
        }
        for i in 0..self.p {
            w[i] = self.w0[self.p - i - 1];
        }
        Ok(())
    }

    pub fn train(&mut self, w: &mut [T], x: &[T], d: &[T], n: usize) -> Result<()> {
        if n < self.p {
            return Err(Error::Config("training sequence less than filter order".into()));
        }

        self.reset();

        for i in 0..self.p {
            self.w0[i] = w[self.p - i - 1];
        }

        for i in 0..n {
            self.push(x[i]);
            let d_hat = self.execute()?;
            self.step(d[i], d_hat)?;
        }

        self.weights(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FirFilter;
    use crate::random::randnf;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[rustfmt::skip]
    const EQRLS_RRRF_AUTOTEST_DATA_SEQUENCE: [f32; 64] = [
        -1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0, 
        -1.0,  1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, 
         1.0, -1.0, -1.0, -1.0,  1.0,  1.0, -1.0,  1.0, 
        -1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0, -1.0,
         1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0, -1.0, 
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 
         1.0, -1.0, -1.0, -1.0,  1.0,  1.0, -1.0,  1.0, 
        -1.0,  1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0
    ];

    // AUTOTEST: channel filter: delta with zero delay
    #[test]
    #[autotest_annotate(autotest_eqrls_rrrf_01)]
    fn test_eqrls_rrrf_01() {
        let tol = 1e-2f32; // error tolerance

        // fixed parameters (do not change)
        let h_len = 4; // channel filter length
        let p = 6; // equalizer order
        let n = 64; // number of symbols to observe

        // bookkeeping variables
        let mut y = vec![0.0f32; n]; // received data sequence (filtered by channel)
                                     //let mut d_hat = vec![0.0f32; n];   // recovered data sequence
        let mut h = vec![0.0f32; h_len]; // channel filter coefficients
        let mut w = vec![0.0f32; p]; // equalizer filter coefficients

        // create equalizer
        let mut eq = RecursiveLeastSquaresEqualizer::<f32>::new(p).unwrap();

        // create channel filter
        h[0] = 1.0f32;
        let mut f = FirFilter::<f32, f32>::new(&h).unwrap();

        // data sequence
        let d = &EQRLS_RRRF_AUTOTEST_DATA_SEQUENCE;

        // filter data signal through channel
        for i in 0..n {
            f.push(d[i]);
            y[i] = f.execute();
        }

        // initialize weights, train equalizer
        eq.train(&mut w, &y, d, n).unwrap();

        // compare filter taps
        assert_abs_diff_eq!(w[0], 1.0f32, epsilon = tol);
        for i in 1..p {
            assert_abs_diff_eq!(w[i], 0.0f32, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_eqrls_rrrf_copy)]
    fn test_eqrls_rrrf_copy() {
        // create initial object
        let mut h = [0.0f32; 9];
        for i in 0..9 {
            h[i] = randnf();
        }
        let mut q0 = RecursiveLeastSquaresEqualizer::<f32>::from_coefficients(&h).unwrap();

        // create channel filter
        let hc = [1.0f32, -0.08f32, 0.32f32, 0.01f32, -0.06f32, 0.07f32, -0.03f32];
        let mut fc = FirFilter::<f32, f32>::new(&hc).unwrap();

        // run training samples through object
        let nstd = 0.001f32;
        let d = &EQRLS_RRRF_AUTOTEST_DATA_SEQUENCE;
        for i in 0..64 {
            let mut v = fc.execute_one(d[i]);
            v += nstd * randnf();
            q0.push(v);
            let y0 = q0.execute().unwrap();
            q0.step(d[i], y0).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run training samples through object
        for i in 0..64 {
            // filtered sample in noise
            let mut v = fc.execute_one(d[i]);
            v += nstd * randnf();

            // push sample through filter
            q0.push(v);
            q1.push(v);

            // compute output
            let y0 = q0.execute().unwrap();
            let y1 = q1.execute().unwrap();
            assert_eq!(y0, y1);

            // step equalization algorithm
            q0.step(d[i], y0).unwrap();
            q1.step(d[i], y1).unwrap();
        }

        // get and compare coefficients
        let mut w0 = vec![0.0f32; 9];
        let mut w1 = vec![0.0f32; 9];
        q0.weights(&mut w0).unwrap();
        q1.weights(&mut w1).unwrap();
        assert_eq!(w0, w1);
    }

    #[test]
    fn test_eqrls_constructor_variants() {
        assert!(RecursiveLeastSquaresEqualizer::<f32>::new(0).is_err());
        assert!(RecursiveLeastSquaresEqualizer::<f32>::from_coefficients(&[]).is_err());

        let coefficients = [1.0f32, 0.5, -0.25];
        let equalizer = RecursiveLeastSquaresEqualizer::from_coefficients(&coefficients).unwrap();
        let mut weights = [0.0f32; 3];
        equalizer.weights(&mut weights).unwrap();
        assert_eq!(weights, [-0.25, 0.5, 1.0]);
    }

    #[test]
    fn test_eqrls_set_coefficients_preserves_buffered_input() {
        let mut equalizer = RecursiveLeastSquaresEqualizer::<f32>::new(3).unwrap();
        equalizer.push(1.0);
        equalizer.push(2.0);
        equalizer.push(3.0);
        assert_eq!(equalizer.execute().unwrap(), 3.0);

        equalizer.set_coefficients(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(equalizer.execute().unwrap(), 1.0);
        let mut weights = [0.0f32; 3];
        equalizer.weights(&mut weights).unwrap();
        assert_eq!(weights, [0.0, 0.0, 1.0]);

        assert!(equalizer.set_coefficients(&[1.0, 0.0]).is_err());
        assert_eq!(equalizer.execute().unwrap(), 1.0);
    }
}
