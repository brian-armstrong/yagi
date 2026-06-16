use crate::error::{Error, Result};
use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::matrix::{matrix_access, matrix_access_mut, matrix_mul, FloatComplex};


#[derive(Clone, Debug)]
pub struct Eqrls<T> {
    p: usize,      // filter order
    lambda: f32,   // RLS forgetting factor
    delta: f32,    // RLS initialization factor
    h0: Vec<T>,    // initial coefficients
    w0: Vec<T>,    // weights [px1]
    w1: Vec<T>,    // weights [px1]
    p0: Vec<T>,    // recursion matrix [pxp]
    p1: Vec<T>,    // recursion matrix [pxp]
    g: Vec<T>,      // gain vector [px1]
    xp0: Vec<T>,    // [1xp]
    zeta: T,         // constant
    gxl: Vec<T>,    // [pxp]
    gxlp0: Vec<T>,  // [pxp]
    n: usize,       // input counter
    buffer: Window<T>,
}

impl<T> Eqrls<T>
where
    T: Clone + Copy + From<f32> + Default + FloatComplex,
    [T]: DotProd<T, Output = T>,
{
    pub fn new(h: Option<&[T]>, p: usize) -> Result<Self> {
        if p == 0 {
            return Err(Error::Config("equalizer length must be greater than 0".into()));
        }

        let mut q = Self {
            p,
            lambda: 0.99,
            delta: 0.1,
            h0: vec![0.0.into(); p],
            w0: vec![0.0.into(); p],
            w1: vec![0.0.into(); p],
            p0: vec![0.0.into(); p * p],
            p1: vec![0.0.into(); p * p],
            g: vec![0.0.into(); p],
            xp0: vec![0.0.into(); p],
            zeta: 0.0.into(),
            gxl: vec![0.0.into(); p * p],
            gxlp0: vec![0.0.into(); p * p],
            n: 0,
            buffer: Window::new(p)?,
        };

        if let Some(h) = h {
            q.h0.copy_from_slice(h);
        } else {
            q.h0[p - 1] = 1.0.into();
        }

        q.reset();
        Ok(q)
    }

    pub fn recreate(&mut self, h: Option<&[T]>, p: usize) -> Result<()> {
        if self.p == p {
            if let Some(h) = h {
                self.h0.copy_from_slice(h);
            }
            Ok(())
        } else {
            *self = Self::new(h, p)?;
            Ok(())
        }
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

    pub fn get_bw(&self) -> f32 {
        self.lambda
    }

    pub fn set_bw(&mut self, lambda: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&lambda) {
            return Err(Error::Config("learning rate must be in (0,1)".into()));
        }
        self.lambda = lambda;
        Ok(())
    }

    pub fn push(&mut self, x: T) -> () {
        self.buffer.push(x)
    }

    pub fn execute(&self) -> Result<T> {
        let r = self.buffer.read();
        let y = self.w0.dotprod(&r);
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
            self.g[r] = (0..self.p).map(|c| matrix_access(&self.p0, self.p, self.p, r, c) * x[c].conj()).sum::<T>() / self.zeta;
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

    pub fn get_weights(&self, w: &mut [T]) -> Result<()> {
        if w.len() != self.p {
            return Err(Error::Config("output weights array length must match filter order".into()));
        }
        for i in 0..self.p {
            w[i] = self.w1[self.p - i - 1];
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

        self.get_weights(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use crate::filter::FirFilter;
    use crate::random::randnf;
    use approx::assert_abs_diff_eq;

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
        let tol = 1e-2f32;        // error tolerance

        // fixed parameters (do not change)
        let h_len = 4;   // channel filter length
        let p = 6;       // equalizer order
        let n = 64;      // number of symbols to observe

        // bookkeeping variables
        let mut y = vec![0.0f32; n];         // received data sequence (filtered by channel)
        //let mut d_hat = vec![0.0f32; n];   // recovered data sequence
        let mut h = vec![0.0f32; h_len];     // channel filter coefficients
        let mut w = vec![0.0f32; p];         // equalizer filter coefficients

        // create equalizer
        let mut eq = Eqrls::<f32>::new(None, p).unwrap();

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
        let mut q0 = Eqrls::<f32>::new(Some(&h), 9).unwrap();

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
        q0.get_weights(&mut w0).unwrap();
        q1.get_weights(&mut w1).unwrap();
        assert_eq!(w0, w1);
    }
}