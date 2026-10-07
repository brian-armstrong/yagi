use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use crate::matrix::{FloatComplex, MatrixRef};

/// Adaptive FIR equalizer with a recursive least-squares update.
///
/// Push received samples with [`Self::push`], compute an estimate with
/// [`Self::execute`], then update the weights with [`Self::step`]. For a
/// whole known training sequence, [`Self::train`] does all three. The
/// equalizer keeps a `length x length` matrix, so memory and work per update
/// grow with the square of the length.
///
/// The weight setters and getters use different orders.
/// [`Self::from_coefficients`] and [`Self::set_coefficients`] take history
/// order, with the first weight applied to the oldest sample.
/// [`Self::weights`] and [`Self::train`] use FIR order, with the first weight
/// applied to the newest sample. Reverse the output of `weights` before
/// passing it to either setter.
#[derive(Clone, Debug)]
#[doc(alias = "Eqrls")]
pub struct RecursiveLeastSquaresEqualizer<T> {
    p: usize,    // filter order
    lambda: f32, // RLS forgetting factor
    delta: f32,  // RLS initialization factor
    h0: Vec<T>,  // initial coefficients
    w0: Vec<T>,  // weights [px1]
    p0: Vec<T>,  // recursion matrix [pxp]
    g: Vec<T>,   // gain vector [px1]
    xp0: Vec<T>, // [1xp]
    n: usize,    // input counter
    buffer: Window<T>,
}

impl<T> RecursiveLeastSquaresEqualizer<T>
where
    T: Clone + Copy + From<f32> + Default + FloatComplex,
    [T]: DotProd<T, Output = T>,
{
    /// Creates an equalizer that starts by passing samples through.
    ///
    /// The one nonzero weight is one, applied to the newest sample. The
    /// forgetting factor starts at 0.99. A new equalizer starts with
    /// zero-valued history.
    ///
    /// # Errors
    ///
    /// Returns an error when `filter_length` is zero or its square overflows
    /// `usize`.
    pub fn new(filter_length: usize) -> Result<Self> {
        let mut q = Self::new_base(filter_length)?;
        q.h0[filter_length - 1] = 1.0.into();
        q.reset();
        Ok(q)
    }

    /// Creates an equalizer with initial weights in history order.
    ///
    /// `coefficients[0]` is applied to the oldest sample. This is the reverse
    /// of the order [`Self::weights`] returns. The forgetting factor starts at
    /// 0.99.
    ///
    /// # Errors
    ///
    /// Returns an error when `coefficients` is empty or its length squared
    /// overflows `usize`.
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
        let square_len = filter_length
            .checked_mul(filter_length)
            .ok_or_else(|| Error::Config("equalizer length squared overflows usize".into()))?;

        Ok(Self {
            p: filter_length,
            lambda: 0.99,
            delta: 0.1,
            h0: vec![0.0.into(); filter_length],
            w0: vec![0.0.into(); filter_length],
            p0: vec![0.0.into(); square_len],
            g: vec![0.0.into(); filter_length],
            xp0: vec![0.0.into(); filter_length],
            n: 0,
            buffer: Window::new(filter_length)?,
        })
    }

    /// Replaces the weights with `coefficients`, in history order.
    ///
    /// `coefficients[0]` is applied to the oldest sample. Reverse the output
    /// of [`Self::weights`] before passing it here. The new weights also
    /// become the ones [`Self::reset`] restores. The sample history and the
    /// internal matrix are unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the length of `coefficients` differs from the
    /// equalizer length. The equalizer is unchanged on error.
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

    /// Restores the initial weights, clears the sample history to zeros, and
    /// resets the internal matrix.
    ///
    /// The initial weights are the ones set at construction or by the last
    /// [`Self::set_coefficients`]. The forgetting factor is unchanged.
    pub fn reset(&mut self) {
        self.n = 0;

        self.p0.fill(T::default());
        for (row_index, row) in self.p0.chunks_exact_mut(self.p).enumerate() {
            row[row_index] = (1.0 / self.delta).into();
        }

        self.w0.copy_from_slice(&self.h0);
        self.buffer.reset();
    }

    /// Returns the forgetting factor.
    pub fn bw(&self) -> f32 {
        self.lambda
    }

    /// Sets the forgetting factor `lambda`.
    ///
    /// The update weights a sample `k` steps in the past by `lambda^k`. Values
    /// near one average over more past samples and track changes more
    /// slowly. The factor must lie in `(0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error when `forgetting_factor` is outside `(0, 1]`. The
    /// factor is unchanged on error.
    pub fn set_bw(&mut self, forgetting_factor: f32) -> Result<()> {
        if !(forgetting_factor > 0.0 && forgetting_factor <= 1.0) {
            return Err(Error::Config("forgetting factor must be in (0,1]".into()));
        }
        self.lambda = forgetting_factor;
        Ok(())
    }

    /// Pushes one received sample into the history.
    pub fn push(&mut self, input: T) {
        self.buffer.push(input)
    }

    /// Returns the estimate for the current sample history.
    ///
    /// The equalizer is unchanged. This never returns an error.
    pub fn execute(&self) -> Result<T> {
        let r = self.buffer.read();
        let y = self.w0.dotprod(r);
        Ok(y)
    }

    /// Updates the weights from the desired value `d` and the estimate
    /// `d_hat`.
    ///
    /// `d_hat` should be the result of [`Self::execute`] for the current
    /// history. Unlike the LMS equalizer, updates start with the first
    /// sample. This never returns an error.
    pub fn step(&mut self, d: T, d_hat: T) -> Result<()> {
        let alpha = d - d_hat;
        let x = self.buffer.read();
        let p0 = MatrixRef::new(&self.p0, self.p, self.p)?;

        self.xp0.fill(T::default());
        for (&x_value, p0_row) in x.iter().zip(p0.rows()) {
            for (xp0_value, &p0_value) in self.xp0.iter_mut().zip(p0_row) {
                *xp0_value += x_value * p0_value;
            }
        }

        let zeta = self.xp0.iter().zip(x).map(|(&xp, &xi)| xp * xi.conj()).sum::<T>() + self.lambda.into();

        for (gain, p0_row) in self.g.iter_mut().zip(p0.rows()) {
            *gain = p0_row.iter().zip(x).map(|(&p0_value, &x_value)| p0_value * x_value.conj()).sum::<T>() / zeta;
        }

        // we can do some substitution here to reduce the amount of math needed
        // specifically, we don't need a matrix_mul of (`g` * `x`) and `p0`. we can reassociate
        // p0_new = (p0 - (g * x) * p0)/lambda -> (p0 - (g * xp0))/lambda
        // (we already have xp0 = x * p0, so rather than computing (g * x) and multiplying by p0,
        //  we can just reuse xp0 and multiply by g)
        let lambda: T = self.lambda.into();
        let one: T = 1.0.into();
        let lambda_inv = one / lambda;
        for (p0_row, &gain) in self.p0.chunks_exact_mut(self.p).zip(&self.g) {
            for (p0_value, &xp0_value) in p0_row.iter_mut().zip(&self.xp0) {
                *p0_value = (*p0_value - gain * xp0_value) * lambda_inv;
            }
        }

        for (weight, &gain) in self.w0.iter_mut().zip(&self.g) {
            *weight += alpha * gain;
        }

        Ok(())
    }

    /// Writes the weights to `w` in FIR order.
    ///
    /// `w[0]` is applied to the newest sample. Reverse `w` before passing it
    /// to [`Self::set_coefficients`] or [`Self::from_coefficients`].
    ///
    /// # Errors
    ///
    /// Returns an error when the length of `w` differs from the equalizer
    /// length. `w` is unchanged on error.
    pub fn weights(&self, w: &mut [T]) -> Result<()> {
        if w.len() != self.p {
            return Err(Error::Config("output weights array length must match filter order".into()));
        }
        for i in 0..self.p {
            w[i] = self.w0[self.p - i - 1];
        }
        Ok(())
    }

    /// Resets the equalizer and trains it on `n` received samples `x` and
    /// desired samples `d`.
    ///
    /// `w` gives the starting weights and receives the trained ones, both in
    /// FIR order like [`Self::weights`]. The starting weights do not become
    /// the ones [`Self::reset`] restores. Extra elements of `x` and `d` are
    /// ignored.
    ///
    /// # Errors
    ///
    /// Returns an error when `n` is less than the equalizer length, the
    /// length of `w` differs from it, or `x` or `d` holds fewer than `n`
    /// samples. The equalizer and `w` are unchanged on error.
    pub fn train(&mut self, w: &mut [T], x: &[T], d: &[T], n: usize) -> Result<()> {
        if n < self.p {
            return Err(Error::Config("training sequence less than filter order".into()));
        }
        if w.len() != self.p {
            return Err(Error::Config("weight length must match equalizer length".into()));
        }
        if x.len() < n || d.len() < n {
            return Err(Error::Config(
                "training input and desired lengths must cover the requested sample count".into(),
            ));
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

    #[test]
    fn rls_rejects_zero_forgetting_factor() {
        let mut eq = RecursiveLeastSquaresEqualizer::<f32>::new(3).unwrap();
        let original = eq.bw();
        assert!(eq.set_bw(0.0).is_err());
        assert!(eq.set_bw(f32::NAN).is_err());
        assert_eq!(eq.bw(), original);
    }

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

    #[test]
    fn test_eqrls_train_rejects_short_buffers() {
        let mut equalizer = RecursiveLeastSquaresEqualizer::<f32>::new(3).unwrap();
        let mut weights = [0.0f32; 3];
        let input = [0.0f32; 3];
        let desired = [0.0f32; 3];

        assert!(equalizer.train(&mut weights[..2], &input, &desired, 3).is_err());
        assert!(equalizer.train(&mut weights, &input[..2], &desired, 3).is_err());
        assert!(equalizer.train(&mut weights, &input, &desired[..2], 3).is_err());
    }
}
