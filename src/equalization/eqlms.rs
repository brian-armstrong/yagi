use crate::buffer::{WDelay, Window};
use crate::error::{Error, Result};
use crate::filter;
use num_complex::ComplexFloat;

#[derive(Clone, Debug)]
pub struct Eqlms<T> {
    h_len: usize,
    mu: f32,
    h0: Vec<T>,
    w0: Vec<T>,
    w1: Vec<T>,
    count: usize,
    buf_full: bool,
    buffer: Window<T>,
    x2: WDelay<f32>,
    x2_sum: f32,
}

impl<T> Eqlms<T>
where
    T: Clone
        + Copy
        + ComplexFloat<Real = f32>
        + From<f32>
        + Default
        + std::iter::Sum
        + std::ops::Mul<f32, Output = T>
        + std::ops::Div<f32, Output = T>,
    f32: std::ops::Mul<T, Output = T>,
{
    pub fn new(h: Option<&[T]>, h_len: usize) -> Result<Self> {
        let mut q = Self {
            h_len,
            mu: 0.5,
            h0: vec![T::default(); h_len],
            w0: vec![T::default(); h_len],
            w1: vec![T::default(); h_len],
            count: 0,
            buf_full: false,
            buffer: Window::new(h_len)?,
            x2: WDelay::create(h_len)?,
            x2_sum: 0.0,
        };

        if let Some(h) = h {
            for (i, val) in q.h0.iter_mut().enumerate() {
                *val = h[h_len - i - 1].conj();
            }
        } else {
            q.h0[h_len / 2] = 1.0.into();
        }

        q.reset();
        Ok(q)
    }

    pub fn new_rnyquist(filter_type: filter::FirFilterShape, k: usize, m: usize, beta: f32, dt: f32) -> Result<Self> {
        if k < 2 {
            return Err(Error::Config("samples/symbol must be greater than 1".into()));
        }
        if m == 0 {
            return Err(Error::Config("filter delay must be greater than 0".into()));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(Error::Config("filter excess bandwidth factor must be in [0,1]".into()));
        }
        if !(-1.0..=1.0).contains(&dt) {
            return Err(Error::Config("filter fractional sample delay must be in [-1,1]".into()));
        }

        let h_len = 2 * k * m + 1;
        let h = filter::fir_design_prototype(filter_type, k, m, beta, dt)?;
        let hc: Vec<T> = h.iter().map(|&x| (x / k as f32).into()).collect();

        Self::new(Some(&hc), h_len)
    }

    pub fn new_lowpass(h_len: usize, fc: f32) -> Result<Self> {
        if h_len == 0 {
            return Err(Error::Config("filter length must be greater than 0".into()));
        }
        if !(0.0..=0.5).contains(&fc) {
            return Err(Error::Config("filter cutoff must be in (0,0.5]".into()));
        }

        let h = filter::fir_design_kaiser(h_len, fc, 40.0, 0.0)?;
        let hc: Vec<T> = h.iter().map(|&x| x * 2.0.into() * fc).collect();

        Self::new(Some(&hc), h_len)
    }

    pub fn reset(&mut self) {
        self.w0.copy_from_slice(&self.h0);
        self.buffer.reset();
        self.x2.reset();
        self.count = 0;
        self.buf_full = false;
        self.x2_sum = 0.0;
    }

    pub fn get_bw(&self) -> f32 {
        self.mu
    }

    pub fn set_bw(&mut self, mu: f32) -> Result<()> {
        if mu < 0.0 {
            return Err(Error::Config("learning rate cannot be less than zero".into()));
        }
        self.mu = mu;
        Ok(())
    }

    pub fn get_length(&self) -> usize {
        self.h_len
    }

    pub fn get_coefficients(&self) -> &[T] {
        &self.w0
    }

    pub fn get_weights(&self) -> Vec<T> {
        self.w0.iter().rev().map(|&x| x.conj()).collect()
    }

    pub fn push(&mut self, x: T) {
        self.buffer.push(x);
        self.update_sumsq(x);
        self.count += 1;
    }

    pub fn push_block(&mut self, x: &[T]) {
        for &xi in x {
            self.push(xi);
        }
    }

    pub fn execute(&self) -> Result<T> {
        let r = self.buffer.read();
        Ok(self.w0.iter().zip(r).map(|(&w, &x)| w.conj() * x).sum())
    }

    pub fn decim_execute(&mut self, x: &[T], k: usize) -> Result<T> {
        if k == 0 {
            return Err(Error::Config("down-sampling rate 'k' must be greater than 0".into()));
        }

        self.push(x[0]);
        let y = self.execute()?;
        self.push_block(&x[1..k]);
        Ok(y)
    }

    pub fn execute_block(&mut self, k: usize, x: &[T], y: &mut [T]) -> Result<()> {
        if k == 0 {
            return Err(Error::Config("down-sampling rate 'k' must be greater than 0".into()));
        }

        for (i, &xi) in x.iter().enumerate() {
            self.push(xi);
            let d_hat = self.execute()?;
            y[i] = d_hat;

            if (self.count + k - 1).is_multiple_of(k) {
                self.step_blind(d_hat);
            }
        }
        Ok(())
    }

    pub fn step(&mut self, d: T, d_hat: T) {
        if !self.buf_full {
            if self.count < self.h_len {
                return;
            } else {
                self.buf_full = true;
            }
        }

        let alpha = d - d_hat;
        let r = self.buffer.read();

        for i in 0..self.h_len {
            self.w1[i] = self.w0[i] + (self.mu * alpha.conj() * r[i]) / self.x2_sum;
        }

        self.w0.copy_from_slice(&self.w1);
    }

    pub fn step_blind(&mut self, d_hat: T) {
        let d = d_hat / d_hat.abs();
        self.step(d, d_hat)
    }

    fn update_sumsq(&mut self, x: T) {
        let x2_n = (x * x.conj()).re();
        self.x2.push(x2_n);
        let x2_0 = self.x2.read();
        self.x2_sum += x2_n - x2_0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{FirFilter, FirFilterShape, FirInterpolationFilter};
    use crate::math::{hamming, sincf};
    use crate::modem::modem::{Modem, ModulationScheme};
    use crate::random::randnf;
    use num_complex::Complex;
    use test_macro::autotest_annotate;

    #[allow(clippy::too_many_arguments)]
    fn testbench_eqlms(
        k: usize,
        m: usize,
        beta: f32,
        init: i32,
        p: usize,
        mu: f32,
        num_symbols: usize,
        update: i32,
        ms: ModulationScheme,
    ) {
        //let tol = 0.025f32; // error tolerance
        let mut mod_ = Modem::new(ms).unwrap();
        let mut interp =
            FirInterpolationFilter::<Complex<f32>, f32>::new_prototype(FirFilterShape::Arkaiser, k, m, beta, 0.0)
                .unwrap();

        // create fixed channel filter
        let h = [
            Complex::new(1.00, 0.00),
            Complex::new(0.00, -0.01),
            Complex::new(-0.11, 0.02),
            Complex::new(0.02, 0.01),
            Complex::new(-0.09, -0.04),
        ];
        let mut fchannel = FirFilter::<Complex<f32>>::new(&h).unwrap();

        // prototype low-pass filter
        let mut hp = vec![Complex::new(0.0, 0.0); 2 * k * p + 1];
        for i in 0..2 * k * p + 1 {
            hp[i] = Complex::new(
                sincf((i as f32) / (k as f32) - (p as f32)) * hamming(i, 2 * k * p + 1).unwrap() / (k as f32),
                0.0,
            );
        }

        // create and initialize equalizer
        let mut eq = match init {
            0 => Eqlms::<Complex<f32>>::new_rnyquist(FirFilterShape::Arkaiser, k, p, beta, 0.0),
            1 => Eqlms::<Complex<f32>>::new_lowpass(2 * k * p + 1, 0.5 / (k as f32)),
            2 => Eqlms::<Complex<f32>>::new(Some(&hp), 2 * k * p + 1),
            _ => Eqlms::<Complex<f32>>::new(None, 2 * k * p + 1),
        }
        .unwrap();
        eq.set_bw(mu).unwrap();

        // run equalization
        let mut buf = vec![Complex::new(0.0, 0.0); k];
        let mut buf_interp = vec![Complex::new(0.0, 0.0); k];
        let mut buf_sym = WDelay::create(m + p).unwrap();
        let mut rmse = 0.0f32;

        for i in 0..2 * num_symbols {
            // generate modulated input symbol
            let sym = mod_.random_symbol();
            let sym_in = mod_.modulate(sym).unwrap();
            buf_sym.push(sym_in);

            // interpolate
            interp.execute(sym_in, &mut buf_interp).unwrap();

            // apply channel filter (in place)
            fchannel.execute_block(&mut buf_interp, &mut buf).unwrap();

            // run through equalizing filter as decimator
            let sym_out = eq.decim_execute(&buf, k).unwrap();

            // skip first m+p symbols
            if i < m + p {
                continue;
            }

            // read input symbol, delayed by m+p symbols
            let sym_in = buf_sym.read();

            // update equalizer weights
            if i < num_symbols {
                match update {
                    0 => eq.step(sym_in, sym_out), // perfect knowledge
                    1 => eq.step_blind(sym_out),   // CM
                    2 => {
                        // decision-directed
                        let _index = mod_.demodulate(sym_out).unwrap();
                        let d_hat = mod_.get_demodulator_sample();
                        eq.step(d_hat, sym_out);
                    }
                    _ => {}
                }
                continue;
            }

            // observe error
            let error = (sym_in - sym_out).norm();
            rmse += error * error;
        }

        rmse = 10.0 * (rmse / (num_symbols as f32)).log10();
        println!("rmse : {:.3} dB", rmse);
        assert!(rmse < -20.0);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_00)]
    fn autotest_eqlms_00() {
        testbench_eqlms(2, 7, 0.3, 0, 7, 0.3, 800, 0, ModulationScheme::Qpsk);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_01)]
    fn autotest_eqlms_01() {
        testbench_eqlms(2, 7, 0.3, 0, 7, 0.3, 800, 1, ModulationScheme::Qpsk);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_02)]
    fn autotest_eqlms_02() {
        testbench_eqlms(2, 7, 0.3, 0, 7, 0.3, 800, 2, ModulationScheme::Qpsk);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_03)]
    fn autotest_eqlms_03() {
        testbench_eqlms(2, 7, 0.3, 0, 7, 0.3, 800, 0, ModulationScheme::Qam16);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_04)]
    fn autotest_eqlms_04() {
        testbench_eqlms(2, 7, 0.3, 1, 7, 0.3, 800, 0, ModulationScheme::Qam16);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_05)]
    fn autotest_eqlms_05() {
        testbench_eqlms(2, 7, 0.3, 2, 7, 0.3, 800, 0, ModulationScheme::Qam16);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_06)]
    fn autotest_eqlms_06() {
        testbench_eqlms(2, 7, 0.3, 3, 6, 0.3, 800, 0, ModulationScheme::Qam16);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_07)]
    fn autotest_eqlms_07() {
        testbench_eqlms(2, 9, 0.3, 0, 7, 0.3, 800, 0, ModulationScheme::Qpsk);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_08)]
    fn autotest_eqlms_08() {
        testbench_eqlms(2, 7, 0.2, 0, 9, 0.3, 800, 0, ModulationScheme::Qpsk);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_09)]
    fn autotest_eqlms_09() {
        testbench_eqlms(2, 7, 0.3, 0, 3, 0.3, 800, 0, ModulationScheme::Qpsk);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_10)]
    fn autotest_eqlms_10() {
        testbench_eqlms(2, 7, 0.3, 0, 7, 0.5, 800, 0, ModulationScheme::Arb64Vt);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_11)]
    fn autotest_eqlms_11() {
        testbench_eqlms(2, 7, 0.3, 0, 7, 0.1, 800, 0, ModulationScheme::Qpsk);
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_config)]
    fn autotest_eqlms_config() {
        // check that object returns None for invalid configurations
        assert!(Eqlms::<Complex<f32>>::new_rnyquist(FirFilterShape::Arkaiser, 0, 12, 0.3, 0.0).is_err());
        assert!(Eqlms::<Complex<f32>>::new_rnyquist(FirFilterShape::Arkaiser, 2, 0, 0.3, 0.0).is_err());
        assert!(Eqlms::<Complex<f32>>::new_rnyquist(FirFilterShape::Arkaiser, 2, 12, 2.0, 0.0).is_err());
        assert!(Eqlms::<Complex<f32>>::new_rnyquist(FirFilterShape::Arkaiser, 2, 12, 0.3, -2.0).is_err());

        assert!(Eqlms::<Complex<f32>>::new_lowpass(0, 0.1).is_err());
        assert!(Eqlms::<Complex<f32>>::new_lowpass(13, -0.1).is_err());

        // create proper object and test other interfaces
        let k = 2;
        let m = 3;
        let h_len = 2 * k * m + 1;
        let mut q = Eqlms::<Complex<f32>>::new(None, h_len).unwrap();
        // assert_eq!(q.print(), Ok(()));

        // test getting/setting properties
        assert_eq!(q.get_length(), h_len);
        let mu = 0.1;
        q.set_bw(mu).unwrap();
        assert_eq!(q.get_bw(), mu);
        assert!(q.set_bw(-1.0).is_err());

        // other configurations
        assert!(q.decim_execute(&[], 0).is_err());

        // test getting weights
        let h = q.get_coefficients();
        for (i, &coeff) in h.iter().enumerate() {
            assert_eq!(coeff, if i == k * m { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 0.0) });
        }
        let w = q.get_coefficients();
        for (i, &coeff) in w.iter().enumerate() {
            assert_eq!(coeff, if i == k * m { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 0.0) });
        }
    }

    #[test]
    #[autotest_annotate(autotest_eqlms_cccf_copy)]
    fn autotest_eqlms_cccf_copy() {
        // create initial object
        let mut q0 = Eqlms::<Complex<f32>>::new_lowpass(21, 0.12345).unwrap();
        q0.set_bw(0.1).unwrap();
        // q0.print().unwrap();

        // run random samples through object
        for _ in 0..120 {
            let x = Complex::new(randnf(), randnf());
            q0.push(x);
        }

        // copy object
        let mut q1 = q0.clone();

        // run random samples through both objects
        for _ in 0..120 {
            // push random sample in
            let x = Complex::new(randnf(), randnf());
            q0.push(x);
            q1.push(x);

            // compute output
            let y0 = q0.execute().unwrap();
            let y1 = q1.execute().unwrap();
            assert_eq!(y0, y1);

            // step equalization algorithm
            let v = Complex::new(randnf(), randnf());
            q0.step(v, y0);
            q1.step(v, y1);
        }

        // get and compare coefficients
        let w0 = q0.get_coefficients();
        let w1 = q1.get_coefficients();
        assert_eq!(w0, w1);
    }
}
