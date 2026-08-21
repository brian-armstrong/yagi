//
// auto-correlator (delay cross-correlation)
//

use crate::buffer::Window;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};
use num_complex::ComplexFloat;

/// Computes auto-correlation with a fixed lag on input signals
#[derive(Clone, Debug)]
pub struct AutoCorr<T> {
    window_size: usize,
    delay: usize,

    w: Window<T>,      // input buffer
    wdelay: Window<T>, // input buffer with delay

    we2: Vec<f32>, // energy buffer
    e2_sum: f32,   // running sum of energy
    ie2: usize,    // read index
}

impl<T> AutoCorr<T>
where
    T: Copy + Default + ComplexFloat<Real = f32>,
    [T]: DotProd<T, Output = T>,
{
    /// create auto-correlator object with a particular window length and delay
    ///
    ///  window_size    : size of the correlator window
    ///  delay          : correlator delay [samples]
    pub fn new(window_size: usize, delay: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(Error::Config("autocorr window size must be greater than zero".into()));
        }

        let mut q = Self {
            window_size,
            delay,

            // create window objects
            w: Window::new(window_size)?,
            wdelay: Window::new(window_size + delay)?,

            // allocate array for squared energy buffer
            we2: vec![0.0; window_size],
            e2_sum: 0.0,
            ie2: 0,
        };

        // clear object
        q.reset();

        Ok(q)
    }

    /// reset auto-correlator object's internals
    pub fn reset(&mut self) {
        // clear/reset internal window buffers
        self.w.reset();
        self.wdelay.reset();

        // reset internal squared energy buffer
        self.e2_sum = 0.0;
        self.we2.fill(0.0);
        self.ie2 = 0; // reset read index to zero
    }

    /// size of the correlator window
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// correlator delay [samples]
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// push sample into auto-correlator object
    pub fn push(&mut self, x: T) {
        // push input sample into buffers
        self.w.push(x); // non-delayed buffer
        self.wdelay.push(x.conj()); // delayed buffer

        // push |x|^2 into buffer at appropriate location
        let e2 = (x * x.conj()).re();
        self.e2_sum -= self.we2[self.ie2];
        self.e2_sum += e2;
        self.we2[self.ie2] = e2;
        self.ie2 = (self.ie2 + 1) % self.window_size;
    }

    /// write block of samples to auto-correlator object
    ///
    ///  x      :   input array
    pub fn write(&mut self, x: &[T]) {
        for &xi in x {
            self.push(xi);
        }
    }

    /// compute single auto-correlation output
    pub fn execute(&self) -> T {
        // read buffers
        let rw = self.w.read();
        let rwdelay = self.wdelay.read();

        // execute vector dot product on arrays (both oldest-first)
        rw.dotprod(&rwdelay[..self.window_size])
    }

    /// compute auto-correlation on block of samples
    ///
    ///  x      :   input array
    ///  rxx    :   output array
    pub fn execute_block(&mut self, x: &[T], rxx: &mut [T]) -> Result<()> {
        if rxx.len() < x.len() {
            return Err(Error::Range("autocorr output array too small".into()));
        }

        for (i, &xi) in x.iter().enumerate() {
            // push input sample into auto-correlator
            self.push(xi);

            // compute output
            rxx[i] = self.execute();
        }
        Ok(())
    }

    /// return sum of squares of buffered samples
    pub fn get_energy(&self) -> f32 {
        // value is already computed; simply return value
        self.e2_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::MSequence;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex32;

    fn autocorr_ref(x: &[Complex32], n: usize, window_size: usize, delay: usize) -> Complex32 {
        let mut sum = Complex32::new(0.0, 0.0);
        for i in 0..window_size {
            // the object's newest sample after n+1 pushes is x[n]
            let a = (n + 1).checked_sub(window_size - i);
            let b = a.and_then(|a| a.checked_sub(delay));
            if let (Some(a), Some(b)) = (a, b) {
                sum += x[a] * x[b].conj();
            }
        }
        sum
    }

    #[test]
    fn test_autocorr_cccf_matches_direct() {
        let window_size = 16;
        let delay = 7;

        // deterministic pseudo-random QPSK-ish sequence
        let mut ms = MSequence::create_default(9).unwrap();
        let x: Vec<Complex32> = (0..96)
            .map(|_| {
                let s = ms.generate_symbol(2);
                let re = if s & 1 != 0 { 1.0 } else { -1.0 };
                let im = if s & 2 != 0 { 1.0 } else { -1.0 };
                Complex32::new(re, im) / 2.0f32.sqrt()
            })
            .collect();

        let mut q = AutoCorr::new(window_size, delay).unwrap();
        for (n, &xn) in x.iter().enumerate() {
            q.push(xn);
            let rxx = q.execute();
            let expected = autocorr_ref(&x, n, window_size, delay);
            assert_abs_diff_eq!(rxx.re, expected.re, epsilon = 1e-4);
            assert_abs_diff_eq!(rxx.im, expected.im, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_autocorr_rrrf_matches_direct() {
        let window_size = 12;
        let delay = 5;

        let mut ms = MSequence::create_default(7).unwrap();
        let xr: Vec<f32> = (0..64).map(|_| if ms.advance() != 0 { 1.0 } else { -1.0 }).collect();
        let xc: Vec<Complex32> = xr.iter().map(|&v| Complex32::new(v, 0.0)).collect();

        let mut q = AutoCorr::new(window_size, delay).unwrap();
        for (n, &xn) in xr.iter().enumerate() {
            q.push(xn);
            let rxx = q.execute();
            let expected = autocorr_ref(&xc, n, window_size, delay).re;
            assert_abs_diff_eq!(rxx, expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_autocorr_periodic_sequence_peaks() {
        let sequence_len = 32;
        let num_reps = 8;
        let window_size = 64;

        let mut ms = MSequence::create_default(9).unwrap();
        let sequence: Vec<Complex32> = (0..sequence_len)
            .map(|_| {
                let s = ms.generate_symbol(2);
                let re = if s & 1 != 0 { 1.0 } else { -1.0 };
                let im = if s & 2 != 0 { 1.0 } else { -1.0 };
                Complex32::new(re, im) / 2.0f32.sqrt()
            })
            .collect();

        // delay equal to the repetition period
        let mut q = AutoCorr::new(window_size, sequence_len).unwrap();

        // write the sequence repeatedly, then pad with zeros
        let mut x: Vec<Complex32> = Vec::new();
        for _ in 0..num_reps {
            x.extend_from_slice(&sequence);
        }
        x.resize(sequence_len * (num_reps + 2), Complex32::new(0.0, 0.0));

        let mut rxx = vec![Complex32::new(0.0, 0.0); x.len()];
        q.execute_block(&x, &mut rxx).unwrap();

        // each sample has unit magnitude, so a full window of signal has energy
        // equal to window_size
        let n_full = window_size + sequence_len - 1;
        let n_last = sequence_len * num_reps - 1;
        for i in n_full..=n_last {
            assert_abs_diff_eq!(rxx[i].re, window_size as f32, epsilon = 1e-3);
            assert_abs_diff_eq!(rxx[i].im, 0.0, epsilon = 1e-3);
        }

        // and the peak over the whole run is no larger than that
        let peak = rxx.iter().map(|v| v.norm()).fold(0.0f32, f32::max);
        assert_abs_diff_eq!(peak, window_size as f32, epsilon = 1e-3);

        // long after the signal ends both windows are zero
        assert_abs_diff_eq!(rxx[x.len() - 1].norm(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_autocorr_get_energy() {
        let window_size = 8;
        let mut q = AutoCorr::new(window_size, 3).unwrap();

        // empty buffer has no energy
        assert_abs_diff_eq!(q.get_energy(), 0.0, epsilon = 1e-6);

        // fill with unit-magnitude samples: energy ramps to window_size
        for i in 1..=window_size {
            q.push(Complex32::new(0.0, 1.0));
            assert_abs_diff_eq!(q.get_energy(), i as f32, epsilon = 1e-5);
        }

        // saturated: the running sum drops the oldest sample
        for _ in 0..window_size {
            q.push(Complex32::new(1.0, 0.0));
            assert_abs_diff_eq!(q.get_energy(), window_size as f32, epsilon = 1e-5);
        }

        // pushing zeros drains it back down
        for i in (0..window_size).rev() {
            q.push(Complex32::new(0.0, 0.0));
            assert_abs_diff_eq!(q.get_energy(), i as f32, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_autocorr_config() {
        assert!(AutoCorr::<Complex32>::new(0, 4).is_err());
        assert!(AutoCorr::<f32>::new(0, 0).is_err());

        let q = AutoCorr::<Complex32>::new(16, 4).unwrap();
        assert_eq!(q.window_size(), 16);
        assert_eq!(q.delay(), 4);

        // output array must be at least as long as the input
        let mut q = AutoCorr::<Complex32>::new(8, 2).unwrap();
        let x = [Complex32::new(1.0, 0.0); 4];
        let mut rxx = [Complex32::new(0.0, 0.0); 3];
        assert!(q.execute_block(&x, &mut rxx).is_err());
    }
}
