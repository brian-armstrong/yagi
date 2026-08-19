//
// tvmpch.rs
//
// Time-varying multi-path channel emulator.
//
// Each tap is a first-order autoregressive process driven by complex Gaussian
// innovations, which makes the tap a Rayleigh-fading path whose rate is set by
// the coherence time. Unlike `Channel::add_multipath`, whose taps are fixed for
// the life of the object, these taps move while the signal runs.
//

use num_complex::Complex32;

use crate::buffer::Window;
use crate::channel::noise::NoiseSource;
use crate::dotprod::DotProd;
use crate::error::{Error, Result};

/// Time-varying multi-path channel emulator
///
/// The larger the standard deviation, the more dramatic the frequency response
/// of the channel. The shorter the coherence time, the faster the channel
/// effects.
#[derive(Debug, Clone)]
pub struct Tvmpch {
    h: Vec<Complex32>,    // filter coefficients, time-reversed
    w: Window<Complex32>, // internal buffer

    std: f32,             // innovation scale
    alpha: f32,           // AR pole: how much of the previous tap is retained
    beta: f32,            // innovation weight (the normalized coherence time)

    noise: NoiseSource,   // Gaussian source driving the taps
}

impl Tvmpch {
    /// create time-varying multi-path channel emulator object
    ///
    /// # Arguments
    ///
    /// * `n` - number of coefficients, `n > 0`
    /// * `std` - standard deviation of coefficients, `std >= 0`
    /// * `tau` - normalized coherence time, `tau` in (0, 1]
    pub fn new(n: usize, std: f32, tau: f32) -> Result<Self> {
        // validate input
        if n < 1 {
            return Err(Error::Config(
                "tvmpch_create(), filter length must be greater than one".into(),
            ));
        }
        if std < 0.0 {
            return Err(Error::Config(
                "tvmpch_create(), standard deviation must be positive".into(),
            ));
        }
        if tau <= 0.0 || tau > 1.0 {
            return Err(Error::Config(
                "tvmpch_create(), coherence time must be in (0,1]".into(),
            ));
        }

        let beta = tau;
        let mut q = Self {
            h: vec![Complex32::new(0.0, 0.0); n],
            w: Window::new(n)?,
            std: 2.0 * std / beta.sqrt(),
            alpha: 1.0 - beta,
            beta,
            noise: NoiseSource::Global,
        };

        // time-reverse coefficients: the direct path sits at the end
        q.h[n - 1] = Complex32::new(1.0, 0.0);

        // reset filter state (clear buffer)
        q.reset();

        Ok(q)
    }

    /// create the emulator drawing its tap innovations from a seeded generator
    pub fn new_seeded(n: usize, std: f32, tau: f32, seed: u64) -> Result<Self> {
        let mut q = Self::new(n, std, tau)?;
        q.noise = NoiseSource::seeded(seed);
        Ok(q)
    }

    /// reset internal state of the filter object
    pub fn reset(&mut self) {
        self.w.reset();
    }

    /// push sample into the filter object's internal buffer
    pub fn push(&mut self, x: Complex32) {
        // update coefficients
        let n = self.h.len();
        for i in 0..n - 1 {
            let v = Complex32::new(self.noise.randnf(), self.noise.randnf());
            self.h[i] = self.h[i] * self.alpha
                + v * self.beta * self.std * std::f32::consts::FRAC_1_SQRT_2;
        }

        // push sample into window buffer
        self.w.push(x);
    }

    /// compute output sample
    pub fn execute(&self) -> Complex32 {
        self.w.read().dotprod(&self.h)
    }

    /// execute filter on one sample, equivalent to `push()` then `execute()`
    pub fn execute_one(&mut self, x: Complex32) -> Complex32 {
        self.push(x);
        self.execute()
    }

    /// execute the filter on a block of input samples
    ///
    /// # Arguments
    ///
    /// * `x` - input array
    /// * `y` - output array, same length as `x`
    pub fn execute_block(&mut self, x: &[Complex32], y: &mut [Complex32]) -> Result<()> {
        if x.len() != y.len() {
            return Err(Error::Config(
                "tvmpch_execute_block(), input and output lengths must match".into(),
            ));
        }
        for (x_i, y_i) in x.iter().zip(y.iter_mut()) {
            *y_i = self.execute_one(*x_i);
        }
        Ok(())
    }

    /// get filter length
    pub fn len(&self) -> usize {
        self.h.len()
    }

    /// whether the filter has no coefficients
    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }

    /// get the current channel coefficients, in impulse-response order
    pub fn get_coefficients(&self) -> Vec<Complex32> {
        self.h.iter().rev().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_tvmpch_rejects_invalid_config() {
        assert!(Tvmpch::new(0, 0.1, 0.05).is_err());
        assert!(Tvmpch::new(31, -0.1, 0.05).is_err());
        assert!(Tvmpch::new(31, 0.1, -0.01).is_err());
        assert!(Tvmpch::new(31, 0.1, 1.01).is_err());
        assert!(Tvmpch::new(31, 0.1, 0.0).is_err());

        assert!(Tvmpch::new(31, 0.1, 0.05).is_ok());
        assert!(Tvmpch::new(1, 0.0, 1.0).is_ok());
    }

    #[test]
    fn test_tvmpch_single_tap_is_passthrough() {
        // with one coefficient there is nothing to fade. the lone tap is the
        // pinned direct path, so the channel is transparent
        let mut q = Tvmpch::new_seeded(1, 1.0, 0.5, 1).unwrap();
        let x: Vec<Complex32> = (0..32)
            .map(|i| Complex32::new(i as f32, -(i as f32)))
            .collect();
        let mut y = vec![Complex32::new(0.0, 0.0); x.len()];
        q.execute_block(&x, &mut y).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn test_tvmpch_zero_std_is_pure_delay() {
        // std=0 kills the innovations, so the taps stay at their initial state.
        let n = 8;
        let mut q = Tvmpch::new_seeded(n, 0.0, 0.1, 2).unwrap();
        let x: Vec<Complex32> = (0..64)
            .map(|i| Complex32::new((0.3 * i as f32).cos(), (0.2 * i as f32).sin()))
            .collect();
        let mut y = vec![Complex32::new(0.0, 0.0); x.len()];
        q.execute_block(&x, &mut y).unwrap();

        for (a, b) in x.iter().zip(y.iter()) {
            assert_abs_diff_eq!(a.re, b.re, epsilon = 1e-6);
            assert_abs_diff_eq!(a.im, b.im, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_tvmpch_direct_path_never_fades() {
        // the final stored tap is the direct path and must stay exactly 1
        let mut q = Tvmpch::new_seeded(16, 2.0, 0.5, 3).unwrap();
        for i in 0..500 {
            q.execute_one(Complex32::new(i as f32, 0.0));
        }
        let h = q.get_coefficients();
        assert_eq!(h[0], Complex32::new(1.0, 0.0));
        // and the fading taps have actually moved
        assert!(
            h[1..].iter().any(|v| v.norm() > 1e-3),
            "expected the echo taps to have faded in"
        );
    }

    #[test]
    fn test_tvmpch_seeded_is_reproducible() {
        let x: Vec<Complex32> = (0..120)
            .map(|i| Complex32::new((0.1 * i as f32).cos(), (0.05 * i as f32).sin()))
            .collect();

        let mut q0 = Tvmpch::new_seeded(31, 0.1, 0.05, 9).unwrap();
        let mut q1 = Tvmpch::new_seeded(31, 0.1, 0.05, 9).unwrap();
        let mut y0 = vec![Complex32::new(0.0, 0.0); x.len()];
        let mut y1 = vec![Complex32::new(0.0, 0.0); x.len()];
        q0.execute_block(&x, &mut y0).unwrap();
        q1.execute_block(&x, &mut y1).unwrap();
        assert_eq!(y0, y1);

        // a clone must continue in lockstep with its source
        let mut q2 = q0.clone();
        for i in 0..120 {
            let xi = Complex32::new(i as f32, -1.0);
            assert_eq!(q0.execute_one(xi), q2.execute_one(xi));
        }

        // a different seed must diverge
        let mut q3 = Tvmpch::new_seeded(31, 0.1, 0.05, 10).unwrap();
        let mut y3 = vec![Complex32::new(0.0, 0.0); x.len()];
        q3.execute_block(&x, &mut y3).unwrap();
        assert_ne!(y0, y3);
    }
}
