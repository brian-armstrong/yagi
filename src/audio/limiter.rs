use super::peak_hold::PeakHold;
use crate::buffer::{WDelay, Window};

// inspired by the excellent post at https://signalsmith-audio.co.uk/writing/2022/limiter/ by Geraint Luff

pub struct Limiter {
    limit: f32,
    delay: WDelay<f32>,
    peak_hold: PeakHold,
    release_scale: f32,
    last_release: f32,
    release_sum: f32,
    release_sum_age: usize,
    smoothing_scale: f32,
    smoothing_window: Window<f32>,
}

impl Limiter {
    pub fn new(limit: f32, attack: usize, hold: usize, release: usize) -> Self {
        let mut smoothing_window = Window::new(attack + 1).unwrap();
        for _ in 0..smoothing_window.len() {
            smoothing_window.push(1.0);
        }
        Self {
            limit,
            delay: WDelay::create(attack).unwrap(),
            peak_hold: PeakHold::new(attack + hold),
            release_scale: 1.0 - (-1.0 / release as f32).exp(),
            last_release: 1.0,
            release_sum: smoothing_window.len() as f32,
            release_sum_age: 0,
            smoothing_scale: 1.0 / smoothing_window.len() as f32,
            smoothing_window,
        }
    }

    pub fn reset(&mut self) {
        self.peak_hold.reset();
        self.delay.reset();
        self.smoothing_window.reset();
        self.last_release = 1.0;
        self.release_sum = self.smoothing_window.len() as f32;
        self.release_sum_age = 0;
        for _ in 0..self.smoothing_window.len() {
            self.smoothing_window.push(1.0);
        }
    }

    pub fn execute(&mut self, x: f32) -> f32 {
        self.delay.push(x);

        // hard limit the gain
        let magnitude = x.abs();
        let hold_magnitude = self.peak_hold.execute(magnitude);
        let peak_min_gain = if hold_magnitude <= self.limit {
            1.0
        } else {
            self.limit / hold_magnitude
        };

        // exponential release
        let decay = (peak_min_gain - self.last_release) * self.release_scale;
        self.last_release += decay;
        self.last_release = self.last_release.min(peak_min_gain);

        // filter the last release value
        self.release_sum_age += 1;
        if self.release_sum_age == self.smoothing_window.len() {
            self.release_sum = self.smoothing_window.read().iter().sum();
            self.release_sum_age = 0;
        }

        let oldest_last_release = self.smoothing_window.read()[0];
        self.smoothing_window.push(self.last_release);
        self.release_sum += self.last_release - oldest_last_release;

        let filtered_gain = self.release_sum * self.smoothing_scale;

        // perform a delay
        let delayed_x = self.delay.read();

        // println!("x: {}, peak_min_gain: {}, last_release: {}, filtered_gain: {}, delayed_x: {}", x, peak_min_gain, self.last_release, filtered_gain, delayed_x);

        delayed_x * filtered_gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_abs_diff_eq;

    #[test]
    fn test_no_limit() {
        let limit = 1.0;
        let attack = 3;
        let hold = 2;
        let release = 4;

        let x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
        let expected = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.0];

        let mut limiter = Limiter::new(limit, attack, hold, release);

        for (&x, &exp_y) in x.iter().zip(expected.iter()) {
            assert_eq!(limiter.execute(x), exp_y);
        }
    }

    #[test]
    fn test_positive_limit() {
        let limit = 1.0;
        let attack = 3;
        let hold = 2;
        let release = 4;
        let x = [0.0, 0.1, 1.1, 0.9, 1.2, 0.4, 0.2, 2.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut limiter = Limiter::new(limit, attack, hold, release);
        for &x in x.iter() {
            assert!(limiter.execute(x).abs() < limit + 1e-5);
        }
    }

    #[test]
    fn test_negative_limit() {
        let limit = 1.0;
        let attack = 3;
        let hold = 2;
        let release = 4;
        let x = [0.0, -0.1, -1.1, -0.9, -1.2, -0.4, -0.2, -2.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut limiter = Limiter::new(limit, attack, hold, release);
        for &x in x.iter() {
            assert!(limiter.execute(x).abs() < limit + 1e-5);
        }
    }

    #[test]
    fn test_mixed_limit() {
        let limit = 1.0;
        let attack = 3;
        let hold = 2;
        let release = 4;
        let x = [0.0, -0.1, -1.1, 0.9, 1.2, 0.4, -0.2, -2.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut limiter = Limiter::new(limit, attack, hold, release);
        for &x in x.iter() {
            assert!(limiter.execute(x).abs() < limit + 1e-5);
        }
    }
}
