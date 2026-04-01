use super::peak_hold::PeakHold;
use crate::{buffer::WDelay, filter::FirFilter};

// inspired by the excellent post at https://signalsmith-audio.co.uk/writing/2022/limiter/ by Geraint Luff

pub struct Limiter {
    limit: f32,
    delay: WDelay<f32>,
    peak_hold: PeakHold,
    release_scale: f32,
    last_release: f32,
    filter: FirFilter<f32, f32>,
}

impl Limiter {
    pub fn new(limit: f32, attack: usize, hold: usize, release: usize) -> Self {
        // TODO confirm this filter bandwidth
        // let mut filter = FirFilter::new_kaiser(attack, 0.25, 60.0, 0.0).unwrap();
        let mut filter = FirFilter::new_rect(attack).unwrap();
        filter.set_scale(1.0 / attack as f32);
        for _ in 0..attack {
            filter.push(1.0);
        }
        Self {
            limit,
            delay: WDelay::create(attack).unwrap(),
            peak_hold: PeakHold::new(attack + hold),
            release_scale: 1.0 - (-1.0 / release as f32).exp(),
            last_release: 1.0,
            filter,
        }
    }

    pub fn reset(&mut self) {
        self.peak_hold.reset();
        self.filter.reset();
        self.delay.reset();
    }

    pub fn execute(&mut self, x: f32) -> f32 {
        self.delay.push(x);

        // hard limit the gain
        let magnitude = x.abs();
        let gain = if x <= self.limit { 1.0 } else { self.limit / magnitude };

        // peak/hold the inverse gain (we need a max-value)
        let peak_min_gain = 1.0 / self.peak_hold.execute(1.0 / gain);

        // exponential release
        let decay = (peak_min_gain - self.last_release) * self.release_scale;
        self.last_release += decay;
        self.last_release = self.last_release.min(peak_min_gain);

        // filter the last release value
        self.filter.push(self.last_release);
        let filtered_gain = self.filter.execute();

        // perform a delay
        let delayed_x = self.delay.read();

        // println!("x: {}, peak_min_gain: {}, last_release: {}, filtered_gain: {}, delayed_x: {}", x, peak_min_gain, self.last_release, filtered_gain, delayed_x);

        delayed_x * filtered_gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_relative_eq;

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

    // #[test]
    // fn test_single_limit() {
    //     let limit = 1.0;
    //     let attack = 3;
    //     let hold = 2;
    //     let release = 4;
    //     let tol = 1e-6;

    //     let x = [0.0, 0.4, 0.8, 0.9, 1.1, 0.9, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0];
    //     let expected = [0.0, 0.0, 0.0, 0.0, 0.4 / 1.1, 0.8 / 1.1, 0.9 / 1.1, 1.1 / 1.1, 0.9 / 1.1, 0.8 / 1.1, 0.4 / 1.1, 0.0, 0.0];

    //     let mut limiter = Limiter::new(limit, attack, hold, release);

    //     for (&x, &exp_y) in x.iter().zip(expected.iter()) {
    //         assert_relative_eq!(limiter.execute(x), exp_y, epsilon = tol);
    //     }
    // }
}