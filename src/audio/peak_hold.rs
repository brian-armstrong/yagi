use core::f32;

/// Hold a recent peak value over a rolling sample history.
///
/// This helper is used by [`super::Limiter`] to sustain gain reduction after a
/// detected peak.
pub struct PeakHold {
    hold: usize,
    current: Vec<f32>,
    current_max: f32,
    previous: Vec<f32>,
}

impl PeakHold {
    /// Create a peak hold with `hold` samples per internal history block.
    ///
    /// # Panics
    ///
    /// `hold` must be nonzero. The constructor does not validate this and a
    /// later [`Self::execute`] will panic otherwise.
    pub fn new(hold: usize) -> Self {
        Self { hold, current: vec![], current_max: f32::MIN, previous: vec![f32::MIN; hold] }
    }

    /// Clear the peak history.
    pub fn reset(&mut self) {
        self.current.clear();
        self.current_max = f32::MIN;
        self.previous = vec![f32::MIN; self.hold];
    }

    /// Record `input` and return the held peak.
    pub fn execute(&mut self, input: f32) -> f32 {
        self.current.push(input);
        self.current_max = self.current_max.max(input);

        let previous_max = self.previous.pop().unwrap();

        let max = previous_max.max(self.current_max);

        if self.current.len() == self.hold {
            self.swap();
        }

        max
    }

    #[inline]
    fn swap(&mut self) {
        debug_assert!(self.previous.is_empty());
        let mut max = f32::MIN;
        for &x in self.current.iter().rev() {
            max = max.max(x);
            self.previous.push(max);
        }
        self.current.clear();
        self.current_max = f32::MIN;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_peak() {
        let hold = 3;
        let input = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let expected = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let mut peak_hold = PeakHold::new(hold);

        for (&x, &exp_y) in input.iter().zip(expected.iter()) {
            assert_eq!(peak_hold.execute(x), exp_y);
        }
    }

    #[test]
    fn test_double_peak() {
        let hold = 3;
        let input = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let expected = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0];

        let mut peak_hold = PeakHold::new(hold);

        for (&x, &exp_y) in input.iter().zip(expected.iter()) {
            assert_eq!(peak_hold.execute(x), exp_y);
        }
    }

    #[test]
    fn test_long_run() {
        let hold = 3;
        #[rustfmt::skip]
        let input = [
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            4.0, 3.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        #[rustfmt::skip]
        let expected = [
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0,
            4.0, 4.0, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 0.0, 0.0,
        ];

        let mut peak_hold = PeakHold::new(hold);

        for (&x, &exp_y) in input.iter().zip(expected.iter()) {
            assert_eq!(peak_hold.execute(x), exp_y);
        }
    }
}
