use core::f32;

/// Running maximum over the most recent `hold + 1` samples.
///
/// Each call records one sample and returns the largest of it and the
/// previous `hold` samples. A peak therefore persists in the output for
/// `hold` calls after it arrives. [`super::Limiter`] uses it to hold gain
/// reduction after a peak.
pub struct PeakHold {
    hold: usize,
    current: Vec<f32>,
    current_max: f32,
    previous: Vec<f32>,
}

impl PeakHold {
    /// Creates a peak hold over `hold + 1` samples.
    ///
    /// `hold` must be nonzero. Zero is accepted here, but the first
    /// [`Self::execute`] call then panics.
    pub fn new(hold: usize) -> Self {
        Self { hold, current: vec![], current_max: f32::MIN, previous: vec![f32::MIN; hold] }
    }

    /// Clears the sample history.
    ///
    /// Later calls return the maximum over the samples recorded since the
    /// reset, until `hold + 1` samples have arrived.
    pub fn reset(&mut self) {
        self.current.clear();
        self.current_max = f32::MIN;
        self.previous = vec![f32::MIN; self.hold];
    }

    /// Records `input` and returns the maximum over the most recent
    /// `hold + 1` samples.
    ///
    /// Before `hold + 1` samples have arrived, the maximum is over the
    /// samples recorded so far.
    ///
    /// # Panics
    ///
    /// Panics when the peak hold was created with `hold` zero.
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
