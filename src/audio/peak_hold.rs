use core::f32;

pub struct PeakHold {
    hold: usize,
    current: Vec<f32>,
    current_max: f32,
    previous: Vec<f32>,
}

impl PeakHold {
    pub fn new(hold: usize) -> Self {
        Self {
            hold,
            current: vec![],
            current_max: f32::MIN,
            previous: vec![f32::MIN; hold],
        }
    }

    pub fn reset(&mut self) {
        self.current.clear();
        self.current_max = f32::MIN;
        self.previous = vec![f32::MIN; self.hold];
    }

    pub fn execute(&mut self, x: f32) -> f32 {
        self.current.push(x);
        self.current_max = self.current_max.max(x);

        let previous_max = self.previous.pop().unwrap();

        let max = previous_max.max(self.current_max);

        if self.current.len() == self.hold {
            self.swap();
        }

        max
    }

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