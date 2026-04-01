// Fft module
// Current state:
// - Primary Fft struct ready to use (some autotests still missing)
// - asgram/spgram/spwaterfall still missing

extern crate rustfft;

pub mod spgram;

use std::sync::Arc;
use num_complex::Complex;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Forward,
    Backward,
}

impl From<Direction> for rustfft::FftDirection {
    fn from(direction: Direction) -> Self {
        match direction {
            Direction::Forward => rustfft::FftDirection::Forward,
            Direction::Backward => rustfft::FftDirection::Inverse,
        }
    }
}

// TODO probably fold this into a larger trait
pub trait FftNum: rustfft::FftNum + std::default::Default {}

impl<T: rustfft::FftNum + std::default::Default> FftNum for T {}

#[derive(Clone)]
pub struct Fft<T> {
    fft: Arc<dyn rustfft::Fft<T>>,
}

impl<T: FftNum> Fft<T> {
    pub fn new(n: usize, direction: Direction) -> Self {
        let mut planner = rustfft::FftPlanner::new();
        let fft = planner.plan_fft(n, direction.into());
        Self { fft }
    }

    pub fn run(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        output.copy_from_slice(input);
        self.fft.process(output);
    }

    pub fn shift(&self, input: &mut [Complex<T>], n: usize) {
        let n2 = if n % 2 == 0 { n / 2 } else { (n - 1) / 2 };
        for i in 0..n2 {
            let temp = input[i];
            input[i] = input[i + n2];
            input[i + n2] = temp;
        }
    }
}

impl<T: FftNum> std::fmt::Debug for Fft<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fft {{ n: {} }}", self.fft.len())
    }
}

pub fn fft_run<T: FftNum>(input: &[Complex<T>], output: &mut [Complex<T>], direction: Direction) {
    let fft = Fft::new(input.len(), direction);
    fft.run(input, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;

    #[test]
    #[autotest_annotate(autotest_fft_shift_4)]
    fn test_shift_4() {
        let mut input = vec![
            Complex::new(0, 0),
            Complex::new(1, 1),
            Complex::new(2, 2),
            Complex::new(3, 3),
        ];
        let expected = vec![
            Complex::new(2, 2),
            Complex::new(3, 3),
            Complex::new(0, 0),
            Complex::new(1, 1),
        ];
        let fft = Fft::new(4, Direction::Forward);
        fft.shift(&mut input, 4);
        assert_eq!(input, expected);
    }

    #[test]
    #[autotest_annotate(autotest_fft_shift_8)]
    fn test_shift_8() {
        let mut input = vec![
            Complex::new(0, 0),
            Complex::new(1, 1),
            Complex::new(2, 2),
            Complex::new(3, 3),
            Complex::new(4, 4),
            Complex::new(5, 5),
            Complex::new(6, 6),
            Complex::new(7, 7),
        ];
        let expected = vec![
            Complex::new(4, 4),
            Complex::new(5, 5),
            Complex::new(6, 6),
            Complex::new(7, 7),
            Complex::new(0, 0),
            Complex::new(1, 1),
            Complex::new(2, 2),
            Complex::new(3, 3),
        ];
        let fft = Fft::new(8, Direction::Forward);
        fft.shift(&mut input, 8);
        assert_eq!(input, expected);
    }

    fn fft_test_runner(x: &[Complex<f32>], test: &[Complex<f32>], n: usize) {
        let tol = 2e-4;

        let mut y = vec![Complex::<f32>::new(0.0, 0.0); n];
        let mut z = vec![Complex::<f32>::new(0.0, 0.0); n];

        // compute FFT
        let fft_forward = Fft::new(n, Direction::Forward);
        fft_forward.run(x, &mut y);

        // compute IFFT
        let fft_backward = Fft::new(n, Direction::Backward);
        fft_backward.run(&y, &mut z);

        // normalize inverse
        for z_i in z.iter_mut() {
            *z_i /= n as f32;
        }

        // validate results
        for i in 0..n {
            let fft_error = (y[i] - test[i]).norm();
            let ifft_error = (x[i] - z[i]).norm();
            assert_relative_eq!(fft_error, 0.0, epsilon = tol);
            assert_relative_eq!(ifft_error, 0.0, epsilon = tol);
        }
    }

    #[rustfmt::skip]
    include!("test_data.rs");

    #[test]
    #[autotest_annotate(autotest_fft_2)]
    fn test_fft_2() {
        fft_test_runner(&FFT_TEST_X2, &FFT_TEST_Y2, 2);
    }

    #[test]
    #[autotest_annotate(autotest_fft_3)]
    fn test_fft_3() {
        fft_test_runner(&FFT_TEST_X3, &FFT_TEST_Y3, 3);
    }

    #[test]
    #[autotest_annotate(autotest_fft_4)]
    fn test_fft_4() {
        fft_test_runner(&FFT_TEST_X4, &FFT_TEST_Y4, 4);
    }

    #[test]
    #[autotest_annotate(autotest_fft_5)]
    fn test_fft_5() {
        fft_test_runner(&FFT_TEST_X5, &FFT_TEST_Y5, 5);
    }

    #[test]
    #[autotest_annotate(autotest_fft_6)]
    fn test_fft_6() {
        fft_test_runner(&FFT_TEST_X6, &FFT_TEST_Y6, 6);
    }

    #[test]
    #[autotest_annotate(autotest_fft_7)]
    fn test_fft_7() {
        fft_test_runner(&FFT_TEST_X7, &FFT_TEST_Y7, 7);
    }

    #[test]
    #[autotest_annotate(autotest_fft_8)]
    fn test_fft_8() {
        fft_test_runner(&FFT_TEST_X8, &FFT_TEST_Y8, 8);
    }

    #[test]
    #[autotest_annotate(autotest_fft_9)]
    fn test_fft_9() {
        fft_test_runner(&FFT_TEST_X9, &FFT_TEST_Y9, 9);
    }

    #[test]
    #[autotest_annotate(autotest_fft_10)]
    fn test_fft_10() {
        fft_test_runner(&FFT_TEST_X10, &FFT_TEST_Y10, 10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_16)]
    fn test_fft_16() {
        fft_test_runner(&FFT_TEST_X16, &FFT_TEST_Y16, 16);
    }

    #[test]
    #[autotest_annotate(autotest_fft_17)]
    fn test_fft_17() {
        fft_test_runner(&FFT_TEST_X17, &FFT_TEST_Y17, 17);
    }

    #[test]
    #[autotest_annotate(autotest_fft_20)]
    fn test_fft_20() {
        fft_test_runner(&FFT_TEST_X20, &FFT_TEST_Y20, 20);
    }

    #[test]
    #[autotest_annotate(autotest_fft_21)]
    fn test_fft_21() {
        fft_test_runner(&FFT_TEST_X21, &FFT_TEST_Y21, 21);
    }

    #[test]
    #[autotest_annotate(autotest_fft_22)]
    fn test_fft_22() {
        fft_test_runner(&FFT_TEST_X22, &FFT_TEST_Y22, 22);
    }

    #[test]
    #[autotest_annotate(autotest_fft_24)]
    fn test_fft_24() {
        fft_test_runner(&FFT_TEST_X24, &FFT_TEST_Y24, 24);
    }

    #[test]
    #[autotest_annotate(autotest_fft_26)]
    fn test_fft_26() {
        fft_test_runner(&FFT_TEST_X26, &FFT_TEST_Y26, 26);
    }

    #[test]
    #[autotest_annotate(autotest_fft_30)]
    fn test_fft_30() {
        fft_test_runner(&FFT_TEST_X30, &FFT_TEST_Y30, 30);
    }

    #[test]
    #[autotest_annotate(autotest_fft_32)]
    fn test_fft_32() {
        fft_test_runner(&FFT_TEST_X32, &FFT_TEST_Y32, 32);
    }

    #[test]
    #[autotest_annotate(autotest_fft_35)]
    fn test_fft_35() {
        fft_test_runner(&FFT_TEST_X35, &FFT_TEST_Y35, 35);
    }

    #[test]
    #[autotest_annotate(autotest_fft_36)]
    fn test_fft_36() {
        fft_test_runner(&FFT_TEST_X36, &FFT_TEST_Y36, 36);
    }

    #[test]
    #[autotest_annotate(autotest_fft_43)]
    fn test_fft_43() {
        fft_test_runner(&FFT_TEST_X43, &FFT_TEST_Y43, 43);
    }

    #[test]
    #[autotest_annotate(autotest_fft_48)]
    fn test_fft_48() {
        fft_test_runner(&FFT_TEST_X48, &FFT_TEST_Y48, 48);
    }

    #[test]
    #[autotest_annotate(autotest_fft_63)]
    fn test_fft_63() {
        fft_test_runner(&FFT_TEST_X63, &FFT_TEST_Y63, 63);
    }

    #[test]
    #[autotest_annotate(autotest_fft_64)]
    fn test_fft_64() {
        fft_test_runner(&FFT_TEST_X64, &FFT_TEST_Y64, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fft_79)]
    fn test_fft_79() {
        fft_test_runner(&FFT_TEST_X79, &FFT_TEST_Y79, 79);
    }

    #[test]
    #[autotest_annotate(autotest_fft_92)]
    fn test_fft_92() {
        fft_test_runner(&FFT_TEST_X92, &FFT_TEST_Y92, 92);
    }

    #[test]
    #[autotest_annotate(autotest_fft_96)]
    fn test_fft_96() {
        fft_test_runner(&FFT_TEST_X96, &FFT_TEST_Y96, 96);
    }

    #[test]
    #[autotest_annotate(autotest_fft_120)]
    fn test_fft_120() {
        fft_test_runner(&FFT_TEST_X120, &FFT_TEST_Y120, 120);
    }

    #[test]
    #[autotest_annotate(autotest_fft_130)]
    fn test_fft_130() {
        fft_test_runner(&FFT_TEST_X130, &FFT_TEST_Y130, 130);
    }

    #[test]
    #[autotest_annotate(autotest_fft_157)]
    fn test_fft_157() {
        fft_test_runner(&FFT_TEST_X157, &FFT_TEST_Y157, 157);
    }

    #[test]
    #[autotest_annotate(autotest_fft_192)]
    fn test_fft_192() {
        fft_test_runner(&FFT_TEST_X192, &FFT_TEST_Y192, 192);
    }

    #[test]
    #[autotest_annotate(autotest_fft_317)]
    fn test_fft_317() {
        fft_test_runner(&FFT_TEST_X317, &FFT_TEST_Y317, 317);
    }

    #[test]
    #[autotest_annotate(autotest_fft_509)]
    fn test_fft_509() {
        fft_test_runner(&FFT_TEST_X509, &FFT_TEST_Y509, 509);
    }
}