use crate::error::{Error, Result};
use std::ops::Range;

/// Fixed-length history of the most recent samples.
///
/// Each [`Self::push`] appends one sample and drops the oldest.
/// [`Self::read`] borrows the whole history as one contiguous slice of
/// [`Self::len`] samples, oldest first. A new window holds `T::default()`
/// values.
///
/// The derived `PartialEq` compares internal storage, not just the history.
/// Two windows with the same [`Self::read`] contents can compare unequal, so
/// compare their `read()` slices instead.
#[derive(Debug, Clone, PartialEq)]
pub struct Window<T> {
    v: Vec<T>,
    len: usize,
    n: usize,
    mask: usize,
    read_index: usize,
}

impl<T: Default + Clone + Copy> Window<T> {
    /// Creates a window that holds `capacity` samples.
    ///
    /// # Errors
    ///
    /// Returns an error when `capacity` is zero.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Config("window size must be greater than zero".to_string()));
        }

        let m = crate::utility::msb_index(capacity as u32) as usize;
        let n_pow2 = 1 << m;
        let mask = n_pow2 - 1;
        let num_allocated = n_pow2 + capacity - 1;

        let mut window = Window { v: vec![T::default(); num_allocated], len: capacity, n: n_pow2, mask, read_index: 0 };

        window.reset();
        Ok(window)
    }

    /// Changes the history length to `n` samples, keeping the newest ones.
    ///
    /// Shrinking drops the oldest samples. Growing adds `T::default()` values
    /// before the existing history.
    ///
    /// # Errors
    ///
    /// Returns an error when `n` is zero. The window is unchanged on error.
    pub fn resize(&mut self, n: usize) -> Result<()> {
        if n == self.len {
            return Ok(());
        }

        let mut new_window = Window::new(n)?;

        if n > self.len {
            // New buffer is larger; push zeros, then old values
            for _ in 0..(n - self.len) {
                new_window.push(T::default());
            }
            for i in 0..self.len {
                new_window.push(self.index(i)?);
            }
        } else {
            // New buffer is shorter; push latest old values
            for i in (self.len - n)..self.len {
                new_window.push(self.index(i)?);
            }
        }

        *self = new_window;
        Ok(())
    }

    /// Fills the history with `T::default()` values.
    pub fn reset(&mut self) {
        self.read_index = 0;
        self.v.fill(T::default());
    }

    /// Returns the history, oldest sample first.
    pub fn read(&self) -> &[T] {
        &self.v[self.read_index..self.read_index + self.len]
    }

    /// Returns the number of samples in the history.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `false`, because a window always holds [`Self::len`] samples.
    pub fn is_empty(&self) -> bool {
        // guaranteed to be nonzero by new
        false
    }

    /// Returns the sample at `index`, where zero is the oldest.
    ///
    /// # Errors
    ///
    /// Returns an error when `index` is `len()` or more.
    pub fn index(&self, index: usize) -> Result<T> {
        if index >= self.len {
            return Err(Error::Range("index value out of range".to_string()));
        }
        Ok(self.v[self.read_index + index])
    }

    /// Replaces the sample at `index`, where zero is the oldest.
    ///
    /// `index` must be less than [`Self::len`]. Unlike [`Self::index`], this
    /// is not checked. A larger index either panics or writes to storage
    /// outside the history, in which case the value is silently lost.
    pub fn set(&mut self, index: usize, value: T) {
        self.v[self.read_index + index] = value;
    }

    /// Appends one sample and drops the oldest.
    pub fn push(&mut self, value: T) {
        self.read_index = (self.read_index + 1) & self.mask;

        if self.read_index == 0 {
            self.v.copy_within(self.n..self.n + self.len - 1, 0);
        }

        self.v[self.read_index + self.len - 1] = value;
    }

    /// Appends each sample of `values` in order.
    pub fn write(&mut self, values: &[T]) {
        for value in values {
            self.push(*value);
        }
    }

    /// Pushes each sample of `input` and visits the history after each push.
    ///
    /// Calls `visit(i, history)` once per input sample, in order, where
    /// `history` is what [`Self::read`] would return after pushing
    /// `input[i]`. The window ends with the same [`Self::read`] history as
    /// after pushing every sample.
    ///
    /// If `visit` panics, the window may keep only part of `input`. When
    /// `input` holds at least [`Self::len`] samples, everything past the
    /// first `len - 1` is visited before any of it is pushed, so a panic
    /// there leaves only those first `len - 1` samples.
    #[inline]
    pub fn execute_block<F>(&mut self, input: &[T], mut visit: F)
    where
        F: FnMut(usize, &[T]),
    {
        let len = self.len;
        self.execute_block_contiguous(input, |indices, samples| {
            for (i, history) in indices.zip(samples.windows(len)) {
                visit(i, history);
            }
        });
    }

    /// Pushes each sample of `input` and visits the histories in contiguous
    /// runs.
    ///
    /// This is [`Self::execute_block`] with fewer copies. Each call
    /// `visit(indices, samples)` covers the input positions in `indices`.
    /// `samples` holds `indices.len() + self.len() - 1` values, and
    /// `samples.windows(self.len())` yields the history for each position in
    /// order. `visit` may be called more than once. The window ends with the
    /// same [`Self::read`] history as after pushing every sample.
    ///
    /// If `visit` panics, the window may keep only part of `input`. When
    /// `input` holds at least [`Self::len`] samples, the run past the first
    /// `len - 1` is visited before it is pushed, so a panic there leaves only
    /// those first `len - 1` samples.
    #[inline]
    pub fn execute_block_contiguous<F>(&mut self, input: &[T], mut visit: F)
    where
        F: FnMut(Range<usize>, &[T]),
    {
        let transition_len = input.len().min(self.len - 1);

        for (i, &value) in input[..transition_len].iter().enumerate() {
            self.push(value);
            visit(i..i + 1, self.read());
        }

        if input.len() >= self.len {
            visit(self.len - 1..input.len(), input);

            // only the last `self.len` samples need to be retained
            self.write(&input[input.len() - self.len..]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_window_config_errors)]
    fn test_window_config_errors() {
        assert!(Window::<f32>::new(0).is_err());
        assert!(Window::<Complex<f32>>::new(0).is_err());
    }

    #[test]
    #[autotest_annotate(autotest_windowf)]
    fn test_windowf() {
        let v = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let test0 = [0.0; 10];
        let test1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let test2 = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 9.0, 8.0, 7.0, 6.0];
        let test3 = [1.0, 1.0, 9.0, 8.0, 7.0, 6.0, 3.0, 3.0, 3.0, 3.0];
        let test4 = [7.0, 6.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0];
        let test5 = [3.0, 3.0, 5.0, 5.0, 5.0, 5.0];
        let test6 = [5.0, 5.0, 5.0, 5.0, 6.0, 7.0];
        let test7 = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0];
        let test8 = [0.0; 10];

        // create window
        let mut w = Window::<f32>::new(10).unwrap();

        assert_eq!(w.read(), &test0);

        // push 4 elements
        w.push(1.0);
        w.push(1.0);
        w.push(1.0);
        w.push(1.0);

        assert_eq!(w.read(), &test1);

        // push 4 more elements
        w.write(&v[0..4]);

        assert_eq!(w.read(), &test2);

        // push 4 more elements
        w.push(3.0);
        w.push(3.0);
        w.push(3.0);
        w.push(3.0);

        assert_eq!(w.read(), &test3);

        // test indexing operation
        assert_eq!(w.index(0).unwrap(), 1.0);
        assert_eq!(w.index(1).unwrap(), 1.0);
        assert_eq!(w.index(2).unwrap(), 9.0);
        assert_eq!(w.index(3).unwrap(), 8.0);
        assert_eq!(w.index(4).unwrap(), 7.0);
        assert_eq!(w.index(5).unwrap(), 6.0);
        assert_eq!(w.index(6).unwrap(), 3.0);
        assert_eq!(w.index(7).unwrap(), 3.0);
        assert_eq!(w.index(8).unwrap(), 3.0);
        assert_eq!(w.index(9).unwrap(), 3.0);
        assert!(w.index(999).is_err()); // out of range

        // push 4 more elements
        w.push(5.0);
        w.push(5.0);
        w.push(5.0);
        w.push(5.0);

        assert_eq!(w.read(), &test4);

        // recreate window (truncate to last 6 elements)
        w.resize(6).unwrap();
        assert_eq!(w.read(), &test5);

        // push 2 more elements
        w.push(6.0);
        w.push(7.0);
        assert_eq!(w.read(), &test6);

        // recreate window (extend to 10 elements)
        w.resize(10).unwrap();
        assert_eq!(w.read(), &test7);

        // reset
        w.reset();
        assert_eq!(w.read(), &test8);
    }

    #[test]
    #[autotest_annotate(autotest_window_copy)]
    fn test_window_copy() {
        let wlen = 20;
        let mut q0 = Window::<Complex<f32>>::new(wlen).unwrap();

        // write some values
        // TODO maybe replace with randnf()
        for _ in 0..wlen {
            let v = Complex::new(rand::random::<f32>(), rand::random::<f32>());
            q0.push(v);
        }

        // copy object
        let mut q1 = q0.clone();

        // write a few more values
        for _ in 0..wlen / 2 {
            let v = Complex::new(rand::random::<f32>(), rand::random::<f32>());
            q0.push(v);
            q1.push(v);
        }

        // read buffers and compare
        assert_eq!(q0.read(), q1.read());
    }

    #[test]
    fn test_window_execute_block_matches_push() {
        let len = 5;
        let initial = [10, 11, 12, 13, 14];
        let input = [20, 21, 22, 23, 24, 25, 26, 27];
        let mut expected = Window::new(len).unwrap();
        let mut actual = Window::new(len).unwrap();
        expected.write(&initial);
        actual.write(&initial);

        let mut expected_histories = Vec::new();
        for &value in &input {
            expected.push(value);
            expected_histories.push(expected.read().to_vec());
        }

        let mut actual_histories = Vec::new();
        actual.execute_block(&input, |i, history| {
            assert_eq!(i, actual_histories.len());
            actual_histories.push(history.to_vec());
        });

        assert_eq!(actual_histories, expected_histories);
        assert_eq!(actual.read(), expected.read());
    }

    #[test]
    fn test_window_execute_block_contiguous() {
        let len = 4;
        let initial = [10, 11, 12, 13];
        let input = [20, 21, 22, 23, 24, 25, 26];
        let mut window = Window::new(len).unwrap();
        window.write(&initial);

        let mut calls = Vec::new();
        window.execute_block_contiguous(&input, |indices, samples| {
            calls.push((indices, samples.to_vec()));
        });

        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0], (0..1, vec![11, 12, 13, 20]));
        assert_eq!(calls[1], (1..2, vec![12, 13, 20, 21]));
        assert_eq!(calls[2], (2..3, vec![13, 20, 21, 22]));
        assert_eq!(calls[3], (3..7, input.to_vec()));
        assert_eq!(window.read(), &[23, 24, 25, 26]);
    }

    #[test]
    fn test_window_execute_block_contiguous_matches_push() {
        for len in 1..=8 {
            for input_len in 0..=2 * len + 2 {
                let initial: Vec<_> = (0..len).map(|i| 100 + i).collect();
                let input: Vec<_> = (0..input_len).map(|i| 200 + i).collect();
                let mut expected = Window::new(len).unwrap();
                let mut actual = Window::new(len).unwrap();
                expected.write(&initial);
                actual.write(&initial);

                let mut expected_histories = Vec::new();
                for &value in &input {
                    expected.push(value);
                    expected_histories.push(expected.read().to_vec());
                }

                let mut actual_histories = vec![None; input_len];
                actual.execute_block_contiguous(&input, |indices, samples| {
                    assert_eq!(samples.len(), indices.len() + len - 1);
                    for (i, history) in indices.zip(samples.windows(len)) {
                        assert!(actual_histories[i].replace(history.to_vec()).is_none());
                    }
                });

                assert_eq!(actual_histories.into_iter().collect::<Option<Vec<_>>>().unwrap(), expected_histories);
                assert_eq!(actual.read(), expected.read());
            }
        }
    }
}
