use crate::buffer::Window;
use crate::error::{Error, Result};
use std::cmp::Ordering;

/// Filter that returns the value of a fixed rank from a sliding window.
///
/// The filter sorts the most recent `filter_length` samples in ascending order
/// and returns the one at the chosen rank. Rank zero selects the minimum, and
/// rank `filter_length - 1` selects the maximum. [`Self::new_medfilt`] creates
/// a median filter. The window starts filled with `T::default()`.
///
/// Push samples with [`Self::push`] or [`Self::write`] and read the result with
/// [`Self::execute`], or use [`Self::execute_one`] or [`Self::execute_block`]
/// to do both.
///
/// Samples must have a total order. Writes reject values that are incomparable
/// with themselves, such as floating-point NaN, before changing the window.
#[derive(Clone, Debug)]
#[doc(alias = "OrdFilt")]
pub struct OrderStatisticFilter<T> {
    k: usize,
    buf: Window<T>,
    buf_sorted: Vec<T>,
}

impl<T> OrderStatisticFilter<T>
where
    T: Clone + Copy + PartialOrd + Default,
{
    /// Creates a filter that selects `rank` from a window of `filter_length`
    /// samples.
    ///
    /// Ranks count from zero in ascending order.
    ///
    /// # Errors
    ///
    /// Returns an error when `filter_length` is zero or `rank` is not less than
    /// `filter_length`.
    pub fn new(filter_length: usize, rank: usize) -> Result<Self> {
        if filter_length == 0 {
            return Err(Error::Config("filter length must be greater than zero".into()));
        }
        if rank >= filter_length {
            return Err(Error::Config("filter rank must be less than the filter length".into()));
        }

        let buf = Window::new(filter_length)?;
        let buf_sorted = vec![T::default(); filter_length];

        let mut q = Self { k: rank, buf, buf_sorted };
        q.reset();
        Ok(q)
    }

    /// Creates a median filter with a window of `2 * filter_semi_length + 1`
    /// samples.
    ///
    /// # Panics
    ///
    /// Panics when the window length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps.
    pub fn new_medfilt(filter_semi_length: usize) -> Result<Self> {
        Self::new(2 * filter_semi_length + 1, filter_semi_length)
    }

    /// Refills the window with `T::default()`.
    pub fn reset(&mut self) {
        self.buf.reset();
    }

    /// Adds one sample to the window without selecting a value.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` is incomparable with itself, such as NaN.
    /// The window is unchanged on error.
    pub fn push(&mut self, input: T) -> Result<()> {
        if input.partial_cmp(&input).is_none() {
            return Err(Error::Value("sample is incomparable with itself".into()));
        }
        self.buf.push(input);
        Ok(())
    }

    /// Adds a block of samples to the window without selecting a value.
    ///
    /// # Errors
    ///
    /// Returns an error when any sample is incomparable with itself, such as
    /// NaN. The entire window is unchanged on error.
    pub fn write(&mut self, input: &[T]) -> Result<()> {
        if input.iter().any(|sample| sample.partial_cmp(sample).is_none()) {
            return Err(Error::Value("sample is incomparable with itself".into()));
        }
        self.buf.write(input);
        Ok(())
    }

    /// Returns the value at the chosen rank in the current window.
    ///
    /// The window is not changed.
    ///
    pub fn execute(&mut self) -> Result<T> {
        let r = self.buf.read();
        self.buf_sorted.copy_from_slice(r);
        self.buf_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Ok(self.buf_sorted[self.k])
    }

    /// Adds one sample and returns the value at the chosen rank.
    ///
    /// # Errors
    ///
    /// Returns an error when `input` is incomparable with itself, such as NaN.
    /// The window is unchanged on error.
    pub fn execute_one(&mut self, input: T) -> Result<T> {
        self.push(input)?;
        self.execute()
    }

    /// Adds each sample of a block and writes the selected value after each one.
    ///
    /// `output` must hold at least `input.len()` values. Extra entries are left
    /// unchanged. Returns `input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`, leaving the filter
    /// and `output` unchanged. An invalid sample stops the block before it is
    /// stored. Earlier inputs and outputs remain processed. Later inputs are not
    /// processed.
    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Config("output block length must be at least the input length".into()));
        }
        for (&xi, yi) in input.iter().zip(output.iter_mut()) {
            *yi = self.execute_one(xi)?;
        }
        Ok(input.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
    use crate::utility::test_helpers::validate_block_execute;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_ordfilt_copy)]
    fn test_ordfilt_copy() {
        // create base object
        let mut q0 = OrderStatisticFilter::<f32>::new(17, 5).unwrap();

        // run samples through filter
        for _ in 0..20 {
            let v = randnf();
            let _ = q0.execute_one(v).unwrap();
        }

        // copy object
        let mut q1 = q0.clone();

        // run samples through both filters in parallel
        for _ in 0..60 {
            let v = randnf();
            let y0 = q0.execute_one(v).unwrap();
            let y1 = q1.execute_one(v).unwrap();

            assert_eq!(y0, y1);
        }

        // No need to explicitly destroy objects in Rust
    }

    #[test]
    fn test_ordfilt_execute_block_contract() {
        let q = OrderStatisticFilter::<f32>::new(5, 2).unwrap();
        validate_block_execute(&q, &[1.0, -0.5, 0.25], -9.0, OrderStatisticFilter::execute_block);
    }
}
