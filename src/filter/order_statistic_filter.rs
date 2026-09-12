use crate::buffer::Window;
use crate::error::{Error, Result};
use std::cmp::Ordering;

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

    pub fn new_medfilt(filter_semi_length: usize) -> Result<Self> {
        Self::new(2 * filter_semi_length + 1, filter_semi_length)
    }

    pub fn reset(&mut self) {
        self.buf.reset();
    }

    pub fn push(&mut self, input: T) {
        self.buf.push(input);
    }

    pub fn write(&mut self, input: &[T]) {
        self.buf.write(input);
    }

    pub fn execute(&mut self) -> Result<T> {
        let r = self.buf.read();
        self.buf_sorted.copy_from_slice(r);
        self.buf_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Ok(self.buf_sorted[self.k])
    }

    pub fn execute_one(&mut self, input: T) -> Result<T> {
        self.push(input);
        self.execute()
    }

    pub fn execute_block(&mut self, input: &[T], output: &mut [T]) -> Result<()> {
        for (&xi, yi) in input.iter().zip(output.iter_mut()) {
            *yi = self.execute_one(xi)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
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
}
