use crate::buffer::Window;
use crate::error::{Error, Result};
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct OrdFilt<T> {
    k: usize,
    buf: Window<T>,
    buf_sorted: Vec<T>,
}

impl<T> OrdFilt<T>
where
    T: Clone + Copy + PartialOrd + Default,
{
    pub fn new(n: usize, k: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::Config("filter length must be greater than zero".into()));
        }
        if k >= n {
            return Err(Error::Config("filter index must be in [0,n-1]".into()));
        }

        let buf = Window::new(n)?;
        let buf_sorted = vec![T::default(); n];

        let mut q = Self { k, buf, buf_sorted };
        q.reset();
        Ok(q)
    }

    pub fn new_medfilt(m: usize) -> Result<Self> {
        Self::new(2 * m + 1, m)
    }

    pub fn reset(&mut self) {
        self.buf.reset();
    }

    pub fn push(&mut self, x: T) -> () {
        self.buf.push(x);
    }

    pub fn write(&mut self, x: &[T]) -> () {
        self.buf.write(x);
    }

    pub fn execute(&mut self) -> Result<T> {
        let r = self.buf.read();
        self.buf_sorted.copy_from_slice(r);
        self.buf_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Ok(self.buf_sorted[self.k])
    }

    pub fn execute_one(&mut self, x: T) -> Result<T> {
        self.push(x);
        self.execute()
    }

    pub fn execute_block(&mut self, x: &[T], y: &mut [T]) -> Result<()> {
        for (&xi, yi) in x.iter().zip(y.iter_mut()) {
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
        let mut q0 = OrdFilt::<f32>::new(17, 5).unwrap();

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
