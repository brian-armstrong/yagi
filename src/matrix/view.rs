use std::ops::{Index, IndexMut};

use crate::error::{Error, Result};

fn validate_shape(actual_len: usize, rows: usize, cols: usize) -> Result<()> {
    let expected_len =
        rows.checked_mul(cols).ok_or_else(|| Error::Range("matrix dimensions overflow usize".to_owned()))?;
    if actual_len != expected_len {
        return Err(Error::Range(format!(
            "matrix shape {rows} x {cols} requires a buffer of length {expected_len}, got {actual_len}"
        )));
    }
    Ok(())
}

/// A checked, row-major matrix view.
///
/// This allows better structured access to a slice of elements. Checks length at construction.
#[derive(Clone, Copy, Debug)]
pub struct MatrixRef<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> MatrixRef<'a, T> {
    /// Construct a view over an exactly-sized row-major buffer.
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Result<Self> {
        validate_shape(data.len(), rows, cols)?;
        Ok(Self { data, rows, cols })
    }

    /// Return the underlying row-major storage.
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }

    /// Return `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Return one row. Panics when `row` is out of bounds.
    pub fn row(&self, row: usize) -> &[T] {
        let range = self.row_range(row);
        &self.data[range]
    }

    /// Iterate over rows in row-major order.
    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        (0..self.rows).map(move |row| self.row(row))
    }

    /// Iterate over a column. Panics when `column` is out of bounds.
    pub fn column(&self, column: usize) -> impl Iterator<Item = &T> {
        assert!(column < self.cols, "matrix column index out of bounds");
        self.data.chunks_exact(self.cols).map(move |row| &row[column])
    }

    fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        assert!(row < self.rows, "matrix row index out of bounds");
        let start = row * self.cols;
        start..start + self.cols
    }
}

impl<T> Index<(usize, usize)> for MatrixRef<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.row(index.0)[index.1]
    }
}

/// A checked, mutable row-major matrix view.
#[derive(Debug)]
pub struct MatrixMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> MatrixMut<'a, T> {
    /// Construct a mutable view over an exactly-sized row-major buffer.
    pub fn new(data: &'a mut [T], rows: usize, cols: usize) -> Result<Self> {
        validate_shape(data.len(), rows, cols)?;
        Ok(Self { data, rows, cols })
    }

    /// Return the underlying row-major storage.
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Borrow this mutable view as a read-only view without revalidating its shape.
    pub fn as_ref(&self) -> MatrixRef<'_, T> {
        MatrixRef { data: &*self.data, rows: self.rows, cols: self.cols }
    }

    /// Return the underlying row-major storage mutably.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Return `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Return one row. Panics when `row` is out of bounds.
    pub fn row(&self, row: usize) -> &[T] {
        let range = self.row_range(row);
        &self.data[range]
    }

    /// Return one row mutably. Panics when `row` is out of bounds.
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        let range = self.row_range(row);
        &mut self.data[range]
    }

    /// Iterate over rows mutably in row-major order.
    pub fn rows_mut(&mut self) -> MatrixRowsMut<'_, T> {
        MatrixRowsMut { remaining: &mut *self.data, rows_remaining: self.rows, cols: self.cols }
    }

    /// Iterate over a column. Panics when `column` is out of bounds.
    pub fn column(&self, column: usize) -> impl Iterator<Item = &T> {
        assert!(column < self.cols, "matrix column index out of bounds");
        self.data.chunks_exact(self.cols).map(move |row| &row[column])
    }

    pub(crate) fn for_each_other_row_mut(
        &mut self,
        pivot: usize,
        mut operation: impl FnMut(&[T], &mut [T]),
    ) -> Result<()> {
        if pivot >= self.rows || self.cols == 0 {
            return Err(Error::Range("matrix pivot row index out of bounds".to_owned()));
        }

        let pivot_start = pivot * self.cols;
        let (before_pivot, pivot_and_after) = self.data.split_at_mut(pivot_start);
        let (pivot_row, after_pivot) = pivot_and_after.split_at_mut(self.cols);

        for target_row in before_pivot.chunks_exact_mut(self.cols) {
            operation(pivot_row, target_row);
        }
        for target_row in after_pivot.chunks_exact_mut(self.cols) {
            operation(pivot_row, target_row);
        }
        Ok(())
    }

    pub(crate) fn two_rows_mut(&mut self, first: usize, second: usize) -> Option<(&mut [T], &mut [T])> {
        if first == second || first >= self.rows || second >= self.rows {
            return None;
        }

        if first < second {
            let (before_second, second_and_after) = self.data.split_at_mut(second * self.cols);
            let first_row = &mut before_second[first * self.cols..(first + 1) * self.cols];
            let second_row = &mut second_and_after[..self.cols];
            Some((first_row, second_row))
        } else {
            let (before_first, first_and_after) = self.data.split_at_mut(first * self.cols);
            let second_row = &mut before_first[second * self.cols..(second + 1) * self.cols];
            let first_row = &mut first_and_after[..self.cols];
            Some((first_row, second_row))
        }
    }

    pub(crate) fn swap_rows(&mut self, first: usize, second: usize) -> Result<()> {
        if first >= self.rows || second >= self.rows {
            return Err(Error::Range("matrix row index out of bounds".to_owned()));
        }
        if first != second {
            let (first_row, second_row) = self.two_rows_mut(first, second).expect("distinct valid rows");
            first_row.swap_with_slice(second_row);
        }
        Ok(())
    }

    fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        assert!(row < self.rows, "matrix row index out of bounds");
        let start = row * self.cols;
        start..start + self.cols
    }
}

impl<T> Index<(usize, usize)> for MatrixMut<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.row(index.0)[index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for MatrixMut<'_, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.row_mut(index.0)[index.1]
    }
}

/// Mutable matrix row iterator
pub struct MatrixRowsMut<'a, T> {
    remaining: &'a mut [T],
    rows_remaining: usize,
    cols: usize,
}

impl<'a, T> Iterator for MatrixRowsMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.rows_remaining == 0 {
            return None;
        }

        self.rows_remaining -= 1;
        let remaining = std::mem::take(&mut self.remaining);
        let (row, tail) = remaining.split_at_mut(self.cols);
        self.remaining = tail;
        Some(row)
    }
}

/// A checked, column-major matrix view used by internal column-oriented kernels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColMatrixRef<'a, T> {
    inner: MatrixRef<'a, T>,
}

impl<'a, T> ColMatrixRef<'a, T> {
    pub(crate) fn column(&self, column: usize) -> &[T] {
        self.inner.row(column)
    }
}

impl<T> Index<(usize, usize)> for ColMatrixRef<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.inner[(index.1, index.0)]
    }
}

/// A checked, mutable column-major matrix view used by internal column-oriented kernels.
#[derive(Debug)]
pub(crate) struct ColMatrixMut<'a, T> {
    inner: MatrixMut<'a, T>,
}

impl<'a, T> ColMatrixMut<'a, T> {
    pub(crate) fn new(data: &'a mut [T], rows: usize, cols: usize) -> Result<Self> {
        Ok(Self { inner: MatrixMut::new(data, cols, rows)? })
    }

    pub(crate) fn as_ref(&self) -> ColMatrixRef<'_, T> {
        ColMatrixRef { inner: self.inner.as_ref() }
    }

    pub(crate) fn column(&self, column: usize) -> &[T] {
        self.inner.row(column)
    }

    pub(crate) fn column_mut(&mut self, column: usize) -> &mut [T] {
        self.inner.row_mut(column)
    }

    pub(crate) fn columns_mut(&mut self) -> MatrixRowsMut<'_, T> {
        self.inner.rows_mut()
    }

    pub(crate) fn two_columns_mut(&mut self, first: usize, second: usize) -> Option<(&mut [T], &mut [T])> {
        self.inner.two_rows_mut(first, second)
    }
}

impl<T> Index<(usize, usize)> for ColMatrixMut<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.inner[(index.1, index.0)]
    }
}

impl<T> IndexMut<(usize, usize)> for ColMatrixMut<'_, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.inner[(index.1, index.0)]
    }
}

#[cfg(test)]
mod tests {
    use super::{ColMatrixMut, MatrixMut, MatrixRef};

    #[test]
    fn test_matrix_views_require_exact_shape() {
        assert!(MatrixRef::new(&[1u8, 2, 3, 4], 2, 2).is_ok());
        assert!(MatrixRef::new(&[1u8, 2, 3], 2, 2).is_err());
        assert!(MatrixRef::new(&[1u8, 2, 3, 4, 5], 2, 2).is_err());

        let mut short = [1u8, 2, 3];
        let mut long = [1u8, 2, 3, 4, 5];
        assert!(MatrixMut::new(&mut short, 2, 2).is_err());
        assert!(MatrixMut::new(&mut long, 2, 2).is_err());
    }

    #[test]
    fn test_matrix_views_reject_shape_overflow() {
        assert!(MatrixRef::<u8>::new(&[], usize::MAX, 2).is_err());

        let mut data = [];
        assert!(MatrixMut::<u8>::new(&mut data, usize::MAX, 2).is_err());
    }

    #[test]
    fn test_matrix_views_support_empty_shapes() {
        let rows = MatrixRef::<u8>::new(&[], 3, 0).unwrap().rows().count();
        assert_eq!(rows, 3);
        assert_eq!(MatrixRef::<u8>::new(&[], 0, 3).unwrap().rows().count(), 0);
        assert_eq!(MatrixRef::<u8>::new(&[], 0, 0).unwrap().shape(), (0, 0));

        let mut data = [];
        assert_eq!(MatrixMut::<u8>::new(&mut data, 3, 0).unwrap().rows_mut().count(), 3);
    }

    #[test]
    fn test_matrix_view_rows_and_indexing() {
        let data = [1u8, 2, 3, 4, 5, 6];
        let matrix = MatrixRef::new(&data, 2, 3).unwrap();

        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix.row(1), [4, 5, 6]);
        assert_eq!(matrix.rows().collect::<Vec<_>>(), vec![&[1, 2, 3][..], &[4, 5, 6][..]]);
        assert_eq!(matrix.column(1).copied().collect::<Vec<_>>(), [2, 5]);
        assert_eq!(matrix[(1, 2)], 6);
        assert_eq!(matrix.as_slice(), data);
    }

    #[test]
    fn test_matrix_mut_view_rows_and_indexing() {
        let mut data = [1u8, 2, 3, 4, 5, 6];
        let mut matrix = MatrixMut::new(&mut data, 2, 3).unwrap();

        matrix[(0, 1)] = 9;
        matrix.row_mut(1)[0] = 8;
        for row in matrix.rows_mut() {
            row[2] += 1;
        }
        assert_eq!(matrix.shape(), (2, 3));
        assert_eq!(matrix.row(0), [1, 9, 4]);
        assert_eq!(matrix.row(1), [8, 5, 7]);
        let readonly = matrix.as_ref();
        assert_eq!(readonly.shape(), (2, 3));
        assert_eq!(readonly.as_slice(), [1, 9, 4, 8, 5, 7]);
        assert_eq!(matrix.as_mut_slice(), [1, 9, 4, 8, 5, 7]);
    }

    #[test]
    fn test_column_major_views() {
        let mut data = [1u8, 4, 2, 5, 3, 6];
        let mut matrix = ColMatrixMut::new(&mut data, 2, 3).unwrap();

        let readonly = matrix.as_ref();
        assert_eq!(readonly.column(1), [2, 5]);
        assert_eq!(readonly[(1, 2)], 6);

        let (first, third) = matrix.two_columns_mut(0, 2).unwrap();
        first[1] = 7;
        third[0] = 8;
        matrix.column_mut(1)[0] = 9;
        for column in matrix.columns_mut() {
            column[0] += 1;
        }

        assert_eq!(matrix.column(1), [10, 5]);
        assert_eq!(matrix[(1, 0)], 7);
        assert_eq!(matrix.as_ref()[(0, 2)], 9);
        assert_eq!(data, [2, 7, 10, 5, 9, 6]);
    }

    #[test]
    fn test_matrix_mut_view_disjoint_rows_and_swap() {
        let mut data = [1u8, 2, 3, 4, 5, 6];
        let mut matrix = MatrixMut::new(&mut data, 2, 3).unwrap();

        let (first, second) = matrix.two_rows_mut(1, 0).unwrap();
        first[0] = 9;
        second[2] = 8;
        matrix.swap_rows(0, 1).unwrap();

        assert_eq!(matrix.as_slice(), [9, 5, 6, 1, 2, 8]);
        assert!(matrix.two_rows_mut(0, 0).is_none());
        assert!(matrix.two_rows_mut(0, 2).is_none());
        assert!(matrix.swap_rows(0, 2).is_err());
    }
}
