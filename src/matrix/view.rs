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
#[derive(Clone, Copy)]
pub(crate) struct MatrixRef<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> MatrixRef<'a, T> {
    pub(crate) fn new(data: &'a [T], rows: usize, cols: usize) -> Result<Self> {
        validate_shape(data.len(), rows, cols)?;
        Ok(Self { data, rows, cols })
    }

    pub(crate) fn as_slice(&self) -> &'a [T] {
        self.data
    }

    #[allow(dead_code)]
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub(crate) fn row(&self, row: usize) -> &[T] {
        let range = self.row_range(row);
        &self.data[range]
    }

    pub(crate) fn rows(&self) -> impl Iterator<Item = &[T]> {
        (0..self.rows).map(move |row| self.row(row))
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
pub(crate) struct MatrixMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> MatrixMut<'a, T> {
    pub(crate) fn new(data: &'a mut [T], rows: usize, cols: usize) -> Result<Self> {
        validate_shape(data.len(), rows, cols)?;
        Ok(Self { data, rows, cols })
    }

    #[allow(dead_code)]
    pub(crate) fn as_slice(&self) -> &[T] {
        self.data
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub(crate) fn row(&self, row: usize) -> &[T] {
        let range = self.row_range(row);
        &self.data[range]
    }

    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [T] {
        let range = self.row_range(row);
        &mut self.data[range]
    }

    pub(crate) fn rows_mut(&mut self) -> MatrixRowsMut<'_, T> {
        MatrixRowsMut { remaining: &mut *self.data, rows_remaining: self.rows, cols: self.cols }
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

    #[allow(dead_code)]
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

/// Mutable row iterator that deliberately supports matrices with zero columns.
pub(crate) struct MatrixRowsMut<'a, T> {
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

#[cfg(test)]
mod tests {
    use super::{MatrixMut, MatrixRef};

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
        assert_eq!(matrix.as_mut_slice(), [1, 9, 4, 8, 5, 7]);
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
