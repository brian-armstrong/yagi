use num_traits::One;
use std::ops::Add;

use crate::error::{Error, Result};

// sparse matrix structure
// example: the following floating-point sparse matrix is represented
//          with the corresponding values:
//    [ 0   0   0   0   0 ]
//    [ 0 2.3   0   0   0 ]
//    [ 0   0   0   0 1.2 ]
//    [ 0   0   0   0   0 ]
//    [ 0 3.4   0 4.4   0 ]
//    [ 0   0   0   0   0 ]
//
//  M (rows)        :   6
//  N (cols)        :   5
//  mlist           :   { {}, {1},   {4}, {},  {1,3}, {} }
//  nlist           :   { {}, {1,4}, {},  {4}, {2} }
//  mvals           :   { {}, {2.3}, {1.2}, {}, {3.4,4.4}, {} }
//  nvals           :   { {}, {2.3, 3.4}, {}, {4.4}, {1.2} }
//  num_mlist       :   { 0, 1, 1, 0, 2, 0 }
//  num_nlist       :   { 0, 2, 0, 1, 1 }
//  max_num_mlist   :   2
//  max_num_nlist   :   2
//
// NOTE: while this particular example does not show a particular
//       improvement in memory use, such a case can be made for
//       extremely large matrices which only have a few non-zero
//       entries.
//
/// Sparse matrix that stores its entries in per-row and per-column lists.
///
/// Each stored entry appears in the list of its row and the list of its
/// column, kept in index order. An entry stays stored after it is set to
/// `T::default()`, and multiplication still visits it. Indexes are stored as
/// `u16`, so dimensions above 65536 do not work.
///
/// Build a matrix with [`Self::new`] or [`Self::from_array`] and
/// [`Self::set`]. [`Self::delete`] and [`Self::reset`] leave the lists
/// inconsistent, as their docs describe, so build a new matrix instead of
/// editing one.
#[derive(Debug, Clone)]
#[doc(alias = "SMatrix")]
pub struct SparseMatrix<T> {
    m: usize,              // number of rows
    n: usize,              // number of columns
    mlist: Vec<Vec<u16>>,  // list of non-zero col indices in each row
    nlist: Vec<Vec<u16>>,  // list of non-zero row indices in each col
    mvals: Vec<Vec<T>>,    // list of non-zero values in each row
    nvals: Vec<Vec<T>>,    // list of non-zero values in each col
    num_mlist: Vec<usize>, // weight of each row, m
    num_nlist: Vec<usize>, // weight of each row, n
    max_num_mlist: usize,  // maximum of num_mlist
    max_num_nlist: usize,  // maximum of num_nlist
}

impl<T: Default + Copy + PartialEq + std::fmt::Display + One + Add<Output = T>> SparseMatrix<T> {
    /// Creates a matrix with no stored entries.
    ///
    /// Keep both dimensions at most 65536. Larger dimensions are accepted, but
    /// an index above 65535 wraps when stored and collides with a lower index.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero.
    pub fn new(rows: usize, columns: usize) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(Error::Config("matrix dimensions must be greater than zero".to_string()));
        }

        Ok(SparseMatrix {
            m: rows,
            n: columns,
            mlist: vec![Vec::new(); rows],
            nlist: vec![Vec::new(); columns],
            mvals: vec![Vec::new(); rows],
            nvals: vec![Vec::new(); columns],
            num_mlist: vec![0; rows],
            num_nlist: vec![0; columns],
            max_num_mlist: 0,
            max_num_nlist: 0,
        })
    }

    /// Creates a matrix from dense row-major values.
    ///
    /// Only values that differ from `T::default()` are stored. Values beyond
    /// the first `rows * columns` are ignored.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero.
    ///
    /// # Panics
    ///
    /// Panics when `values` has fewer than `rows * columns` elements.
    pub fn from_array(values: &[T], rows: usize, columns: usize) -> Result<Self> {
        let mut q = Self::new(rows, columns)?;

        for i in 0..rows {
            for j in 0..columns {
                if values[i * columns + j] != T::default() {
                    q.set(i, j, values[i * columns + j]);
                }
            }
        }

        Ok(q)
    }

    /// Prints the row and column lists to standard output.
    ///
    /// The output is meant for inspection and debugging. Its format may change.
    pub fn print(&self) {
        println!("dims : {} {}", self.m, self.n);
        println!("max  : {} {}", self.max_num_mlist, self.max_num_nlist);
        print!("rows :");
        for i in 0..self.m {
            print!(" {}", self.num_mlist[i]);
        }
        println!();
        print!("cols :");
        for j in 0..self.n {
            print!(" {}", self.num_nlist[j]);
        }
        println!();

        // print mlist
        println!("row indices:");
        for i in 0..self.m {
            if self.num_mlist[i] == 0 {
                continue;
            }
            print!("  {:3} :", i);
            for j in 0..self.num_mlist[i] {
                print!(" {}", self.mlist[i][j]);
            }
            println!();
        }

        // print nlist
        println!("column indices:");
        for j in 0..self.n {
            if self.num_nlist[j] == 0 {
                continue;
            }
            print!("  {:3} :", j);
            for i in 0..self.num_nlist[j] {
                print!(" {}", self.nlist[j][i]);
            }
            println!();
        }

        // print mvals
        println!("row values:");
        for i in 0..self.m {
            print!("  {:3} :", i);
            for j in 0..self.num_mlist[i] {
                print!(" {:6.2}", self.mvals[i][j]);
            }
            println!();
        }

        // print nvals
        println!("column values:");
        for j in 0..self.n {
            print!("  {:3} :", j);
            for i in 0..self.num_nlist[j] {
                print!(" {:6.2}", self.nvals[j][i]);
            }
            println!();
        }
    }

    /// Prints the matrix in dense form to standard output.
    ///
    /// The output is meant for inspection and debugging. Its format may change.
    pub fn print_expanded(&self) {
        for i in 0..self.m {
            let mut t = 0;
            for j in 0..self.n {
                if t == self.num_mlist[i] {
                    print!("{:6.2} ", T::default());
                } else if self.mlist[i][t] == j as u16 {
                    print!("{:6.2} ", self.mvals[i][t]);
                    t += 1;
                } else {
                    print!("{:6.2} ", T::default());
                }
            }
            println!();
        }
    }

    /// Returns the shape as `(rows, columns)`.
    pub fn size(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Sets every stored value to `T::default()`.
    ///
    /// The entries stay stored, so [`Self::isset`] still reports them and
    /// multiplication still visits them.
    pub fn clear(&mut self) {
        for i in 0..self.m {
            for j in 0..self.num_mlist[i] {
                self.mvals[i][j] = T::default();
            }
        }

        for j in 0..self.n {
            for i in 0..self.num_nlist[j] {
                self.nvals[j][i] = T::default();
            }
        }
    }

    /// Marks every row and column as empty.
    ///
    /// # Warning
    ///
    /// The stored entries are not removed. [`Self::get`], [`Self::isset`] and
    /// [`Self::set`] still find them, while the multiplication methods skip
    /// them. A value set at an old position after a reset is therefore ignored
    /// by multiplication. Create a new matrix instead of resetting a populated
    /// one.
    pub fn reset(&mut self) {
        for i in 0..self.m {
            self.num_mlist[i] = 0;
        }
        for j in 0..self.n {
            self.num_nlist[j] = 0;
        }

        self.max_num_mlist = 0;
        self.max_num_nlist = 0;
    }

    /// Returns whether an entry is stored at `(row, column)`.
    ///
    /// This reports storage, not value. A stored entry can hold
    /// `T::default()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `(row, column)` is outside the matrix.
    pub fn isset(&self, row: usize, column: usize) -> Result<bool> {
        if row >= self.m || column >= self.n {
            return Err(Error::Range(format!(
                "smatrix_isset({},{}), index exceeds matrix dimension ({},{})",
                row, column, self.m, self.n
            )));
        }

        Ok(self.mlist[row].contains(&(column as u16)))
    }

    // insert element at index
    fn insert(&mut self, m: usize, n: usize, v: T) -> Result<()> {
        if m >= self.m || n >= self.n {
            return Err(Error::Range(format!(
                "smatrix_insert({},{}), index exceeds matrix dimension ({},{})",
                m, n, self.m, self.n
            )));
        }

        // check to see if element is already set
        if self.isset(m, n)? {
            // simply set the value and return
            // println!("smatrix_insert(), value already set...");
            self.set(m, n, v);
            return Ok(());
        }

        // increment list sizes
        self.num_mlist[m] += 1;
        self.num_nlist[n] += 1;

        // find index within list to insert new value
        let mindex = Self::indexsearch(&self.mlist[m], self.num_mlist[m] - 1, n as u16);
        let nindex = Self::indexsearch(&self.nlist[n], self.num_nlist[n] - 1, m as u16);

        // insert indices to appropriate place in list
        self.mlist[m].insert(mindex, n as u16);
        self.nlist[n].insert(nindex, m as u16);

        // insert values to appropriate place in list
        self.mvals[m].insert(mindex, v);
        self.nvals[n].insert(nindex, v);

        // update maximum
        self.max_num_mlist = self.max_num_mlist.max(self.num_mlist[m]);
        self.max_num_nlist = self.max_num_nlist.max(self.num_nlist[n]);

        Ok(())
    }

    /// Removes the stored entry at `(row, column)`.
    ///
    /// Removing an entry that is not stored leaves the matrix unchanged.
    ///
    /// # Warning
    ///
    /// Removal shifts the row and column index lists but not the value lists.
    /// Every later entry in the same row or column is then paired with its
    /// neighbor's value, both for [`Self::get`] and for multiplication. For
    /// example, after deleting `(0, 0)` from a row holding `[1, 2, 3]`,
    /// `get(0, 1)` returns 1. Build a new matrix instead when other entries
    /// must be kept.
    ///
    /// # Errors
    ///
    /// Returns an error when `(row, column)` is outside the matrix.
    pub fn delete(&mut self, row: usize, column: usize) -> Result<()> {
        if row > self.m || column > self.n {
            return Err(Error::Range(format!(
                "smatrix_delete({},{}), index exceeds matrix dimension ({},{})",
                row, column, self.m, self.n
            )));
        }

        // check to see if element is already not set
        if !self.isset(row, column)? {
            return Ok(());
        }

        // remove value from mlist
        let mindex = self.mlist[row].iter().position(|&x| x == column as u16).unwrap();
        self.mlist[row].remove(mindex);

        // remove value from nlist
        let nindex = self.nlist[column].iter().position(|&x| x == row as u16).unwrap();
        self.nlist[column].remove(nindex);

        // reduce sizes
        self.num_mlist[row] -= 1;
        self.num_nlist[column] -= 1;

        // reset maxima
        if self.max_num_mlist == self.num_mlist[row] + 1 {
            self.reset_max_mlist();
        }

        if self.max_num_nlist == self.num_nlist[column] + 1 {
            self.reset_max_nlist();
        }

        Ok(())
    }

    /// Stores `value` at `(row, column)`, replacing any earlier value there.
    ///
    /// A value equal to `T::default()` is still stored.
    ///
    /// # Panics
    ///
    /// Panics when `(row, column)` is outside the matrix.
    pub fn set(&mut self, row: usize, column: usize, value: T) {
        if row >= self.m || column >= self.n {
            panic!("smatrix_set({},{}), index exceeds matrix dimension ({},{})", row, column, self.m, self.n);
        }

        // insert new element if not already allocated
        if !self.isset(row, column).unwrap() {
            self.insert(row, column, value).unwrap();
            return;
        }

        // set value
        let mindex = self.mlist[row].iter().position(|&x| x == column as u16).unwrap();
        self.mvals[row][mindex] = value;

        let nindex = self.nlist[column].iter().position(|&x| x == row as u16).unwrap();
        self.nvals[column][nindex] = value;
    }

    /// Returns the value at `(row, column)`, or `T::default()` when no entry
    /// is stored there.
    ///
    /// # Panics
    ///
    /// Panics when `(row, column)` is outside the matrix.
    pub fn get(&self, row: usize, column: usize) -> T {
        if row >= self.m || column >= self.n {
            panic!("smatrix_get({},{}), index exceeds matrix dimension ({},{})", row, column, self.m, self.n);
        }

        if let Some(mindex) = self.mlist[row].iter().position(|&x| x == column as u16) {
            self.mvals[row][mindex]
        } else {
            T::default()
        }
    }

    /// Makes a new matrix the identity, with ones on the main diagonal.
    ///
    /// For a rectangular matrix, the diagonal runs to the smaller dimension.
    /// This calls [`Self::reset`] first, so on a matrix that already has
    /// entries it has the problems described there. Multiplication can then
    /// ignore old entries and even a diagonal entry that was already stored.
    pub fn eye(&mut self) {
        // reset all elements
        self.reset();

        // set values along diagonal
        let dmin = self.m.min(self.n);
        for i in 0..dmin {
            self.set(i, i, T::one());
        }
    }

    /// Computes the matrix product of this matrix and `b` into `c`.
    ///
    /// The column count of this matrix must equal the row count of `b`. `c`
    /// must have the row count of this matrix and the column count of `b`.
    ///
    /// Old entries in `c` are set to `T::default()` with [`Self::clear`], not
    /// removed. An entry is stored wherever a row of this matrix and a column
    /// of `b` share a stored index, even when the sum is zero.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes do not fit those rules. `c` is
    /// unchanged on error.
    pub fn mul(&self, b: &SparseMatrix<T>, c: &mut SparseMatrix<T>) -> Result<()> {
        // validate input
        if c.m != self.m || c.n != b.n || self.n != b.m {
            return Err(Error::Range("smatrix_mul(), invalid dimensions".to_string()));
        }

        // clear output matrix (retain memory allocation)
        c.clear();

        for r in 0..c.m {
            // find number of non-zero entries in row 'r' of matrix 'self'
            let nnz_a_row = self.num_mlist[r];

            // if this number is zero, there will not be any non-zero
            // entries in the corresponding row of the output matrix 'c'
            if nnz_a_row == 0 {
                continue;
            }

            for col in 0..c.n {
                // find number of non-zero entries in column 'col' of matrix 'b'
                let nnz_b_col = b.num_nlist[col];

                let mut p = T::default();
                let mut set_value = false;

                // find common elements between non-zero elements in
                // row 'r' of matrix 'self' and col 'col' of matrix 'b'
                let mut i = 0; // reset array index for rows of 'self'
                let mut j = 0; // reset array index for cols of 'b'
                while i < nnz_a_row && j < nnz_b_col {
                    let ca = self.mlist[r][i];
                    let rb = b.nlist[col][j];
                    if ca == rb {
                        // match found between self[r,ca] and b[rb,col]
                        p = p + self.mvals[r][i] * b.nvals[col][j];
                        set_value = true;
                        i += 1;
                        j += 1;
                    } else if ca < rb {
                        i += 1; // increment index for 'self'
                    } else {
                        j += 1; // increment index for 'b'
                    }
                }

                // set value if any multiplications have been made
                if set_value {
                    c.set(r, col, p);
                }
            }
        }

        Ok(())
    }

    /// Multiplies this matrix by the column vector `x`, writing the result to
    /// `y`.
    ///
    /// `x` needs one element per column and `y` one per row. Extra input is
    /// ignored and extra output is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics when `y` is too short, or when a stored entry's column lies
    /// beyond the end of `x`.
    pub fn vmul(&self, x: &[T], y: &mut [T]) {
        // initialize to zero
        for yi in &mut y[..self.m] {
            *yi = T::default();
        }

        for i in 0..self.m {
            let mut p = T::default();
            for j in 0..self.num_mlist[i] {
                let col_index = self.mlist[i][j] as usize;
                p = p + self.mvals[i][j] * x[col_index];
            }
            y[i] = p;
        }
    }

    fn reset_max_mlist(&mut self) {
        self.max_num_mlist = self.num_mlist.iter().max().copied().unwrap_or(0);
    }

    fn reset_max_nlist(&mut self) {
        self.max_num_nlist = self.num_nlist.iter().max().copied().unwrap_or(0);
    }

    fn indexsearch(v: &[u16], n: usize, x: u16) -> usize {
        let mut i = 0;
        while i < n && v[i] <= x {
            i += 1;
        }
        i
    }
}

impl SparseMatrix<u8> {
    // ...

    /// Multiplies this binary matrix by a dense row-major `f32` matrix.
    ///
    /// `x` is `mx` by `nx` and `y` is `my` by `ny`. `mx` must equal the column
    /// count of this matrix, `my` its row count, and `ny` must equal `nx`.
    /// Every stored entry counts as one, whatever its `u8` value, including
    /// zero. All of `y` is set to zero before the product is written,
    /// including elements beyond `my * ny`.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes do not fit those rules. `y` is
    /// unchanged on error.
    ///
    /// # Panics
    ///
    /// Panics when an element that a stored entry needs lies beyond the end
    /// of `x` or `y`.
    pub fn mul_f32(&self, x: &[f32], mx: usize, nx: usize, y: &mut [f32], my: usize, ny: usize) -> Result<()> {
        // ensure lengths are valid
        if my != self.m || ny != nx || self.n != mx {
            return Err(Error::Range("smatrix_mul(), invalid dimensions".to_string()));
        }

        // clear output matrix
        y.fill(0.0);

        for i in 0..self.m {
            // find non-zero column entries in this row
            for &j in &self.mlist[i][..self.num_mlist[i]] {
                for k in 0..ny {
                    y[i * ny + k] += x[j as usize * nx + k];
                }
            }
        }

        Ok(())
    }

    /// Multiplies this binary matrix by the `f32` column vector `x`, writing
    /// the result to `y`.
    ///
    /// Every stored entry counts as one, whatever its `u8` value. `x` needs one
    /// element per column and `y` one per row. Extra input is ignored and
    /// extra output is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics when `y` is too short, or when a stored entry's column lies
    /// beyond the end of `x`.
    pub fn vmul_f32(&self, x: &[f32], y: &mut [f32]) {
        for i in 0..self.m {
            // reset total
            y[i] = 0.0;

            // only accumulate values on non-zero entries
            for &j in &self.mlist[i][..self.num_mlist[i]] {
                y[i] += x[j as usize];
            }
        }
    }

    /// Returns `v` modulo two.
    pub fn wrap_bool(v: u8) -> u8 {
        v % 2
    }

    /// Reduces every element of `v` modulo two, in place.
    pub fn wrap_bools(v: &mut [u8]) {
        for vi in v.iter_mut() {
            *vi %= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_smatrixf_vmul)]
    fn test_smatrixf_vmul() {
        let tol = 1e-6f32;

        // Create sparse matrix and set values
        let mut a = SparseMatrix::<f32>::new(4, 5).unwrap();
        a.set(0, 4, 4.0);
        a.set(2, 3, 3.0);
        a.set(3, 0, 2.0);
        a.set(3, 4, 0.0);
        a.set(3, 4, 1.0);

        // Initialize input vector
        let x = vec![7.0, 1.0, 5.0, 2.0, 2.0];

        let y_test = vec![8.0, 0.0, 6.0, 16.0];
        let mut y = vec![0.0; 4];

        // Multiply and run test
        a.vmul(&x, &mut y);

        // Check values
        for i in 0..4 {
            assert_abs_diff_eq!(y[i], y_test[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_smatrixf_mul)]
    fn test_smatrixf_mul() {
        let tol = 1e-6f32;

        // Initialize matrices
        let mut a = SparseMatrix::<f32>::new(4, 5).unwrap();
        let mut b = SparseMatrix::<f32>::new(5, 3).unwrap();
        let mut c = SparseMatrix::<f32>::new(4, 3).unwrap();

        // Initialize 'a'
        a.set(0, 4, 4.0);
        a.set(2, 3, 3.0);
        a.set(3, 0, 2.0);
        a.set(3, 4, 0.0);
        a.set(3, 4, 1.0);

        // Initialize 'b'
        b.set(0, 0, 7.0);
        b.set(0, 1, 6.0);
        b.set(3, 1, 5.0);
        b.set(4, 0, 2.0);

        // Compute 'c'
        a.mul(&b, &mut c).unwrap();

        let c_test = vec![8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 16.0, 12.0, 0.0];

        // Check values
        for i in 0..4 {
            for j in 0..3 {
                assert_abs_diff_eq!(c.get(i, j), c_test[i * 3 + j], epsilon = tol);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_smatrixb_vmul)]
    fn test_smatrixb_vmul() {
        // Create sparse matrix and set values
        let mut a = SparseMatrix::<u8>::new(8, 12).unwrap();
        a.set(0, 0, 1);
        a.set(2, 0, 1);
        a.set(6, 0, 1);
        a.set(3, 2, 1);
        a.set(6, 2, 1);
        a.set(7, 2, 1);
        a.set(1, 3, 1);
        a.set(7, 5, 1);
        a.set(3, 6, 1);
        a.set(5, 6, 1);
        a.set(7, 6, 1);
        a.set(3, 7, 1);
        a.set(2, 8, 1);
        a.set(5, 8, 1);
        a.set(2, 9, 1);
        a.set(5, 10, 1);
        a.set(6, 10, 1);
        a.set(6, 11, 1);

        // Generate vectors
        let x = vec![1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1];
        let y_test = vec![1, 0, 1, 1, 0, 1, 0, 0];
        let mut y = vec![0; 8];

        // Multiply and run test
        a.vmul(&x, &mut y);
        SparseMatrix::<u8>::wrap_bools(&mut y);

        assert_eq!(y, y_test);
    }

    #[test]
    #[autotest_annotate(autotest_smatrixb_mul)]
    fn test_smatrixb_mul() {
        #[rustfmt::skip]
        let a_test: Vec<u8> = vec![
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0
        ];

        #[rustfmt::skip]
        let b_test: Vec<u8> = vec![
            1, 1, 0, 0, 0,
            0, 0, 0, 0, 1,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 1,
            0, 0, 0, 1, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 0,
            0, 1, 0, 0, 1,
            1, 0, 0, 1, 0,
            0, 1, 0, 0, 0
        ];

        #[rustfmt::skip]
        let c_test: Vec<u8> = vec![
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 1,
            0, 0, 0, 1, 0
        ];

        let a = SparseMatrix::<u8>::from_array(&a_test, 8, 12).unwrap();
        let b = SparseMatrix::<u8>::from_array(&b_test, 12, 5).unwrap();
        let mut c = SparseMatrix::<u8>::new(8, 5).unwrap();

        // Compute output
        a.mul(&b, &mut c).unwrap();

        for i in 0..8 {
            for j in 0..5 {
                assert_eq!(SparseMatrix::<u8>::wrap_bool(c.get(i, j)), c_test[i * 5 + j]);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_smatrixb_mulf)]
    fn test_smatrixb_mulf() {
        let tol = 1e-6f32;

        // Create sparse matrix and set values
        let mut a = SparseMatrix::<u8>::new(8, 12).unwrap();
        a.set(0, 0, 1);
        a.set(2, 0, 1);
        a.set(6, 0, 1);
        a.set(3, 2, 1);
        a.set(6, 2, 1);
        a.set(7, 2, 1);
        a.set(1, 3, 1);
        a.set(7, 5, 1);
        a.set(3, 6, 1);
        a.set(5, 6, 1);
        a.set(7, 6, 1);
        a.set(3, 7, 1);
        a.set(2, 8, 1);
        a.set(5, 8, 1);
        a.set(2, 9, 1);
        a.set(5, 10, 1);
        a.set(6, 10, 1);
        a.set(6, 11, 1);

        // Generate vectors
        #[rustfmt::skip]
        let x: Vec<f32> = vec![
            -4.3, -0.7,  3.7,
            -1.7,  2.8,  4.3,
             2.0,  1.9,  0.6,
             3.6,  1.0, -3.7,
             4.3,  0.7,  2.1,
             4.6,  0.5,  0.8,
             1.6, -3.8, -0.8,
            -1.9, -2.1,  2.8,
            -1.5,  2.5,  0.8,
             8.4,  1.5, -3.1,
            -5.8,  0.0,  2.5,
            -4.9, -2.1, -1.5
        ];

        #[rustfmt::skip]
        let y_test: Vec<f32> = vec![
             -4.3, -0.7,  3.7,
              3.6,  1.0, -3.7,
              2.6,  3.3,  1.4,
              1.7, -4.0,  2.6,
              0.0,  0.0,  0.0,
             -5.7, -1.3,  2.5,
            -13.0, -0.9,  5.3,
              8.2, -1.4,  0.6
        ];

        let mut y = vec![0.0f32; 24];

        // Multiply and run test
        a.mul_f32(&x, 12, 3, &mut y, 8, 3).unwrap();

        for i in 0..24 {
            assert_abs_diff_eq!(y[i], y_test[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_smatrixb_vmulf)]
    fn test_smatrixb_vmulf() {
        let tol = 1e-6f32;

        // Create sparse matrix and set values
        let mut a = SparseMatrix::<u8>::new(8, 12).unwrap();
        a.set(0, 0, 1);
        a.set(2, 0, 1);
        a.set(6, 0, 1);
        a.set(3, 2, 1);
        a.set(6, 2, 1);
        a.set(7, 2, 1);
        a.set(1, 3, 1);
        a.set(7, 5, 1);
        a.set(3, 6, 1);
        a.set(5, 6, 1);
        a.set(7, 6, 1);
        a.set(3, 7, 1);
        a.set(2, 8, 1);
        a.set(5, 8, 1);
        a.set(2, 9, 1);
        a.set(5, 10, 1);
        a.set(6, 10, 1);
        a.set(6, 11, 1);

        // Generate vectors
        #[rustfmt::skip]
        let x: Vec<f32> = vec![
            3.4, -5.7,  0.3, 2.3,  1.9,  3.9,
            2.3, -4.0, -0.5, 1.5, -0.6, -1.0
        ];

        let y_test: Vec<f32> = vec![3.4, 2.3, 4.4, -1.4, 0.0, 1.2, 2.1, 6.5];

        let mut y = vec![0.0f32; 8];

        // Multiply and run test
        a.vmul_f32(&x, &mut y);

        for i in 0..8 {
            assert_abs_diff_eq!(y[i], y_test[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_smatrixi_vmul)]
    fn test_smatrixi_vmul() {
        // A = [
        //  0 0 0 0 4
        //  0 0 0 0 0
        //  0 0 0 3 0
        //  2 0 0 0 1

        // create sparse matrix and set values
        let mut a = SparseMatrix::<i16>::new(4, 5).unwrap();
        a.set(0, 4, 4);
        a.set(2, 3, 3);
        a.set(3, 0, 2);
        a.set(3, 4, 0);
        a.set(3, 4, 1);

        // initialize input vector
        let x = [7, 1, 5, 2, 2];

        let y_test = [8, 0, 6, 16];
        let mut y = [0; 4];

        // multiply and run test
        a.vmul(&x, &mut y);

        // check values
        assert_eq!(y[0], y_test[0]);
        assert_eq!(y[1], y_test[1]);
        assert_eq!(y[2], y_test[2]);
        assert_eq!(y[3], y_test[3]);
    }

    #[test]
    #[autotest_annotate(autotest_smatrixi_mul)]
    fn test_smatrixi_mul() {
        // initialize matrices
        let mut a = SparseMatrix::<i16>::new(4, 5).unwrap();
        let mut b = SparseMatrix::<i16>::new(5, 3).unwrap();
        let mut c = SparseMatrix::<i16>::new(4, 3).unwrap();

        // initialize 'a'
        // 0 0 0 0 4
        // 0 0 0 0 0
        // 0 0 0 3 0
        // 2 0 0 0 1
        a.set(0, 4, 4);
        a.set(2, 3, 3);
        a.set(3, 0, 2);
        a.set(3, 4, 0);
        a.set(3, 4, 1);

        // initialize 'b'
        // 7 6 0
        // 0 0 0
        // 0 0 0
        // 0 5 0
        // 2 0 0
        b.set(0, 0, 7);
        b.set(0, 1, 6);
        b.set(3, 1, 5);
        b.set(4, 0, 2);

        // compute 'c'
        //  8   0   0
        //  0   0   0
        //  0  15   0
        // 16  12   0
        a.mul(&b, &mut c).unwrap();
        #[rustfmt::skip]
        let c_test = [
             8,   0,   0,
             0,   0,   0,
             0,  15,   0,
            16,  12,   0
        ];

        // check values
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(c.get(i, j), c_test[i * 3 + j]);
            }
        }
    }
}
