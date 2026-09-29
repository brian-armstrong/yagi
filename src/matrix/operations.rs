use crate::error::{Error, Result};
use crate::matrix::{matrix_inv, matrix_ludecomp_doolittle, FloatComplex, MatrixMut, MatrixRef};

/// Computes `z = x + y`, element by element.
///
/// # Errors
///
/// Returns an error when `x`, `y` and `z` do not all have the same shape. `z`
/// is unchanged on error.
pub fn matrix_add<T>(x: MatrixRef<'_, T>, y: MatrixRef<'_, T>, z: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    if x.shape() != y.shape() || x.shape() != z.shape() {
        return Err(Error::Range("matrix_add(), input and output shapes must match".to_string()));
    }
    for ((output, &left), &right) in z.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
        *output = left + right;
    }
    Ok(())
}

/// Computes `z = x - y`, element by element.
///
/// # Errors
///
/// Returns an error when `x`, `y` and `z` do not all have the same shape. `z`
/// is unchanged on error.
pub fn matrix_sub<T>(x: MatrixRef<'_, T>, y: MatrixRef<'_, T>, z: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    if x.shape() != y.shape() || x.shape() != z.shape() {
        return Err(Error::Range("matrix_sub(), input and output shapes must match".to_string()));
    }
    for ((output, &left), &right) in z.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
        *output = left - right;
    }
    Ok(())
}

/// Computes the element-by-element product of `x` and `y` into `z`.
///
/// # Errors
///
/// Returns an error when `x`, `y` and `z` do not all have the same shape. `z`
/// is unchanged on error.
pub fn matrix_pmul<T>(x: MatrixRef<'_, T>, y: MatrixRef<'_, T>, z: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    if x.shape() != y.shape() || x.shape() != z.shape() {
        return Err(Error::Range("matrix_pmul(), input and output shapes must match".to_string()));
    }
    for ((output, &left), &right) in z.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
        *output = left * right;
    }
    Ok(())
}

/// Computes the element-by-element quotient of `x` and `y` into `z`.
///
/// # Errors
///
/// Returns an error when `x`, `y` and `z` do not all have the same shape. `z`
/// is unchanged on error.
pub fn matrix_pdiv<T>(x: MatrixRef<'_, T>, y: MatrixRef<'_, T>, z: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    if x.shape() != y.shape() || x.shape() != z.shape() {
        return Err(Error::Range("matrix_pdiv(), input and output shapes must match".to_string()));
    }
    for ((output, &left), &right) in z.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
        *output = left / right;
    }
    Ok(())
}

/// Computes the matrix product `z = x * y`.
///
/// The column count of `x` must equal the row count of `y`. `z` must have the
/// row count of `x` and the column count of `y`.
///
/// # Errors
///
/// Returns an error when the shapes do not fit those rules. `z` is unchanged
/// on error.
///
/// ```
/// use yagi::matrix::{matrix_mul, MatrixMut, MatrixRef};
///
/// let left = [1.0f32, 2.0, 3.0, 4.0];
/// let right = [5.0f32, 6.0, 7.0, 8.0];
/// let mut product = [0.0f32; 4];
/// let mut product = MatrixMut::new(&mut product, 2, 2).unwrap();
///
/// matrix_mul(
///     MatrixRef::new(&left, 2, 2).unwrap(),
///     MatrixRef::new(&right, 2, 2).unwrap(),
///     &mut product,
/// )
/// .unwrap();
/// assert_eq!(product.as_slice(), [19.0, 22.0, 43.0, 50.0]);
/// ```
pub fn matrix_mul<T>(x: MatrixRef<'_, T>, y: MatrixRef<'_, T>, z: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (xr, xc) = x.shape();
    let (yr, yc) = y.shape();
    if z.shape() != (xr, yc) || xc != yr {
        return Err(Error::Range("matrix_mul(), invalid dimensions".to_string()));
    }

    for (x_row, z_row) in x.rows().zip(z.rows_mut()) {
        z_row.fill(T::default());
        for (&x_value, y_row) in x_row.iter().zip(y.rows()) {
            for (z_value, &y_value) in z_row.iter_mut().zip(y_row) {
                *z_value += x_value * y_value;
            }
        }
    }

    Ok(())
}

/// Multiplies a matrix by a column vector.
///
/// `vector` must have one element per column of `matrix`, and `output` one
/// element per row.
///
/// # Errors
///
/// Returns an error when either length does not match. `output` is unchanged
/// on error.
pub fn matrix_mul_vec<T>(matrix: MatrixRef<'_, T>, vector: &[T], output: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let (rows, cols) = matrix.shape();
    if vector.len() != cols || output.len() != rows {
        return Err(Error::Range("matrix_mul_vec(), vector dimensions do not match the matrix".to_string()));
    }

    for (matrix_row, output_value) in matrix.rows().zip(output.iter_mut()) {
        let mut sum = T::default();
        for (&matrix_value, &vector_value) in matrix_row.iter().zip(vector) {
            sum += matrix_value * vector_value;
        }
        *output_value = sum;
    }

    Ok(())
}

/// Places `x` and `y` side by side in `z`.
///
/// `x` and `y` must have the same row count. `z` must have that row count and
/// as many columns as `x` and `y` together.
///
/// # Errors
///
/// Returns an error when the shapes do not fit those rules, including when
/// the combined column count overflows `usize`. `z` is unchanged on error.
pub fn matrix_aug<T>(x: MatrixRef<'_, T>, y: MatrixRef<'_, T>, z: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (rx, cx) = x.shape();
    let (ry, cy) = y.shape();
    let expected_columns = cx
        .checked_add(cy)
        .ok_or_else(|| Error::Range("matrix_aug(), column count overflows usize".to_string()))?;
    if rx != ry || z.shape() != (rx, expected_columns) {
        return Err(Error::Range("matrix_aug(), invalid dimensions".to_string()));
    }
    for ((x_row, y_row), z_row) in x.rows().zip(y.rows()).zip(z.rows_mut()) {
        let (z_left, z_right) = z_row.split_at_mut(cx);
        z_left.copy_from_slice(x_row);
        z_right.copy_from_slice(y_row);
    }

    Ok(())
}

/// Computes `z = x * inverse(y)`.
///
/// All three matrices must be square with the same size. The inverse is formed
/// explicitly with [`matrix_inv`], so its limits apply here too.
///
/// # Errors
///
/// Returns an error when the shapes differ or are not square, or when
/// [`matrix_inv`] fails on `y`. `z` is unchanged on error.
pub fn matrix_div<T>(x: MatrixRef<'_, T>, y: MatrixRef<'_, T>, z: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (n, columns) = x.shape();
    if n != columns || y.shape() != (n, n) || z.shape() != (n, n) {
        return Err(Error::Range("matrix_div(), all matrices must have the same square shape".to_string()));
    }
    let mut y_inv = y.as_slice().to_vec();
    let mut y_inv = MatrixMut::new(&mut y_inv, n, n)?;
    matrix_inv(&mut y_inv)?;

    matrix_mul(x, y_inv.as_ref(), z)
}

/// Returns the determinant of a 2 by 2 matrix.
///
/// # Errors
///
/// Returns an error when `x` is not 2 by 2.
pub fn matrix_det2x2<T>(x: MatrixRef<'_, T>) -> Result<T>
where
    T: FloatComplex,
{
    if x.shape() != (2, 2) {
        return Err(Error::Range("matrix_det2x2(), invalid dimensions".to_string()));
    }

    Ok(x[(0, 0)] * x[(1, 1)] - x[(0, 1)] * x[(1, 0)])
}

/// Returns the determinant of a square matrix.
///
/// A 2 by 2 matrix uses the direct formula. Larger matrices use the product of
/// the diagonal of an unpivoted LU factorization, as in
/// [`matrix_ludecomp_doolittle`]. An empty matrix gives one.
///
/// # Warning
///
/// A zero pivot before the last one makes the result infinite or NaN with no
/// error, and many nonsingular matrices have one. Every matrix of 3 by 3 or
/// larger with a zero in the top-left corner is affected, and small pivots
/// lose accuracy in the same way. A zero final pivot correctly gives zero, as
/// for `diag(1, 1, 0)`.
///
/// # Errors
///
/// Returns an error when `x` is not square.
pub fn matrix_det<T>(x: MatrixRef<'_, T>) -> Result<T>
where
    T: FloatComplex,
{
    let (n, c) = x.shape();
    if n != c {
        return Err(Error::Range("matrix_det(), matrix must be square".to_string()));
    }

    if n == 2 {
        return matrix_det2x2(x);
    }

    let square_len = n
        .checked_mul(n)
        .ok_or_else(|| Error::Range("matrix_det(), matrix dimensions overflow usize".to_string()))?;

    // Compute L/U decomposition (Doolittle's method)
    let mut l = vec![T::default(); square_len];
    let mut u = vec![T::default(); square_len];
    let mut p = vec![T::default(); square_len];
    let mut l_view = MatrixMut::new(&mut l, n, n)?;
    let mut u_view = MatrixMut::new(&mut u, n, n)?;
    let mut p_view = MatrixMut::new(&mut p, n, n)?;
    matrix_ludecomp_doolittle(x, &mut l_view, &mut u_view, &mut p_view)?;

    // Evaluate along the diagonal of U
    let mut det = T::one();
    for (row_index, row) in u_view.as_ref().rows().enumerate() {
        det *= row[row_index];
    }

    Ok(det)
}

/// Overwrites a matrix with its transpose.
///
/// Complex values are not conjugated. The view keeps its original shape, so
/// for a rectangular matrix the caller must read the storage back with the
/// row and column counts swapped. This never returns an error.
pub fn matrix_trans<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (rows, columns) = x.shape();
    let y = x.as_slice().to_vec();
    let input = MatrixRef::new(&y, rows, columns)?;
    let mut output = MatrixMut::new(x.as_mut_slice(), columns, rows)?;

    for (row_index, input_row) in input.rows().enumerate() {
        for (column_index, &value) in input_row.iter().enumerate() {
            output.row_mut(column_index)[row_index] = value;
        }
    }
    Ok(())
}

/// Overwrites a matrix with its conjugate transpose.
///
/// The view keeps its original shape, so for a rectangular matrix the caller
/// must read the storage back with the row and column counts swapped. This
/// never returns an error.
pub fn matrix_hermitian<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    matrix_trans(x)?;
    for value in x.as_mut_slice() {
        *value = value.conj();
    }
    Ok(())
}

/// Computes `x * x^T` into `xxt`, without conjugation.
///
/// `xxt` must be square, with one row per row of `x`.
///
/// # Errors
///
/// Returns an error when `xxt` has the wrong shape. `xxt` is unchanged on
/// error.
pub fn matrix_mul_transpose<T>(x: MatrixRef<'_, T>, xxt: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (m, _) = x.shape();
    if xxt.shape() != (m, m) {
        return Err(Error::Range("matrix_mul_transpose(), output shape must be rows by rows".to_string()));
    }

    for (x_row, xxt_row) in x.rows().zip(xxt.rows_mut()) {
        for (xxt_value, other_row) in xxt_row.iter_mut().zip(x.rows()) {
            let mut sum = T::default();
            for (&left, &right) in x_row.iter().zip(other_row) {
                sum += left * right;
            }
            *xxt_value = sum;
        }
    }
    Ok(())
}

/// Computes `x^T * x` into `xtx`, without conjugation.
///
/// `xtx` must be square, with one row per column of `x`.
///
/// # Errors
///
/// Returns an error when `xtx` has the wrong shape. `xtx` is unchanged on
/// error.
pub fn matrix_transpose_mul<T>(x: MatrixRef<'_, T>, xtx: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (_, n) = x.shape();
    if xtx.shape() != (n, n) {
        return Err(Error::Range("matrix_transpose_mul(), output shape must be columns by columns".to_string()));
    }
    xtx.as_mut_slice().fill(T::default());

    for x_row in x.rows() {
        for (row_index, &left) in x_row.iter().enumerate() {
            for (xtx_value, &right) in xtx.row_mut(row_index).iter_mut().zip(x_row) {
                *xtx_value += left * right;
            }
        }
    }
    Ok(())
}

/// Computes `x * x^H` into `xxh`.
///
/// `xxh` must be square, with one row per row of `x`.
///
/// # Errors
///
/// Returns an error when `xxh` has the wrong shape. `xxh` is unchanged on
/// error.
pub fn matrix_mul_hermitian<T>(x: MatrixRef<'_, T>, xxh: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (m, _) = x.shape();
    if xxh.shape() != (m, m) {
        return Err(Error::Range("matrix_mul_hermitian(), output shape must be rows by rows".to_string()));
    }

    for (x_row, xxh_row) in x.rows().zip(xxh.rows_mut()) {
        for (xxh_value, other_row) in xxh_row.iter_mut().zip(x.rows()) {
            let mut sum = T::default();
            for (&left, &right) in x_row.iter().zip(other_row) {
                sum += left * right.conj();
            }
            *xxh_value = sum;
        }
    }
    Ok(())
}

/// Computes `x^H * x` into `xhx`.
///
/// `xhx` must be square, with one row per column of `x`.
///
/// # Errors
///
/// Returns an error when `xhx` has the wrong shape. `xhx` is unchanged on
/// error.
pub fn matrix_hermitian_mul<T>(x: MatrixRef<'_, T>, xhx: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (_, n) = x.shape();
    if xhx.shape() != (n, n) {
        return Err(Error::Range("matrix_hermitian_mul(), output shape must be columns by columns".to_string()));
    }
    xhx.as_mut_slice().fill(T::default());

    for x_row in x.rows() {
        for (row_index, &left) in x_row.iter().enumerate() {
            for (xhx_value, &right) in xhx.row_mut(row_index).iter_mut().zip(x_row) {
                *xhx_value += left.conj() * right;
            }
        }
    }
    Ok(())
}
