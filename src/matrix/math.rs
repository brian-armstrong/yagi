use crate::error::{Error, Result};
use crate::matrix::{matrix_inv, matrix_ludecomp_doolittle, FloatComplex, MatrixMut, MatrixRef};

/// Add elements of two matrices
///
/// # Arguments
///
/// * `x` - 1st input matrix [size: r x c]
/// * `y` - 2nd input matrix [size: r x c]
/// * `z` - Output matrix [size: r x c]
/// * `r` - Number of rows
/// * `c` - Number of columns
pub fn matrix_add<T>(x: &[T], y: &[T], z: &mut [T], r: usize, c: usize) -> Result<()>
where
    T: FloatComplex,
{
    let x = MatrixRef::new(x, r, c)?;
    let y = MatrixRef::new(y, r, c)?;
    let mut z = MatrixMut::new(z, r, c)?;
    for ((output, &left), &right) in z.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
        *output = left + right;
    }
    Ok(())
}

/// Subtract elements of two matrices
///
/// # Arguments
///
/// * `x` - 1st input matrix [size: r x c]
/// * `y` - 2nd input matrix [size: r x c]
/// * `z` - Output matrix [size: r x c]
/// * `r` - Number of rows
/// * `c` - Number of columns
pub fn matrix_sub<T>(x: &[T], y: &[T], z: &mut [T], r: usize, c: usize) -> Result<()>
where
    T: FloatComplex,
{
    let x = MatrixRef::new(x, r, c)?;
    let y = MatrixRef::new(y, r, c)?;
    let mut z = MatrixMut::new(z, r, c)?;
    for ((output, &left), &right) in z.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
        *output = left - right;
    }
    Ok(())
}

/// Point-wise multiplication
///
/// # Arguments
///
/// * `x` - 1st input matrix [size: r x c]
/// * `y` - 2nd input matrix [size: r x c]
/// * `z` - Output matrix [size: r x c]
/// * `r` - Number of rows
/// * `c` - Number of columns
pub fn matrix_pmul<T>(x: &[T], y: &[T], z: &mut [T], r: usize, c: usize) -> Result<()>
where
    T: FloatComplex,
{
    let x = MatrixRef::new(x, r, c)?;
    let y = MatrixRef::new(y, r, c)?;
    let mut z = MatrixMut::new(z, r, c)?;
    for ((output, &left), &right) in z.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
        *output = left * right;
    }
    Ok(())
}

/// Point-wise division
///
/// # Arguments
///
/// * `x` - 1st input matrix [size: r x c]
/// * `y` - 2nd input matrix [size: r x c]
/// * `z` - Output matrix [size: r x c]
/// * `r` - Number of rows
/// * `c` - Number of columns
pub fn matrix_pdiv<T>(x: &[T], y: &[T], z: &mut [T], r: usize, c: usize) -> Result<()>
where
    T: FloatComplex,
{
    let x = MatrixRef::new(x, r, c)?;
    let y = MatrixRef::new(y, r, c)?;
    let mut z = MatrixMut::new(z, r, c)?;
    for ((output, &left), &right) in z.as_mut_slice().iter_mut().zip(x.as_slice()).zip(y.as_slice()) {
        *output = left / right;
    }
    Ok(())
}

/// Multiply two matrices together
pub fn matrix_mul<T>(
    x: &[T],
    xr: usize,
    xc: usize,
    y: &[T],
    yr: usize,
    yc: usize,
    z: &mut [T],
    zr: usize,
    zc: usize,
) -> Result<()>
where
    T: FloatComplex,
{
    if zr != xr || zc != yc || xc != yr {
        return Err(Error::Range("matrix_mul(), invalid dimensions".to_string()));
    }

    let x = MatrixRef::new(x, xr, xc)?;
    let y = MatrixRef::new(y, yr, yc)?;
    let mut z = MatrixMut::new(z, zr, zc)?;

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

/// Multiply a matrix by a column vector.
pub fn matrix_mul_vec<T>(matrix: &[T], rows: usize, cols: usize, vector: &[T], output: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let matrix = MatrixRef::new(matrix, rows, cols)?;
    let vector = MatrixRef::new(vector, cols, 1)?;
    let mut output = MatrixMut::new(output, rows, 1)?;

    for (matrix_row, output_row) in matrix.rows().zip(output.rows_mut()) {
        let mut sum = T::default();
        for (&matrix_value, &vector_value) in matrix_row.iter().zip(vector.as_slice()) {
            sum += matrix_value * vector_value;
        }
        output_row[0] = sum;
    }

    Ok(())
}

/// Augment matrices x and y: z = [x | y]
pub fn matrix_aug<T>(
    x: &[T],
    rx: usize,
    cx: usize,
    y: &[T],
    ry: usize,
    cy: usize,
    z: &mut [T],
    rz: usize,
    cz: usize,
) -> Result<()>
where
    T: FloatComplex,
{
    let expected_columns = cx
        .checked_add(cy)
        .ok_or_else(|| Error::Range("matrix_aug(), column count overflows usize".to_string()))?;
    if rz != rx || rz != ry || rx != ry || cz != expected_columns {
        return Err(Error::Range("matrix_aug(), invalid dimensions".to_string()));
    }

    let x = MatrixRef::new(x, rx, cx)?;
    let y = MatrixRef::new(y, ry, cy)?;
    let mut z = MatrixMut::new(z, rz, cz)?;
    for ((x_row, y_row), z_row) in x.rows().zip(y.rows()).zip(z.rows_mut()) {
        let (z_left, z_right) = z_row.split_at_mut(cx);
        z_left.copy_from_slice(x_row);
        z_right.copy_from_slice(y_row);
    }

    Ok(())
}

/// Right matrix division: `z = x * y^-1`.
pub fn matrix_div<T>(x: &[T], y: &[T], z: &mut [T], n: usize) -> Result<()>
where
    T: FloatComplex,
{
    let mut y_inv = y.to_vec();
    matrix_inv(&mut y_inv, n, n)?;

    matrix_mul(x, n, n, &y_inv, n, n, z, n, n)
}

/// Matrix determinant (2 x 2)
pub fn matrix_det2x2<T>(x: &[T], r: usize, c: usize) -> Result<T>
where
    T: FloatComplex,
{
    if r != 2 || c != 2 {
        return Err(Error::Range("matrix_det2x2(), invalid dimensions".to_string()));
    }
    let x = MatrixRef::new(x, r, c)?;

    Ok(x[(0, 0)] * x[(1, 1)] - x[(0, 1)] * x[(1, 0)])
}

/// Matrix determinant (n x n)
pub fn matrix_det<T>(x: &[T], r: usize, c: usize) -> Result<T>
where
    T: FloatComplex,
{
    if r != c {
        return Err(Error::Range("matrix_det(), matrix must be square".to_string()));
    }

    let n = r;
    let x = MatrixRef::new(x, n, n)?;
    if n == 2 {
        return matrix_det2x2(x.as_slice(), 2, 2);
    }

    let square_len = n
        .checked_mul(n)
        .ok_or_else(|| Error::Range("matrix_det(), matrix dimensions overflow usize".to_string()))?;

    // Compute L/U decomposition (Doolittle's method)
    let mut l = vec![T::default(); square_len];
    let mut u = vec![T::default(); square_len];
    let mut p = vec![T::default(); square_len];
    matrix_ludecomp_doolittle(x.as_slice(), n, n, &mut l, &mut u, &mut p)?;

    // Evaluate along the diagonal of U
    let mut det = T::one();
    for (row_index, row) in MatrixRef::new(&u, n, n)?.rows().enumerate() {
        det *= row[row_index];
    }

    Ok(det)
}

/// Compute matrix transpose
pub fn matrix_trans<T>(x: &mut [T], xr: usize, xc: usize) -> Result<()>
where
    T: FloatComplex,
{
    matrix_hermitian(x, xr, xc)?;

    for xi in x {
        *xi = xi.conj();
    }
    Ok(())
}

/// Compute matrix Hermitian transpose
pub fn matrix_hermitian<T>(x: &mut [T], xr: usize, xc: usize) -> Result<()>
where
    T: FloatComplex,
{
    let y = MatrixRef::new(x, xr, xc)?.as_slice().to_vec();
    let input = MatrixRef::new(&y, xr, xc)?;
    let mut output = MatrixMut::new(x, xc, xr)?;

    for (row_index, input_row) in input.rows().enumerate() {
        for (column_index, &value) in input_row.iter().enumerate() {
            output.row_mut(column_index)[row_index] = value;
        }
    }
    Ok(())
}

/// Compute x*x' on m x n matrix, result: m x m
pub fn matrix_mul_transpose<T>(x: &[T], m: usize, n: usize, xxt: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let x = MatrixRef::new(x, m, n)?;
    let mut xxt = MatrixMut::new(xxt, m, m)?;

    for (x_row, xxt_row) in x.rows().zip(xxt.rows_mut()) {
        for (xxt_value, other_row) in xxt_row.iter_mut().zip(x.rows()) {
            let mut sum = T::default();
            for (&left, &right) in x_row.iter().zip(other_row) {
                sum += left * right.conj();
            }
            *xxt_value = sum;
        }
    }
    Ok(())
}

/// Compute x'*x on m x n matrix, result: n x n
pub fn matrix_transpose_mul<T>(x: &[T], m: usize, n: usize, xtx: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let x = MatrixRef::new(x, m, n)?;
    let mut xtx = MatrixMut::new(xtx, n, n)?;
    xtx.as_mut_slice().fill(T::default());

    for x_row in x.rows() {
        for (row_index, &left) in x_row.iter().enumerate() {
            for (xtx_value, &right) in xtx.row_mut(row_index).iter_mut().zip(x_row) {
                *xtx_value += left.conj() * right;
            }
        }
    }
    Ok(())
}

/// Compute x*x.' on m x n matrix, result: m x m
pub fn matrix_mul_hermitian<T>(x: &[T], m: usize, n: usize, xxh: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let x = MatrixRef::new(x, m, n)?;
    let mut xxh = MatrixMut::new(xxh, m, m)?;

    for (x_row, xxh_row) in x.rows().zip(xxh.rows_mut()) {
        for (xxh_value, other_row) in xxh_row.iter_mut().zip(x.rows()) {
            let mut sum = T::default();
            for (&left, &right) in x_row.iter().zip(other_row) {
                sum += left * right;
            }
            *xxh_value = sum;
        }
    }
    Ok(())
}

/// Compute x.'*x on m x n matrix, result: n x n
pub fn matrix_hermitian_mul<T>(x: &[T], m: usize, n: usize, xhx: &mut [T]) -> Result<()>
where
    T: FloatComplex,
{
    let x = MatrixRef::new(x, m, n)?;
    let mut xhx = MatrixMut::new(xhx, n, n)?;
    xhx.as_mut_slice().fill(T::default());

    for x_row in x.rows() {
        for (row_index, &left) in x_row.iter().enumerate() {
            for (xhx_value, &right) in xhx.row_mut(row_index).iter_mut().zip(x_row) {
                *xhx_value += left * right;
            }
        }
    }
    Ok(())
}
