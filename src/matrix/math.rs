use crate::error::{Error, Result};
use crate::matrix::{matrix_access, matrix_inv, matrix_ludecomp_doolittle, FloatComplex};

/// Add elements of two matrices
///
/// # Arguments
///
/// * `x` - 1st input matrix [size: r x c]
/// * `y` - 2nd input matrix [size: r x c]
/// * `z` - Output matrix [size: r x c]
/// * `r` - Number of rows
/// * `c` - Number of columns
pub fn matrix_add<T>(x: &[T], y: &[T], z: &mut [T], r: usize, c: usize)
where
    T: FloatComplex,
{
    for i in 0..(r * c) {
        z[i] = x[i] + y[i];
    }
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
pub fn matrix_sub<T>(x: &[T], y: &[T], z: &mut [T], r: usize, c: usize)
where
    T: FloatComplex,
{
    for i in 0..(r * c) {
        z[i] = x[i] - y[i];
    }
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
pub fn matrix_pmul<T>(x: &[T], y: &[T], z: &mut [T], r: usize, c: usize)
where
    T: FloatComplex,
{
    for i in 0..(r * c) {
        z[i] = x[i] * y[i];
    }
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
pub fn matrix_pdiv<T>(x: &[T], y: &[T], z: &mut [T], r: usize, c: usize)
where
    T: FloatComplex,
{
    for i in 0..(r * c) {
        z[i] = x[i] / y[i];
    }
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

    for r in 0..zr {
        for c in 0..zc {
            let mut sum = T::default();
            for i in 0..xc {
                sum += matrix_access(x, xr, xc, r, i) * matrix_access(y, yr, yc, i, c);
            }
            z[r * zc + c] = sum;
        }
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
    if rz != rx || rz != ry || rx != ry || cz != cx + cy {
        return Err(Error::Range("matrix_aug(), invalid dimensions".to_string()));
    }

    for r in 0..rz {
        let mut n = 0;
        for c in 0..cx {
            z[r * cz + n] = matrix_access(x, rx, cx, r, c);
            n += 1;
        }
        for c in 0..cy {
            z[r * cz + n] = matrix_access(y, ry, cy, r, c);
            n += 1;
        }
    }

    Ok(())
}

/// Solve set of linear equations
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

    Ok(x[0] * x[3] - x[1] * x[2])
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
    if n == 2 {
        return matrix_det2x2(x, 2, 2);
    }

    // Compute L/U decomposition (Doolittle's method)
    let mut l = vec![T::default(); n * n];
    let mut u = vec![T::default(); n * n];
    let mut p = vec![T::default(); n * n];
    matrix_ludecomp_doolittle(x, n, n, &mut l, &mut u, &mut p)?;

    // Evaluate along the diagonal of U
    let mut det = T::default();
    for i in 0..n {
        det *= matrix_access(&u, n, n, i, i);
    }

    Ok(det)
}

/// Compute matrix transpose
pub fn matrix_trans<T>(x: &mut [T], xr: usize, xc: usize)
where
    T: FloatComplex,
{
    matrix_hermitian(x, xr, xc);

    for i in 0..(xr * xc) {
        x[i] = x[i].conj();
    }
}

/// Compute matrix Hermitian transpose
pub fn matrix_hermitian<T>(x: &mut [T], xr: usize, xc: usize)
where
    T: FloatComplex,
{
    let y = x.to_vec();

    for r in 0..xr {
        for c in 0..xc {
            x[c * xr + r] = y[r * xc + c];
        }
    }
}

/// Compute x*x' on m x n matrix, result: m x m
pub fn matrix_mul_transpose<T>(x: &[T], m: usize, n: usize, xxt: &mut [T])
where
    T: FloatComplex,
{
    for i in 0..m * m {
        xxt[i] = T::default();
    }

    for r in 0..m {
        for c in 0..m {
            let mut sum = T::default();
            for i in 0..n {
                let prod = matrix_access(x, m, n, r, i) * matrix_access(x, m, n, c, i).conj();
                sum += prod;
            }
            xxt[r * m + c] = sum;
        }
    }
}

/// Compute x'*x on m x n matrix, result: n x n
pub fn matrix_transpose_mul<T>(x: &[T], m: usize, n: usize, xtx: &mut [T])
where
    T: FloatComplex,
{
    for i in 0..n * n {
        xtx[i] = T::default();
    }

    for r in 0..n {
        for c in 0..n {
            let mut sum = T::default();
            for i in 0..m {
                let prod = matrix_access(x, m, n, i, r).conj() * matrix_access(x, m, n, i, c);
                sum += prod;
            }
            xtx[r * n + c] = sum;
        }
    }
}

/// Compute x*x.' on m x n matrix, result: m x m
pub fn matrix_mul_hermitian<T>(x: &[T], m: usize, n: usize, xxh: &mut [T])
where
    T: FloatComplex,
{
    for i in 0..m * m {
        xxh[i] = T::default();
    }

    for r in 0..m {
        for c in 0..m {
            let mut sum = T::default();
            for i in 0..n {
                sum += matrix_access(x, m, n, r, i) * matrix_access(x, m, n, c, i);
            }
            xxh[r * m + c] = sum;
        }
    }
}

/// Compute x.'*x on m x n matrix, result: n x n
pub fn matrix_hermitian_mul<T>(x: &[T], m: usize, n: usize, xhx: &mut [T])
where
    T: FloatComplex,
{
    for i in 0..n * n {
        xhx[i] = T::default();
    }

    for r in 0..n {
        for c in 0..n {
            let mut sum = T::default();
            for i in 0..m {
                sum += matrix_access(x, m, n, i, r) * matrix_access(x, m, n, i, c);
            }
            xhx[r * n + c] = sum;
        }
    }
}
