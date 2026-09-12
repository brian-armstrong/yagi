use crate::error::{Error, Result};
use crate::matrix::{matrix_access, matrix_access_mut, FloatComplex};
use num_traits::Zero;

/// Compute matrix inverse
pub fn matrix_inv<T>(x: &mut [T], xr: usize, xc: usize) -> Result<()>
where
    T: FloatComplex,
{
    // Ensure lengths are valid
    if xr != xc {
        return Err(Error::Range("matrix_inv(), invalid dimensions".to_string()));
    }

    // Allocate temporary memory
    let mut x_aug = vec![T::zero(); 2 * xr * xc];
    let xc_2 = xc * 2;

    // Initialize augmented matrix
    for r in 0..xr {
        // Copy matrix elements
        for c in 0..xc {
            let v = matrix_access(x, xr, xc, r, c);
            matrix_access_mut(&mut x_aug, xr, xc_2, r, c, v);
        }
        // Append identity matrix
        for c in 0..xc {
            let v = if r == c { T::one() } else { T::zero() };
            matrix_access_mut(&mut x_aug, xr, xc_2, r, xc + c, v);
        }
    }

    // Perform Gauss-Jordan elimination on x_aug
    matrix_gjelim(&mut x_aug, xr, xc_2)?;

    // Copy result from right half of x_aug
    for r in 0..xr {
        for c in 0..xc {
            let v = matrix_access(&x_aug, xr, xc_2, r, xc + c);
            matrix_access_mut(x, xr, xc, r, c, v);
        }
    }

    Ok(())
}

/// Gauss-Jordan elimination
pub fn matrix_gjelim<T>(x: &mut [T], xr: usize, xc: usize) -> Result<()>
where
    T: FloatComplex,
{
    // Choose pivot rows based on maximum element along column
    for r in 0..xr {
        let mut v_max = T::Real::zero();
        let mut r_opt = r;

        for r_hat in r..xr {
            let v = matrix_access(x, xr, xc, r_hat, r).abs();
            if v > v_max || r_hat == r {
                r_opt = r_hat;
                v_max = v;
            }
        }

        // If the maximum is zero, matrix is singular
        if v_max == T::Real::zero() {
            return Err(Error::Value("matrix_gjelim(), matrix singular to machine precision".to_string()));
        }

        // If row does not match column (e.g. maximum value does not
        // lie on the diagonal) swap the rows
        if r != r_opt {
            matrix_swaprows(x, xr, xc, r, r_opt);
        }

        // Pivot on the diagonal element
        matrix_pivot(x, xr, xc, r, r)?;
    }

    // Scale by diagonal
    for r in 0..xr {
        let g = T::one() / matrix_access(x, xr, xc, r, r);
        for c in 0..xc {
            let v = matrix_access(x, xr, xc, r, c) * g;
            matrix_access_mut(x, xr, xc, r, c, v);
        }
    }

    Ok(())
}

/// Pivot on element (r, c)
fn matrix_pivot<T>(x: &mut [T], xr: usize, xc: usize, r: usize, c: usize) -> Result<()>
where
    T: FloatComplex,
{
    let v = matrix_access(x, xr, xc, r, c);
    if v == T::zero() {
        return Err(Error::Value("matrix_pivot(), pivoting on zero".to_string()));
    }

    // Pivot using back-substitution
    for pivot_r in 0..xr {
        if pivot_r == r {
            continue;
        }

        // Compute multiplier
        let g = matrix_access(x, xr, xc, pivot_r, c) / v;

        // Back-substitution
        for pivot_c in 0..xc {
            x[pivot_r * xc + pivot_c] =
                g * matrix_access(x, xr, xc, r, pivot_c) - matrix_access(x, xr, xc, pivot_r, pivot_c);
        }
    }

    Ok(())
}

/// Swap rows of a matrix
fn matrix_swaprows<T>(x: &mut [T], _xr: usize, xc: usize, r1: usize, r2: usize)
where
    T: FloatComplex,
{
    if r1 == r2 {
        return;
    }

    for c in 0..xc {
        x.swap(r1 * xc + c, r2 * xc + c);
    }
}
