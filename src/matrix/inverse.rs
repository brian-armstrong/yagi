use crate::error::{Error, Result};
use crate::matrix::view::MatrixMut;
use crate::matrix::FloatComplex;
use num_traits::Zero;

/// Compute matrix inverse
pub fn matrix_inv<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (xr, xc) = x.shape();
    // Ensure lengths are valid
    if xr != xc {
        return Err(Error::Range("matrix_inv(), invalid dimensions".to_string()));
    }

    let input = x.as_ref();
    let augmented_columns = xc
        .checked_mul(2)
        .ok_or_else(|| Error::Range("matrix_inv(), augmented matrix dimensions overflow usize".to_string()))?;
    let augmented_len = xr
        .checked_mul(augmented_columns)
        .ok_or_else(|| Error::Range("matrix_inv(), augmented matrix dimensions overflow usize".to_string()))?;

    // Allocate temporary memory
    let mut x_aug = vec![T::zero(); augmented_len];

    // Initialize augmented matrix
    let mut augmented = MatrixMut::new(&mut x_aug, xr, augmented_columns)?;
    for (row_index, input_row) in input.rows().enumerate() {
        let augmented_row = augmented.row_mut(row_index);
        augmented_row[..xc].copy_from_slice(input_row);
        augmented_row[xc..].fill(T::zero());
        augmented_row[xc + row_index] = T::one();
    }

    // Perform Gauss-Jordan elimination on x_aug
    matrix_gjelim(&mut augmented)?;

    // Copy result from right half of x_aug
    for row_index in 0..xr {
        x.row_mut(row_index).copy_from_slice(&augmented.row(row_index)[xc..]);
    }

    Ok(())
}

/// Gauss-Jordan elimination
pub fn matrix_gjelim<T>(x: &mut MatrixMut<'_, T>) -> Result<()>
where
    T: FloatComplex,
{
    let (xr, xc) = x.shape();
    if xr > xc {
        return Err(Error::Range("matrix_gjelim(), matrix must have at least as many columns as rows".to_string()));
    }

    // Choose pivot rows based on maximum element along column
    for r in 0..xr {
        let mut v_max = T::Real::zero();
        let mut r_opt = r;

        for r_hat in r..xr {
            let v = x.row(r_hat)[r].abs();
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
            x.swap_rows(r, r_opt)?;
        }

        // Pivot on the diagonal element
        matrix_pivot(x, r, r)?;
    }

    // Scale by diagonal
    for r in 0..xr {
        let row = x.row_mut(r);
        let g = T::one() / row[r];
        for value in row {
            *value *= g;
        }
    }

    Ok(())
}

/// Pivot on element (r, c)
fn matrix_pivot<T>(matrix: &mut MatrixMut<'_, T>, r: usize, c: usize) -> Result<()>
where
    T: FloatComplex,
{
    let v = matrix.row(r)[c];
    if v == T::zero() {
        return Err(Error::Value("matrix_pivot(), pivoting on zero".to_string()));
    }

    // Pivot using back-substitution
    matrix.for_each_other_row_mut(r, |pivot_row, target_row| {
        // Compute multiplier
        let g = target_row[c] / v;

        // Back-substitution
        for (target_value, &pivot_value) in target_row.iter_mut().zip(pivot_row) {
            *target_value = g * pivot_value - *target_value;
        }
    })?;

    Ok(())
}
