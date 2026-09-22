use num_traits::{Float, One, Zero};

use crate::error::{Error, Result};
use crate::matrix::{ColMatrixMut, ColMatrixRef, FloatComplex};

/// Reusable least-squares workspace for a tall, full-column-rank matrix.
///
/// Holds a compact Householder QR of the design matrix `A` (`rows` × `columns`).
/// We factor `A` at construction time and then reuse the reflectors so that
/// `solve` will not do redundant work.
///
/// The `factors` field serves two purposes after construction:
/// - the strictly-upper triangle of column `j` holds all of `R` except its diagonal entries
/// - the lower triangle of column `j` holds the reflector vector `v_j`
///
/// For example, a 6 × 4 `A`:
///
/// ```text
///           col 0     col 1     col 2     col 3
///         +--------------------------------------+
///  row 0  | v0[0]     R[0][1]   R[0][2]   R[0][3] |
///  row 1  | v0[1]     v1[0]     R[1][2]   R[1][3] |
///  row 2  | v0[2]     v1[1]     v2[0]     R[2][3] |
///  row 3  | v0[3]     v1[2]     v2[1]     v3[0]   |
///  row 4  | v0[4]     v1[3]     v2[2]     v3[1]   |
///  row 5  | v0[5]     v1[4]     v2[3]     v3[2]   |
///         +--------------------------------------+
///
///  r_diagonal = [R[0][0], R[1][1], R[2][2], R[3][3]]
/// ```
///
/// Because both sides want the diagonal storage, we spill R into r_diagonal.
/// We only need R during back-substitution, so we retrieve the diagonal there.
pub(crate) struct LeastSquaresWorkspace<T>
where
    T: FloatComplex,
{
    rows: usize,
    columns: usize,
    factors: Vec<T>,
    r_diagonal: Vec<T>,
    reflector_scales: Vec<T::Real>,
    scratch: Vec<T>,
}

impl<T> LeastSquaresWorkspace<T>
where
    T: FloatComplex,
{
    /// Create a new reusable workspace from a column-major design matrix.
    pub(crate) fn from_design_matrix(matrix: ColMatrixRef<T>) -> Result<Self> {
        let (rows, columns) = matrix.shape();
        if columns == 0 {
            return Err(Error::Range("least-squares matrix must have at least one column".to_owned()));
        }
        if rows < columns {
            return Err(Error::Range("least-squares matrix must have at least as many rows as columns".to_owned()));
        }
        if matrix.columns().any(|column| column.iter().any(|value| !value.abs().is_finite())) {
            return Err(Error::Value("least-squares matrix must contain only finite values".to_owned()));
        }

        let scratch_len = rows
            .checked_add(columns)
            .ok_or_else(|| Error::Range("least-squares workspace size overflows usize".to_owned()))?;

        // create the storage space for `factors`
        // this will start as a copy of `matrix` but get factorized in place
        let mut factor_storage = matrix.as_slice().to_vec();
        let mut factors = ColMatrixMut::new(&mut factor_storage, rows, columns)?;

        let two = T::Real::one() + T::Real::one();
        let mut r_diagonal = vec![T::zero(); columns];
        let mut reflector_scales = vec![T::Real::zero(); columns];
        let mut largest_diagonal = T::Real::zero();

        // reduce A to upper-triangular R one column at a time. for each column, we
        // build a Householder reflector H = I - beta v v^H. this reflector maps
        // the active part of the column (at the diagonal and below) to
        // diagonal*e_0, zeroing everything below the diagonal:
        //
        //   v = x - diagonal*e_0,   beta = 2 / (v^H v),   H x = x - beta v (v^H x)
        //
        // once these reflectors are calculated, they are applied to the trailing
        // unfactored columns. they are then stored in place and replayed in solve()
        // to form Q^H rhs.
        for column in 0..columns {
            let active_column = &factors.column(column)[column..];
            let norm = Self::scaled_norm(active_column);
            if !norm.is_finite() || norm == T::Real::zero() {
                return Err(Error::Value("least-squares matrix is rank deficient".to_owned()));
            }

            // construct the diagonal from the first element of this column
            let leading = active_column[0];
            let leading_magnitude = leading.abs();
            let phase = if leading_magnitude == T::Real::zero() {
                T::one()
            } else {
                leading / T::from(leading_magnitude).unwrap()
            };

            // diagonal entry of R for this column, -phase*||x||. we invert the sign
            // of the leading element and minimize cancellation in v.
            let diagonal = -phase * T::from(norm).unwrap();

            // form the complete reflector v_j in the active part of this column
            let factor_column = &mut factors.column_mut(column)[column..];
            factor_column[0] -= diagonal;
            if !factor_column[0].abs().is_finite() || factor_column[0].abs() == T::Real::zero() {
                return Err(Error::Value("least-squares factorization is not finite".to_owned()));
            }

            let mut reflector_norm_squared = factor_column.iter().fold(T::Real::zero(), |sum, value| {
                let magnitude = value.abs();
                sum + magnitude * magnitude
            });

            if !reflector_norm_squared.is_finite() || reflector_norm_squared == T::Real::zero() {
                // rescale the reflector if directly summing its squares overflowed or underflowed
                let scale = factor_column.iter().fold(T::Real::zero(), |scale, value| scale.max(value.abs()));
                if !scale.is_finite() || scale == T::Real::zero() {
                    return Err(Error::Value("least-squares factorization is not finite".to_owned()));
                }
                let scale = T::from(scale).unwrap();
                for value in factor_column.iter_mut() {
                    *value = *value / scale;
                }
                reflector_norm_squared = factor_column.iter().fold(T::Real::zero(), |sum, value| {
                    let magnitude = value.abs();
                    sum + magnitude * magnitude
                });
            }

            if !reflector_norm_squared.is_finite() || reflector_norm_squared == T::Real::zero() {
                return Err(Error::Value("least-squares factorization is not finite".to_owned()));
            }

            // compute beta = 2 / (v^H v)
            let reflector_scale = two / reflector_norm_squared;
            // R's diagonal lives out-of-band so the cell can keep v[0]
            r_diagonal[column] = diagonal;
            reflector_scales[column] = reflector_scale;
            largest_diagonal = largest_diagonal.max(diagonal.abs());

            // apply the transform constructed from this reflector to the unreduced columns.
            // after we apply every transform (from each column), we will have constructed R
            // in `factors`
            let reflector_scale = T::from(reflector_scale).unwrap();
            for trailing_index in column + 1..columns {
                let (reflector_column, trailing_column) = factors
                    .two_columns_mut(column, trailing_index)
                    .expect("distinct Householder columns are in bounds");
                let reflector = &reflector_column[column..];
                let trailing_column = &mut trailing_column[column..];
                // coefficient = beta * (v^H * trailing_column)
                let coefficient = reflector
                    .iter()
                    .zip(trailing_column.iter())
                    .map(|(&reflector_value, &matrix_value)| reflector_value.conj() * matrix_value)
                    .sum::<T>()
                    * reflector_scale;
                // H x = x - beta v (v^H x)
                for (matrix_value, &reflector_value) in trailing_column.iter_mut().zip(reflector) {
                    *matrix_value -= reflector_value * coefficient;
                }
            }
        }

        if factors.as_ref().columns().any(|column| column.iter().any(|value| !value.abs().is_finite())) {
            return Err(Error::Value("least-squares factorization is not finite".to_owned()));
        }

        // reject if any R diagonal is within rounding of zero relative to the
        // largest pivot, eps * max(rows, columns) * largest_diagonal
        let dimension_scale = <T::Real as num_traits::NumCast>::from(rows.max(columns)).unwrap();
        let rank_tolerance = T::Real::epsilon() * dimension_scale * largest_diagonal;
        if r_diagonal.iter().any(|diagonal| diagonal.abs() <= rank_tolerance) {
            return Err(Error::Value("least-squares matrix is numerically rank deficient".to_owned()));
        }

        Ok(Self {
            rows,
            columns,
            factors: factor_storage,
            r_diagonal,
            reflector_scales,
            scratch: vec![T::zero(); scratch_len],
        })
    }

    /// Solve the least-squares problem `min ||A*x - rhs||` using this factorization.
    ///
    /// `solution` is left unchanged if any step produces a non-finite value.
    pub(crate) fn solve(&mut self, rhs: &[T], solution: &mut [T]) -> Result<()> {
        // two phases
        // - form `Q^H * rhs` by replaying the stored reflectors in order
        // - back-substitute `R x = Q^H * rhs`

        if rhs.len() != self.rows || solution.len() != self.columns {
            return Err(Error::Range(
                "least-squares right-hand side and solution lengths must match the matrix shape".to_owned(),
            ));
        }
        if rhs.iter().any(|value| !value.abs().is_finite()) {
            return Err(Error::Value("least-squares right-hand side must contain only finite values".to_owned()));
        }

        let (transformed_rhs, candidate) = self.scratch.split_at_mut(self.rows);
        transformed_rhs.copy_from_slice(rhs);

        let factors = ColMatrixRef::new(&self.factors, self.rows, self.columns)?;

        // phase 1: apply Q^H = H_{columns-1} ... H_1 H_0 to the scratch rhs in place
        for column in 0..self.columns {
            let reflector = &factors.column(column)[column..];
            // coefficient = beta * (v^H * rhs)
            let coefficient = reflector
                .iter()
                .zip(&transformed_rhs[column..])
                .map(|(&reflector_value, &rhs_value)| reflector_value.conj() * rhs_value)
                .sum::<T>()
                * T::from(self.reflector_scales[column]).unwrap();
            // H x = x - beta v (v^H x)
            for (rhs_value, &reflector_value) in transformed_rhs[column..].iter_mut().zip(reflector) {
                *rhs_value -= reflector_value * coefficient;
            }
        }

        if transformed_rhs.iter().any(|value| !value.abs().is_finite()) {
            return Err(Error::Value("least-squares transformed right-hand side is not finite".to_owned()));
        }

        // phase 2: back-substitute R x = Q^H * rhs
        for row in (0..self.columns).rev() {
            let mut value = transformed_rhs[row];
            for column in row + 1..self.columns {
                value -= factors[(row, column)] * candidate[column];
            }
            candidate[row] = value / self.r_diagonal[row];
            if !candidate[row].abs().is_finite() {
                return Err(Error::Value("least-squares solution is not finite".to_owned()));
            }
        }

        solution.copy_from_slice(candidate);
        Ok(())
    }

    fn scaled_norm(values: &[T]) -> T::Real {
        // find the norm of a vector via running rescale
        let mut scale = T::Real::zero();
        let mut sum = T::Real::one();
        for value in values {
            let magnitude = value.abs();
            if magnitude != T::Real::zero() {
                if scale < magnitude {
                    // new largest magnitude found: update the scale and rescale the sum
                    let ratio = scale / magnitude;
                    sum = T::Real::one() + sum * ratio * ratio;
                    scale = magnitude;
                } else {
                    // current magnitude smaller than scale: just update the sum
                    let ratio = magnitude / scale;
                    sum = sum + ratio * ratio;
                }
            }
        }
        // resolve norm from scale and sum
        scale * sum.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::LeastSquaresWorkspace;
    use crate::matrix::{ColMatrixRef, FloatComplex};
    use approx::assert_abs_diff_eq;
    use num_complex::Complex64;
    use num_traits::Zero;

    fn to_column_major<T: Copy>(matrix: &[T], rows: usize, columns: usize) -> Vec<T> {
        (0..columns).flat_map(|column| (0..rows).map(move |row| matrix[row * columns + column])).collect()
    }

    fn factorize<T: FloatComplex>(matrix: &[T], rows: usize, columns: usize) -> LeastSquaresWorkspace<T> {
        let column_major = to_column_major(matrix, rows, columns);
        let view = ColMatrixRef::new(&column_major, rows, columns).unwrap();
        LeastSquaresWorkspace::from_design_matrix(view).unwrap()
    }

    fn matrix_vector_mul<T: FloatComplex>(matrix: &[T], rows: usize, columns: usize, vector: &[T]) -> Vec<T> {
        matrix
            .chunks_exact(columns)
            .take(rows)
            .map(|row| row.iter().zip(vector).map(|(&a, &x)| a * x).sum())
            .collect()
    }

    fn reconstruct<T: FloatComplex>(factorization: &LeastSquaresWorkspace<T>) -> (Vec<T>, Vec<T>) {
        let rows = factorization.rows;
        let columns = factorization.columns;
        let factors = ColMatrixRef::new(&factorization.factors, rows, columns).unwrap();
        let mut q = vec![T::zero(); rows * rows];
        for row in 0..rows {
            q[row * rows + row] = T::one();
        }

        // Q = H_0 H_1 ... H_n
        for column in 0..columns {
            let reflector = &factors.column(column)[column..];
            let scale = T::from(factorization.reflector_scales[column]).unwrap();
            for row in 0..rows {
                let mut coefficient = T::zero();
                for (offset, &reflector_value) in reflector.iter().enumerate() {
                    coefficient += q[row * rows + column + offset] * reflector_value;
                }
                coefficient *= scale;
                for (offset, &reflector_value) in reflector.iter().enumerate() {
                    q[row * rows + column + offset] -= coefficient * reflector_value.conj();
                }
            }
        }

        // R's diagonal lives out-of-band
        let r = |inner: usize, column: usize| {
            if inner == column {
                factorization.r_diagonal[column]
            } else {
                factors[(inner, column)]
            }
        };
        let mut reconstructed = vec![T::zero(); rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                reconstructed[row * columns + column] =
                    (0..=column).map(|inner| q[row * rows + inner] * r(inner, column)).sum();
            }
        }
        (q, reconstructed)
    }

    #[test]
    fn test_householder_qr_real_rectangular_reconstruction_and_solve() {
        #[rustfmt::skip]
        let matrix = [
            1.0f64,  2.0,  0.0,
               0.0,  1.0,  1.0,
               2.0, -1.0,  1.0,
               1.0,  0.0, -1.0,
               3.0,  1.0,  2.0,
        ];
        let expected = [0.5, -1.25, 2.0];
        let rhs = matrix_vector_mul(&matrix, 5, 3, &expected);
        let mut factorization = factorize(&matrix, 5, 3);
        let (q, reconstructed) = reconstruct(&factorization);

        for (&actual, &expected) in reconstructed.iter().zip(&matrix) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
        }
        for first in 0..5 {
            for second in 0..5 {
                let product: f64 = (0..5).map(|row| q[row * 5 + first] * q[row * 5 + second]).sum();
                assert_abs_diff_eq!(product, if first == second { 1.0 } else { 0.0 }, epsilon = 1e-12);
            }
        }

        let mut solution = [0.0; 3];
        factorization.solve(&rhs, &mut solution).unwrap();
        for (&actual, &expected) in solution.iter().zip(&expected) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_householder_qr_complex_rectangular_reconstruction_and_solve() {
        let matrix = [
            Complex64::new(1.0, 0.5),
            Complex64::new(-0.25, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(2.0, 0.25),
            Complex64::new(1.5, -0.5),
            Complex64::new(0.75, 0.5),
            Complex64::new(-1.0, 0.25),
            Complex64::new(0.5, -1.5),
        ];
        let expected = [Complex64::new(0.5, -0.75), Complex64::new(-1.25, 0.25)];
        let rhs = matrix_vector_mul(&matrix, 4, 2, &expected);
        let mut factorization = factorize(&matrix, 4, 2);
        let (q, reconstructed) = reconstruct(&factorization);

        for (&actual, &expected) in reconstructed.iter().zip(&matrix) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-12);
        }
        for first in 0..4 {
            for second in 0..4 {
                let product: Complex64 = (0..4).map(|row| q[row * 4 + first].conj() * q[row * 4 + second]).sum();
                let expected = if first == second { Complex64::new(1.0, 0.0) } else { Complex64::zero() };
                assert_abs_diff_eq!(product.re, expected.re, epsilon = 1e-12);
                assert_abs_diff_eq!(product.im, expected.im, epsilon = 1e-12);
            }
        }

        let mut solution = [Complex64::zero(); 2];
        factorization.solve(&rhs, &mut solution).unwrap();
        for (&actual, &expected) in solution.iter().zip(&expected) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_householder_qr_least_squares_residual_is_orthogonal() {
        let matrix = [1.0f64, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let rhs = [4.1, 2.0, 0.2, -1.9, -3.8];
        let mut factorization = factorize(&matrix, 5, 2);
        let mut solution = [0.0; 2];
        factorization.solve(&rhs, &mut solution).unwrap();
        let fitted = matrix_vector_mul(&matrix, 5, 2, &solution);
        let residual: Vec<_> = rhs.iter().zip(fitted).map(|(&observed, fitted)| observed - fitted).collect();

        for column in 0..2 {
            let dot: f64 = (0..5).map(|row| matrix[row * 2 + column] * residual[row]).sum();
            assert_abs_diff_eq!(dot, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_least_squares_workspace_reuses_factorization_and_scratch() {
        let matrix = [1.0f64, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let mut factorization = factorize(&matrix, 4, 2);
        let mut solution = [0.0; 2];

        for expected in [[2.0, -0.5], [-1.0, 3.0]] {
            let rhs = matrix_vector_mul(&matrix, 4, 2, &expected);
            factorization.solve(&rhs, &mut solution).unwrap();
            for (&actual, &expected) in solution.iter().zip(&expected) {
                assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_least_squares_workspace_rejects_invalid_or_rank_deficient_matrices() {
        // a rejection may come from either the view (bad shape) or the factorization
        fn rejects(matrix: &[f64], rows: usize, columns: usize) -> bool {
            ColMatrixRef::new(matrix, rows, columns)
                .and_then(LeastSquaresWorkspace::from_design_matrix)
                .is_err()
        }

        assert!(rejects(&[], 3, 0));
        assert!(rejects(&[0.0; 6], 2, 3));
        assert!(rejects(&[0.0; 3], 2, 2));
        assert!(rejects(&[1.0, f64::NAN, 2.0, 3.0], 2, 2));

        let rank_deficient = [1.0f64, 2.0, 2.0, 4.0, 3.0, 6.0];
        let rank_deficient = to_column_major(&rank_deficient, 3, 2);
        assert!(rejects(&rank_deficient, 3, 2));

        let nearly_dependent = [1.0f64, 1.0, 1.0, 1.0 + f64::EPSILON, 1.0, 1.0 - f64::EPSILON];
        let nearly_dependent = to_column_major(&nearly_dependent, 3, 2);
        assert!(rejects(&nearly_dependent, 3, 2));
    }

    #[test]
    fn test_least_squares_workspace_solve_errors_preserve_output() {
        let matrix = [1.0f64, 0.0, 1.0, 1.0, 1.0, 2.0];
        let mut factorization = factorize(&matrix, 3, 2);
        let original = [42.0, -7.0];

        let mut solution = original;
        assert!(factorization.solve(&[1.0, 2.0], &mut solution).is_err());
        assert_eq!(solution, original);

        let mut solution = original;
        assert!(factorization.solve(&[1.0, f64::NAN, 3.0], &mut solution).is_err());
        assert_eq!(solution, original);
    }
}
