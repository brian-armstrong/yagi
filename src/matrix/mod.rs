// Matrix module
// Current state:
// - Autotests matching and passing
// - ready to use
// - may rethink later - struct or trait based approach may be interesting

mod base;
mod cholesky;
mod conjugate_gradient;
mod gram_schmidt;
mod householder_qr;
mod inverse;
mod linear_solve;
mod lu_decomposition;
mod math;
mod qr_decomposition;
mod sparse;
mod view;

pub use base::{matrix_eye, matrix_ones, matrix_print, matrix_zeros, FloatComplex, IsComplex};
pub use cholesky::matrix_chol;
pub use conjugate_gradient::matrix_cgsolve;
pub use gram_schmidt::{matrix_gramschmidt, matrix_proj};
pub(crate) use householder_qr::LeastSquaresWorkspace;
pub use inverse::{matrix_gjelim, matrix_inv};
pub use linear_solve::matrix_linsolve;
pub use lu_decomposition::{matrix_ludecomp_crout, matrix_ludecomp_doolittle};
pub use math::{
    matrix_add, matrix_aug, matrix_det, matrix_det2x2, matrix_div, matrix_hermitian, matrix_hermitian_mul, matrix_mul,
    matrix_mul_hermitian, matrix_mul_transpose, matrix_mul_vec, matrix_pdiv, matrix_pmul, matrix_sub, matrix_trans,
    matrix_transpose_mul,
};
pub use qr_decomposition::matrix_qrdecomp_gramschmidt;
pub use sparse::*;
pub(crate) use view::ColMatrixMut;
pub use view::{MatrixMut, MatrixRef, MatrixRowsMut};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::Complex;
    use test_macro::autotest_annotate;
    include!("test_data.rs");

    #[test]
    #[autotest_annotate(autotest_matrixf_add)]
    fn test_matrixf_add() {
        let tol = 1e-6f32;

        // x [size: 5 x 4]
        // y [size: 5 x 4]
        // z [size: 5 x 4]
        let mut z = vec![0.0f32; 20];
        matrix_add(
            MatrixRef::new(&MATRIXF_DATA_ADD_X, 5, 4).unwrap(),
            MatrixRef::new(&MATRIXF_DATA_ADD_Y, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut z, 5, 4).unwrap(),
        )
        .unwrap();

        for i in 0..20 {
            assert_abs_diff_eq!(MATRIXF_DATA_ADD_Z[i], z[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_aug)]
    fn test_matrixf_aug() {
        let tol = 1e-6f32;

        // x [size: 5 x 4]
        // y [size: 5 x 3]
        // z [size: 5 x 7]
        let mut z = vec![0.0f32; 35];
        matrix_aug(
            MatrixRef::new(&MATRIXF_DATA_AUG_X, 5, 4).unwrap(),
            MatrixRef::new(&MATRIXF_DATA_AUG_Y, 5, 3).unwrap(),
            &mut MatrixMut::new(&mut z, 5, 7).unwrap(),
        )
        .unwrap();

        for i in 0..35 {
            assert_abs_diff_eq!(MATRIXF_DATA_AUG_Z[i], z[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_cgsolve)]
    fn test_matrixf_cgsolve() {
        let tol = 0.01f32;

        // A [size: 8 x 8], symmetric positive definite matrix
        // x [size: 8 x 1]
        // b [size: 8 x 1]
        let mut x = vec![0.0f32; 8];
        matrix_cgsolve(MatrixRef::new(&MATRIXF_DATA_CGSOLVE_A, 8, 8).unwrap(), &MATRIXF_DATA_CGSOLVE_B, &mut x, None)
            .unwrap();

        for i in 0..8 {
            assert_abs_diff_eq!(MATRIXF_DATA_CGSOLVE_X[i], x[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_chol)]
    fn test_matrixf_chol() {
        let tol = 1e-4f32;

        // A [size: 4 x 4]
        // L [size: 4 x 4]
        let mut l = vec![0.0f32; 16];

        // run decomposition
        matrix_chol(MatrixRef::new(&MATRIXF_DATA_CHOL_A, 4, 4).unwrap(), &mut MatrixMut::new(&mut l, 4, 4).unwrap())
            .unwrap();

        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXF_DATA_CHOL_L[i], l[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_gramschmidt)]
    fn test_matrixf_gramschmidt() {
        let tol = 1e-6f32;

        // A [size: 4 x 3]
        // V [size: 4 x 3]
        let mut v = vec![0.0f32; 12];
        matrix_gramschmidt(
            MatrixRef::new(&MATRIXF_DATA_GRAMSCHMIDT_A, 4, 3).unwrap(),
            &mut MatrixMut::new(&mut v, 4, 3).unwrap(),
        )
        .unwrap();

        for i in 0..12 {
            assert_abs_diff_eq!(MATRIXF_DATA_GRAMSCHMIDT_V[i], v[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_inv)]
    fn test_matrixf_inv() {
        let tol = 1e-6f32;

        // x [size: 5 x 5]
        // y [size: 5 x 5]
        let mut y = MATRIXF_DATA_INV_X.to_vec();
        matrix_inv(&mut MatrixMut::new(&mut y, 5, 5).unwrap()).unwrap();

        for i in 0..25 {
            assert_abs_diff_eq!(MATRIXF_DATA_INV_Y[i], y[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_linsolve)]
    fn test_matrixf_linsolve() {
        let tol = 1e-6f32;

        // A [size: 5 x 5]
        // x [size: 5 x 1]
        // b [size: 5 x 1]
        let mut x = vec![0.0f32; 5];
        let mut scratch = vec![0.0f32; 5 * 6];

        // run solver with caller-provided augmented-matrix storage
        matrix_linsolve(
            MatrixRef::new(&MATRIXF_DATA_LINSOLVE_A, 5, 5).unwrap(),
            &MATRIXF_DATA_LINSOLVE_B,
            &mut x,
            Some(&mut scratch),
        )
        .unwrap();

        for i in 0..5 {
            assert_abs_diff_eq!(MATRIXF_DATA_LINSOLVE_X[i], x[i], epsilon = tol);
        }

        let mut x = vec![0.0f32; 5];

        // run solver with its own allocated storage
        matrix_linsolve(
            MatrixRef::new(&MATRIXF_DATA_LINSOLVE_A, 5, 5).unwrap(),
            &MATRIXF_DATA_LINSOLVE_B,
            &mut x,
            None,
        )
        .unwrap();

        for i in 0..5 {
            assert_abs_diff_eq!(MATRIXF_DATA_LINSOLVE_X[i], x[i], epsilon = tol);
        }

        // the scratch buffer must hold the complete n x (n+1) matrix
        assert!(matrix_linsolve(
            MatrixRef::new(&MATRIXF_DATA_LINSOLVE_A, 5, 5).unwrap(),
            &MATRIXF_DATA_LINSOLVE_B,
            &mut x,
            Some(&mut scratch[..29]),
        )
        .is_err());
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_ludecomp_crout)]
    fn test_matrixf_ludecomp_crout() {
        let tol = 1e-6f32;

        let mut l = vec![0.0f32; 64];
        let mut u = vec![0.0f32; 64];
        let mut p = vec![0.0f32; 64];

        let mut lu_test = vec![0.0f32; 64];

        // run decomposition
        matrix_ludecomp_crout(
            MatrixRef::new(&MATRIXF_DATA_LUDECOMP_A, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut l, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut u, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut p, 8, 8).unwrap(),
        )
        .unwrap();

        // multiply LU
        matrix_mul(
            MatrixRef::new(&l, 8, 8).unwrap(),
            MatrixRef::new(&u, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut lu_test, 8, 8).unwrap(),
        )
        .unwrap();

        for r in 0..8 {
            for c in 0..8 {
                if r < c {
                    assert_abs_diff_eq!(l[r * 8 + c], 0.0, epsilon = tol);
                } else if r == c {
                    assert_abs_diff_eq!(u[r * 8 + c], 1.0, epsilon = tol);
                } else {
                    assert_abs_diff_eq!(u[r * 8 + c], 0.0, epsilon = tol);
                }
            }
        }

        for i in 0..64 {
            assert_abs_diff_eq!(MATRIXF_DATA_LUDECOMP_A[i], lu_test[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_ludecomp_doolittle)]
    fn test_matrixf_ludecomp_doolittle() {
        let tol = 1e-6f32;

        let mut l = vec![0.0f32; 64];
        let mut u = vec![0.0f32; 64];
        let mut p = vec![0.0f32; 64];

        let mut lu_test = vec![0.0f32; 64];

        // run decomposition
        matrix_ludecomp_doolittle(
            MatrixRef::new(&MATRIXF_DATA_LUDECOMP_A, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut l, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut u, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut p, 8, 8).unwrap(),
        )
        .unwrap();

        // multiply LU
        matrix_mul(
            MatrixRef::new(&l, 8, 8).unwrap(),
            MatrixRef::new(&u, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut lu_test, 8, 8).unwrap(),
        )
        .unwrap();

        for r in 0..8 {
            for c in 0..8 {
                if r < c {
                    assert_abs_diff_eq!(l[r * 8 + c], 0.0, epsilon = tol);
                } else if r == c {
                    assert_abs_diff_eq!(l[r * 8 + c], 1.0, epsilon = tol);
                } else {
                    assert_abs_diff_eq!(u[r * 8 + c], 0.0, epsilon = tol);
                }
            }
        }

        for i in 0..64 {
            assert_abs_diff_eq!(MATRIXF_DATA_LUDECOMP_A[i], lu_test[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_mul)]
    fn test_matrixf_mul() {
        let tol = 1e-6f32;

        // x [size: 5 x 4]
        // y [size: 4 x 3]
        // z [size: 5 x 3]
        let mut z = vec![0.0f32; 15];
        matrix_mul(
            MatrixRef::new(&MATRIXF_DATA_MUL_X, 5, 4).unwrap(),
            MatrixRef::new(&MATRIXF_DATA_MUL_Y, 4, 3).unwrap(),
            &mut MatrixMut::new(&mut z, 5, 3).unwrap(),
        )
        .unwrap();

        for i in 0..15 {
            assert_abs_diff_eq!(MATRIXF_DATA_MUL_Z[i], z[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_qrdecomp)]
    fn test_matrixf_qrdecomp() {
        let tol = 1e-4f32; // error tolerance

        let mut q = [0.0f32; 16];
        let mut r = [0.0f32; 16];

        let mut qr_test = [0.0f32; 16]; // Q*R
        let mut qqt_test = [0.0f32; 16]; // Q*Q^T

        // run decomposition
        matrix_qrdecomp_gramschmidt(
            MatrixRef::new(&MATRIXF_DATA_QRDECOMP_A, 4, 4).unwrap(),
            &mut MatrixMut::new(&mut q, 4, 4).unwrap(),
            &mut MatrixMut::new(&mut r, 4, 4).unwrap(),
        )
        .unwrap();

        // compute Q*R
        matrix_mul(
            MatrixRef::new(&q, 4, 4).unwrap(),
            MatrixRef::new(&r, 4, 4).unwrap(),
            &mut MatrixMut::new(&mut qr_test, 4, 4).unwrap(),
        )
        .unwrap();

        // compute Q*Q^T
        matrix_mul_transpose(MatrixRef::new(&q, 4, 4).unwrap(), &mut MatrixMut::new(&mut qqt_test, 4, 4).unwrap())
            .unwrap();

        // ensure Q*R = A
        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXF_DATA_QRDECOMP_A[i], qr_test[i], epsilon = tol);
        }

        // ensure Q*Q = I(4)
        let mut i4 = [0.0f32; 16];
        matrix_eye(&mut MatrixMut::new(&mut i4, 4, 4).unwrap()).unwrap();
        for i in 0..16 {
            assert_abs_diff_eq!(qqt_test[i], i4[i], epsilon = tol);
        }

        // ensure Q and R are correct
        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXF_DATA_QRDECOMP_Q[i], q[i], epsilon = tol);
            assert_abs_diff_eq!(MATRIXF_DATA_QRDECOMP_R[i], r[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixf_transmul)]
    fn test_matrixf_transmul() {
        let tol = 1e-4f32;

        let mut xxt = vec![0.0f32; 25]; // [size: 5 x 5]
        let mut xxh = vec![0.0f32; 25]; // [size: 5 x 5]
        let mut xtx = vec![0.0f32; 16]; // [size: 4 x 4]
        let mut xhx = vec![0.0f32; 16]; // [size: 4 x 4]

        // run matrix multiplications
        matrix_mul_transpose(
            MatrixRef::new(&MATRIXF_DATA_TRANSMUL_X, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut xxt, 5, 5).unwrap(),
        )
        .unwrap();
        matrix_mul_hermitian(
            MatrixRef::new(&MATRIXF_DATA_TRANSMUL_X, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut xxh, 5, 5).unwrap(),
        )
        .unwrap();
        matrix_transpose_mul(
            MatrixRef::new(&MATRIXF_DATA_TRANSMUL_X, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut xtx, 4, 4).unwrap(),
        )
        .unwrap();
        matrix_hermitian_mul(
            MatrixRef::new(&MATRIXF_DATA_TRANSMUL_X, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut xhx, 4, 4).unwrap(),
        )
        .unwrap();

        // run tests
        for i in 0..25 {
            assert_abs_diff_eq!(MATRIXF_DATA_TRANSMUL_XXT[i], xxt[i], epsilon = tol);
        }

        for i in 0..25 {
            assert_abs_diff_eq!(MATRIXF_DATA_TRANSMUL_XXH[i], xxh[i], epsilon = tol);
        }

        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXF_DATA_TRANSMUL_XTX[i], xtx[i], epsilon = tol);
        }

        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXF_DATA_TRANSMUL_XHX[i], xhx[i], epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_add)]
    fn test_matrixcf_add() {
        let tol: f32 = 1e-6;
        let mut z = [Complex::new(0.0, 0.0); 20];

        matrix_add(
            MatrixRef::new(&MATRIXCF_DATA_ADD_X, 5, 4).unwrap(),
            MatrixRef::new(&MATRIXCF_DATA_ADD_Y, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut z, 5, 4).unwrap(),
        )
        .unwrap();

        for i in 0..20 {
            assert_abs_diff_eq!(z[i].re, MATRIXCF_DATA_ADD_Z[i].re, epsilon = tol);
            assert_abs_diff_eq!(z[i].im, MATRIXCF_DATA_ADD_Z[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_aug)]
    fn test_matrixcf_aug() {
        let tol: f32 = 1e-6;
        let mut z = [Complex::new(0.0, 0.0); 35];

        matrix_aug(
            MatrixRef::new(&MATRIXCF_DATA_AUG_X, 5, 4).unwrap(),
            MatrixRef::new(&MATRIXCF_DATA_AUG_Y, 5, 3).unwrap(),
            &mut MatrixMut::new(&mut z, 5, 7).unwrap(),
        )
        .unwrap();

        for i in 0..35 {
            assert_abs_diff_eq!(z[i].re, MATRIXCF_DATA_AUG_Z[i].re, epsilon = tol);
            assert_abs_diff_eq!(z[i].im, MATRIXCF_DATA_AUG_Z[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_chol)]
    fn test_matrixcf_chol() {
        let tol: f32 = 1e-3;
        let mut l = [Complex::new(0.0, 0.0); 16];

        matrix_chol(MatrixRef::new(&MATRIXCF_DATA_CHOL_A, 4, 4).unwrap(), &mut MatrixMut::new(&mut l, 4, 4).unwrap())
            .unwrap();

        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXCF_DATA_CHOL_L[i].re, l[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_CHOL_L[i].im, l[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_inv)]
    fn test_matrixcf_inv() {
        let tol: f32 = 1e-6;
        let mut y = MATRIXCF_DATA_INV_X;

        matrix_inv(&mut MatrixMut::new(&mut y, 5, 5).unwrap()).unwrap();

        for i in 0..25 {
            assert_abs_diff_eq!(MATRIXCF_DATA_INV_Y[i].re, y[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_INV_Y[i].im, y[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_linsolve)]
    fn test_matrixcf_linsolve() {
        let tol = 1e-6f32; // error tolerance

        // A [size: 5 x 5]
        // x [size: 5 x 1]
        // b [size: 5 x 1]
        let mut x = [Complex::new(0.0, 0.0); 5];

        // run solver
        matrix_linsolve(
            MatrixRef::new(&MATRIXCF_DATA_LINSOLVE_A, 5, 5).unwrap(),
            &MATRIXCF_DATA_LINSOLVE_B,
            &mut x,
            None,
        )
        .unwrap();

        for i in 0..5 {
            assert_abs_diff_eq!(MATRIXCF_DATA_LINSOLVE_X[i].re, x[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_LINSOLVE_X[i].im, x[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_ludecomp_crout)]
    fn test_matrixcf_ludecomp_crout() {
        let tol: f32 = 1e-5;
        let mut l = [Complex::new(0.0, 0.0); 64];
        let mut u = [Complex::new(0.0, 0.0); 64];
        let mut p = [Complex::new(0.0, 0.0); 64];
        let mut lu_test = [Complex::new(0.0, 0.0); 64];

        matrix_ludecomp_crout(
            MatrixRef::new(&MATRIXCF_DATA_LUDECOMP_A, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut l, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut u, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut p, 8, 8).unwrap(),
        )
        .unwrap();
        matrix_mul(
            MatrixRef::new(&l, 8, 8).unwrap(),
            MatrixRef::new(&u, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut lu_test, 8, 8).unwrap(),
        )
        .unwrap();

        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                if r < c {
                    assert_abs_diff_eq!(l[r * 8 + c].re, 0.0, epsilon = tol);
                    assert_abs_diff_eq!(l[r * 8 + c].im, 0.0, epsilon = tol);
                } else if r == c {
                    assert_abs_diff_eq!(u[r * 8 + c].re, 1.0, epsilon = tol);
                    assert_abs_diff_eq!(u[r * 8 + c].im, 0.0, epsilon = tol);
                }
                if r > c {
                    assert_abs_diff_eq!(u[r * 8 + c].re, 0.0, epsilon = tol);
                    assert_abs_diff_eq!(u[r * 8 + c].im, 0.0, epsilon = tol);
                }
                assert_abs_diff_eq!(MATRIXCF_DATA_LUDECOMP_A[i].re, lu_test[i].re, epsilon = tol);
                assert_abs_diff_eq!(MATRIXCF_DATA_LUDECOMP_A[i].im, lu_test[i].im, epsilon = tol);
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_ludecomp_doolittle)]
    fn test_matrixcf_ludecomp_doolittle() {
        let tol = 1e-5f32; // error tolerance

        let mut l = [Complex::new(0.0, 0.0); 64];
        let mut u = [Complex::new(0.0, 0.0); 64];
        let mut p = [Complex::new(0.0, 0.0); 64];

        // run decomposition
        matrix_ludecomp_doolittle(
            MatrixRef::new(&MATRIXCF_DATA_LUDECOMP_A, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut l, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut u, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut p, 8, 8).unwrap(),
        )
        .unwrap();

        let mut lu_test = [Complex::new(0.0, 0.0); 64];
        matrix_mul(
            MatrixRef::new(&l, 8, 8).unwrap(),
            MatrixRef::new(&u, 8, 8).unwrap(),
            &mut MatrixMut::new(&mut lu_test, 8, 8).unwrap(),
        )
        .unwrap();

        // ensure L has 1's along diagonal and 0's above
        // ensure U has 0's below diagonal
        for r in 0..8 {
            for c in 0..8 {
                if r < c {
                    assert_abs_diff_eq!(l[r * 8 + c].re, 0.0, epsilon = tol);
                    assert_abs_diff_eq!(l[r * 8 + c].im, 0.0, epsilon = tol);
                } else if r == c {
                    assert_abs_diff_eq!(l[r * 8 + c].re, 1.0, epsilon = tol);
                    assert_abs_diff_eq!(l[r * 8 + c].im, 0.0, epsilon = tol);
                } else if r > c {
                    assert_abs_diff_eq!(u[r * 8 + c].re, 0.0, epsilon = tol);
                    assert_abs_diff_eq!(u[r * 8 + c].im, 0.0, epsilon = tol);
                }
            }
        }

        // ensure L*U = A
        for i in 0..64 {
            assert_abs_diff_eq!(MATRIXCF_DATA_LUDECOMP_A[i].re, lu_test[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_LUDECOMP_A[i].im, lu_test[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_mul)]
    fn test_matrixcf_mul() {
        let tol: f32 = 1e-6;
        let mut z = [Complex::new(0.0, 0.0); 15];

        matrix_mul(
            MatrixRef::new(&MATRIXCF_DATA_MUL_X, 5, 4).unwrap(),
            MatrixRef::new(&MATRIXCF_DATA_MUL_Y, 4, 3).unwrap(),
            &mut MatrixMut::new(&mut z, 5, 3).unwrap(),
        )
        .unwrap();

        for i in 0..15 {
            assert_abs_diff_eq!(MATRIXCF_DATA_MUL_Z[i].re, z[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_MUL_Z[i].im, z[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_qrdecomp)]
    fn test_matrixcf_qrdecomp() {
        let tol = 1e-4f32; // error tolerance

        let mut q = [Complex::new(0.0, 0.0); 16];
        let mut r = [Complex::new(0.0, 0.0); 16];

        let mut qr_test = [Complex::new(0.0, 0.0); 16]; // Q*R
        let mut qqt_test = [Complex::new(0.0, 0.0); 16]; // Q*Q^T

        // run decomposition
        matrix_qrdecomp_gramschmidt(
            MatrixRef::new(&MATRIXCF_DATA_QRDECOMP_A, 4, 4).unwrap(),
            &mut MatrixMut::new(&mut q, 4, 4).unwrap(),
            &mut MatrixMut::new(&mut r, 4, 4).unwrap(),
        )
        .unwrap();

        // compute Q*R
        matrix_mul(
            MatrixRef::new(&q, 4, 4).unwrap(),
            MatrixRef::new(&r, 4, 4).unwrap(),
            &mut MatrixMut::new(&mut qr_test, 4, 4).unwrap(),
        )
        .unwrap();

        // compute Q*Q^T
        matrix_mul_transpose(MatrixRef::new(&q, 4, 4).unwrap(), &mut MatrixMut::new(&mut qqt_test, 4, 4).unwrap())
            .unwrap();

        // ensure Q*R = A
        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXCF_DATA_QRDECOMP_A[i].re, qr_test[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_QRDECOMP_A[i].im, qr_test[i].im, epsilon = tol);
        }

        // ensure Q*Q^T = I(4)
        let mut i4 = [Complex::new(0.0, 0.0); 16];
        matrix_eye(&mut MatrixMut::new(&mut i4, 4, 4).unwrap()).unwrap();
        for i in 0..16 {
            assert_abs_diff_eq!(qqt_test[i].re, i4[i].re, epsilon = tol);
            assert_abs_diff_eq!(qqt_test[i].im, i4[i].im, epsilon = tol);
        }

        // ensure Q and R are correct
        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXCF_DATA_QRDECOMP_Q[i].re, q[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_QRDECOMP_Q[i].im, q[i].im, epsilon = tol);

            assert_abs_diff_eq!(MATRIXCF_DATA_QRDECOMP_R[i].re, r[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_QRDECOMP_R[i].im, r[i].im, epsilon = tol);
        }
    }

    #[test]
    #[autotest_annotate(autotest_matrixcf_transmul)]
    fn test_matrixcf_transmul() {
        let tol = 1e-4f32; // error tolerance

        let mut xxt = [Complex::new(0.0, 0.0); 25]; // [size: 5 x 5]
        let mut xxh = [Complex::new(0.0, 0.0); 25]; // [size: 5 x 5]
        let mut xtx = [Complex::new(0.0, 0.0); 16]; // [size: 4 x 4]
        let mut xhx = [Complex::new(0.0, 0.0); 16]; // [size: 4 x 4]

        // run matrix multiplications
        matrix_mul_transpose(
            MatrixRef::new(&MATRIXCF_DATA_TRANSMUL_X, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut xxt, 5, 5).unwrap(),
        )
        .unwrap();
        matrix_mul_hermitian(
            MatrixRef::new(&MATRIXCF_DATA_TRANSMUL_X, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut xxh, 5, 5).unwrap(),
        )
        .unwrap();
        matrix_transpose_mul(
            MatrixRef::new(&MATRIXCF_DATA_TRANSMUL_X, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut xtx, 4, 4).unwrap(),
        )
        .unwrap();
        matrix_hermitian_mul(
            MatrixRef::new(&MATRIXCF_DATA_TRANSMUL_X, 5, 4).unwrap(),
            &mut MatrixMut::new(&mut xhx, 4, 4).unwrap(),
        )
        .unwrap();

        // run tests
        for i in 0..25 {
            assert_abs_diff_eq!(MATRIXCF_DATA_TRANSMUL_XXT[i].re, xxt[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_TRANSMUL_XXT[i].im, xxt[i].im, epsilon = tol);
        }

        for i in 0..25 {
            assert_abs_diff_eq!(MATRIXCF_DATA_TRANSMUL_XXH[i].re, xxh[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_TRANSMUL_XXH[i].im, xxh[i].im, epsilon = tol);
        }

        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXCF_DATA_TRANSMUL_XTX[i].re, xtx[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_TRANSMUL_XTX[i].im, xtx[i].im, epsilon = tol);
        }

        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXCF_DATA_TRANSMUL_XHX[i].re, xhx[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_TRANSMUL_XHX[i].im, xhx[i].im, epsilon = tol);
        }
    }

    #[test]
    fn test_matrix_mul_known_product() {
        let left = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let right = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected = [58.0f32, 64.0, 139.0, 154.0];
        let mut output = [0.0f32; 4];

        matrix_mul(
            MatrixRef::new(&left, 2, 3).unwrap(),
            MatrixRef::new(&right, 3, 2).unwrap(),
            &mut MatrixMut::new(&mut output, 2, 2).unwrap(),
        )
        .unwrap();

        assert_eq!(output, expected);
    }

    #[test]
    fn test_matrix_mul_dimension_mismatch_is_error() {
        let left = [1.0f32; 6];
        let right = [1.0f32; 8];
        let mut output = [42.0f32; 4];

        let result = matrix_mul(
            MatrixRef::new(&left, 2, 3).unwrap(),
            MatrixRef::new(&right, 4, 2).unwrap(),
            &mut MatrixMut::new(&mut output, 2, 2).unwrap(),
        );

        assert!(result.is_err());
        assert_eq!(output, [42.0; 4]);
    }

    #[test]
    fn test_matrix_mul_vec_known_product_and_length_validation() {
        let matrix = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = [7.0f32, 8.0, 9.0];
        let mut output = [0.0f32; 2];

        matrix_mul_vec(MatrixRef::new(&matrix, 2, 3).unwrap(), &vector, &mut output).unwrap();
        assert_eq!(output, [50.0, 122.0]);

        let mut unchanged = [42.0f32; 2];
        assert!(matrix_mul_vec(MatrixRef::new(&matrix, 2, 3).unwrap(), &vector[..2], &mut unchanged).is_err());
        assert_eq!(unchanged, [42.0; 2]);
    }

    #[test]
    fn test_matrix_mul_accumulates_inner_terms_in_ascending_order() {
        let left = [1.0e20f32, -1.0e20, 1.0];
        let right = [1.0f32, 1.0, 1.0];
        let mut output = [0.0f32];

        matrix_mul(
            MatrixRef::new(&left, 1, 3).unwrap(),
            MatrixRef::new(&right, 3, 1).unwrap(),
            &mut MatrixMut::new(&mut output, 1, 1).unwrap(),
        )
        .unwrap();

        assert_eq!(output, [1.0]);
    }

    #[test]
    fn test_matrix_inverse_round_trip() {
        let original = [4.0f32, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0];
        let mut inverse = original;

        matrix_inv(&mut MatrixMut::new(&mut inverse, 3, 3).unwrap()).unwrap();

        for row in 0..3 {
            for column in 0..3 {
                let product = (0..3).map(|inner| original[row * 3 + inner] * inverse[inner * 3 + column]).sum::<f32>();
                let expected = if row == column { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(product, expected, epsilon = 2e-5);
            }
        }
    }

    #[test]
    fn test_matrix_inverse_validates_length() {
        let mut singular = [1.0f32, 2.0, 2.0, 4.0];
        assert!(matrix_inv(&mut MatrixMut::new(&mut singular, 2, 2).unwrap()).is_err());
        assert_eq!(singular, [1.0, 2.0, 2.0, 4.0]);
    }

    #[test]
    fn test_matrix_linsolve_validates_buffers() {
        let a = [2.0f32, 0.0, 0.0, 2.0];
        let b = [2.0f32, 4.0];
        let mut x = [42.0f32; 2];

        assert!(matrix_linsolve(MatrixRef::new(&a, 2, 2).unwrap(), &b[..1], &mut x, None).is_err());
        assert_eq!(x, [42.0; 2]);

        let mut short_scratch = [0.0f32; 5];
        assert!(matrix_linsolve(MatrixRef::new(&a, 2, 2).unwrap(), &b, &mut x, Some(&mut short_scratch)).is_err());
        assert_eq!(x, [42.0; 2]);

        let mut scratch = [0.0f32; 7];
        scratch[6] = 99.0;
        matrix_linsolve(MatrixRef::new(&a, 2, 2).unwrap(), &b, &mut x, Some(&mut scratch)).unwrap();
        assert_eq!(x, [1.0, 2.0]);
        assert_eq!(scratch[6], 99.0);
    }

    #[test]
    fn test_matrix_chol_reconstructs_hermitian_input() {
        let a = [Complex::new(4.0f32, 0.0), Complex::new(2.0, -2.0), Complex::new(2.0, 2.0), Complex::new(11.0, 0.0)];
        let mut l = [Complex::new(0.0f32, 0.0); 4];

        matrix_chol(MatrixRef::new(&a, 2, 2).unwrap(), &mut MatrixMut::new(&mut l, 2, 2).unwrap()).unwrap();

        for row in 0..2 {
            for column in 0..2 {
                let reconstructed: Complex<f32> =
                    (0..2).map(|inner| l[row * 2 + inner] * l[column * 2 + inner].conj()).sum();
                assert_abs_diff_eq!(reconstructed.re, a[row * 2 + column].re, epsilon = 1e-5);
                assert_abs_diff_eq!(reconstructed.im, a[row * 2 + column].im, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_matrix_lu_reconstructs_permuted_input() {
        fn verify<F>(decompose: F)
        where
            F: for<'a, 'b, 'c, 'd> Fn(
                MatrixRef<'a, f32>,
                &mut MatrixMut<'b, f32>,
                &mut MatrixMut<'c, f32>,
                &mut MatrixMut<'d, f32>,
            ) -> crate::error::Result<()>,
        {
            let a = [4.0f32, 3.0, 2.0, 3.0, 5.0, 1.0, 1.0, 1.0, 3.0];
            let mut l = [0.0f32; 9];
            let mut u = [0.0f32; 9];
            let mut p = [0.0f32; 9];
            let mut pa = [0.0f32; 9];
            let mut lu = [0.0f32; 9];

            decompose(
                MatrixRef::new(&a, 3, 3).unwrap(),
                &mut MatrixMut::new(&mut l, 3, 3).unwrap(),
                &mut MatrixMut::new(&mut u, 3, 3).unwrap(),
                &mut MatrixMut::new(&mut p, 3, 3).unwrap(),
            )
            .unwrap();
            matrix_mul(
                MatrixRef::new(&p, 3, 3).unwrap(),
                MatrixRef::new(&a, 3, 3).unwrap(),
                &mut MatrixMut::new(&mut pa, 3, 3).unwrap(),
            )
            .unwrap();
            matrix_mul(
                MatrixRef::new(&l, 3, 3).unwrap(),
                MatrixRef::new(&u, 3, 3).unwrap(),
                &mut MatrixMut::new(&mut lu, 3, 3).unwrap(),
            )
            .unwrap();

            for (permuted, reconstructed) in pa.iter().zip(lu) {
                assert_abs_diff_eq!(*permuted, reconstructed, epsilon = 1e-5);
            }
        }

        verify(matrix_ludecomp_crout);
        verify(matrix_ludecomp_doolittle);
    }

    #[test]
    fn test_matrix_chol_and_lu_validate_buffers() {
        let input = [2.0f32, 1.0, 1.0, 2.0];
        let mut chol_output = [42.0f32; 4];
        assert!(matrix_chol(
            MatrixRef::new(&input, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut chol_output, 1, 4).unwrap(),
        )
        .is_err());
        assert_eq!(chol_output, [42.0; 4]);

        let mut l = [42.0f32; 4];
        let mut u = [42.0f32; 3];
        let mut p = [42.0f32; 4];
        assert!(matrix_ludecomp_crout(
            MatrixRef::new(&input, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut l, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut u, 1, 3).unwrap(),
            &mut MatrixMut::new(&mut p, 2, 2).unwrap(),
        )
        .is_err());
        assert_eq!(l, [42.0; 4]);
        assert_eq!(u, [42.0; 3]);
        assert_eq!(p, [42.0; 4]);
    }

    #[test]
    fn test_matrix_gramschmidt_and_qr_validate_buffers() {
        let input = [2.0f32, 1.0, 1.0, 2.0];
        let mut gram_output = [42.0f32; 4];
        assert!(matrix_gramschmidt(
            MatrixRef::new(&input, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut gram_output, 1, 4).unwrap(),
        )
        .is_err());
        assert_eq!(gram_output, [42.0; 4]);

        let mut q = [42.0f32; 4];
        let mut r = [42.0f32; 3];
        assert!(matrix_qrdecomp_gramschmidt(
            MatrixRef::new(&input, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut q, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut r, 1, 3).unwrap(),
        )
        .is_err());
        assert_eq!(q, [42.0; 4]);
        assert_eq!(r, [42.0; 3]);
    }

    #[test]
    fn test_matrix_cgsolve_converges() {
        let a = [4.0f32, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let b = [1.0f32, 2.0, 3.0];
        let mut x = [0.0f32; 3];

        matrix_cgsolve(MatrixRef::new(&a, 3, 3).unwrap(), &b, &mut x, None).unwrap();

        let residual_squared = (0..3)
            .map(|row| {
                let ax = (0..3).map(|column| a[row * 3 + column] * x[column]).sum::<f32>();
                let residual = b[row] - ax;
                residual * residual
            })
            .sum::<f32>();
        assert!(residual_squared.sqrt() < 1e-5, "residual was {residual_squared:e}");
    }

    #[test]
    fn test_matrix_cgsolve_zero_rhs_terminates_at_zero_solution() {
        let a = [2.0f32, 1.0, 1.0, 2.0];
        let b = [0.0f32; 2];
        let mut x = [42.0f32; 2];

        matrix_cgsolve(MatrixRef::new(&a, 2, 2).unwrap(), &b, &mut x, None).unwrap();

        assert_eq!(x, [0.0; 2]);
    }

    #[test]
    fn test_matrix_cgsolve_validates_buffers() {
        let a = [2.0f32, 1.0, 1.0, 2.0];
        let b = [1.0f32, 1.0];
        let mut x = [42.0f32; 2];

        assert!(matrix_cgsolve(MatrixRef::new(&a, 2, 2).unwrap(), &b[..1], &mut x, None).is_err());
        assert_eq!(x, [42.0; 2]);
        assert!(matrix_cgsolve(MatrixRef::new(&a, 2, 2).unwrap(), &b, &mut x[..1], None).is_err());
        assert_eq!(x, [42.0; 2]);
        assert!(matrix_cgsolve::<f32>(MatrixRef::new(&[], 0, 0).unwrap(), &[], &mut [], None).is_err());
    }

    #[test]
    fn test_matrix_cgsolve_reports_breakdown_without_changing_output() {
        let a = [0.0f32; 4];
        let b = [1.0f32, 0.0];
        let mut x = [42.0f32; 2];

        let error = matrix_cgsolve(MatrixRef::new(&a, 2, 2).unwrap(), &b, &mut x, None).unwrap_err();

        assert!(matches!(error, crate::error::Error::NoConvergence(_)));
        assert_eq!(x, [42.0; 2]);
    }

    #[test]
    fn test_matrix_cgsolve_rejects_nonfinite_inputs_without_changing_output() {
        let a = [f32::NAN, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0];
        let mut x = [42.0f32; 2];

        assert!(matrix_cgsolve(MatrixRef::new(&a, 2, 2).unwrap(), &b, &mut x, None).is_err());
        assert_eq!(x, [42.0; 2]);
    }

    #[test]
    fn test_matrix_det_known_values() {
        let one_by_one = [7.0f32];
        let two_by_two = [4.0f32, 7.0, 2.0, 6.0];
        let three_by_three = [6.0f32, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];

        assert_eq!(matrix_det(MatrixRef::new(&one_by_one, 1, 1).unwrap()).unwrap(), 7.0);
        assert_eq!(matrix_det(MatrixRef::new(&two_by_two, 2, 2).unwrap()).unwrap(), 10.0);
        assert_eq!(matrix_det(MatrixRef::new(&three_by_three, 3, 3).unwrap()).unwrap(), -306.0);
    }

    #[test]
    fn test_matrix_base_initializers_validate_shape() {
        let mut identity = [0.0f32; 4];
        matrix_eye(&mut MatrixMut::new(&mut identity, 2, 2).unwrap()).unwrap();
        assert_eq!(identity, [1.0, 0.0, 0.0, 1.0]);

        matrix_ones(&mut MatrixMut::new(&mut identity, 2, 2).unwrap()).unwrap();
        assert_eq!(identity, [1.0; 4]);
        matrix_zeros(&mut MatrixMut::new(&mut identity, 2, 2).unwrap()).unwrap();
        assert_eq!(identity, [0.0; 4]);
    }

    #[test]
    fn test_matrix_elementwise_operations_validate_before_output_mutation() {
        let left = [12.0f32, 10.0, 8.0, 6.0];
        let right = [3.0f32, 2.0, 4.0, 3.0];
        let mut output = [0.0f32; 4];

        matrix_add(
            MatrixRef::new(&left, 2, 2).unwrap(),
            MatrixRef::new(&right, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut output, 2, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(output, [15.0, 12.0, 12.0, 9.0]);
        matrix_sub(
            MatrixRef::new(&left, 2, 2).unwrap(),
            MatrixRef::new(&right, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut output, 2, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(output, [9.0, 8.0, 4.0, 3.0]);
        matrix_pmul(
            MatrixRef::new(&left, 2, 2).unwrap(),
            MatrixRef::new(&right, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut output, 2, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(output, [36.0, 20.0, 32.0, 18.0]);
        matrix_pdiv(
            MatrixRef::new(&left, 2, 2).unwrap(),
            MatrixRef::new(&right, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut output, 2, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(output, [4.0, 5.0, 2.0, 2.0]);
    }

    #[test]
    fn test_matrix_aug_validates_buffers_before_output_mutation() {
        let left = [1.0f32, 2.0];
        let right = [3.0f32, 4.0, 5.0, 6.0];
        let mut output = [0.0f32; 6];

        matrix_aug(
            MatrixRef::new(&left, 2, 1).unwrap(),
            MatrixRef::new(&right, 2, 2).unwrap(),
            &mut MatrixMut::new(&mut output, 2, 3).unwrap(),
        )
        .unwrap();
        assert_eq!(output, [1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_matrix_transpose_helpers_complex() {
        let original = [
            Complex::new(1.0f32, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
            Complex::new(7.0, 8.0),
            Complex::new(9.0, 10.0),
            Complex::new(11.0, 12.0),
        ];
        let mut hermitian = original;
        let mut trans = original;

        matrix_hermitian(&mut MatrixMut::new(&mut hermitian, 2, 3).unwrap()).unwrap();
        matrix_trans(&mut MatrixMut::new(&mut trans, 2, 3).unwrap()).unwrap();

        let expected = [
            Complex::new(1.0, 2.0),
            Complex::new(7.0, 8.0),
            Complex::new(3.0, 4.0),
            Complex::new(9.0, 10.0),
            Complex::new(5.0, 6.0),
            Complex::new(11.0, 12.0),
        ];
        assert_eq!(hermitian, expected);
        assert_eq!(trans, expected.map(|value| value.conj()));
    }
}
