// Matrix module
// Current state:
// - Autotests matching and passing
// - ready to use
// - may rethink later - struct or trait based approach may be interesting

pub mod base;
pub mod cgsolve;
pub mod chol;
pub mod gramschmidt;
pub mod inv;
pub mod linsolve;
pub mod ludecomp;
pub mod math;
pub mod qrdecomp;
pub mod sparse;

pub use crate::matrix::base::*;
pub use crate::matrix::cgsolve::*;
pub use crate::matrix::chol::*;
pub use crate::matrix::gramschmidt::*;
pub use crate::matrix::inv::*;
pub use crate::matrix::linsolve::*;
pub use crate::matrix::ludecomp::*;
pub use crate::matrix::math::*;
pub use crate::matrix::qrdecomp::*;
pub use crate::matrix::sparse::*;

#[cfg(test)]
mod tests {
    use crate::matrix::{
        matrix_access, matrix_add, matrix_aug, matrix_cgsolve, matrix_chol, matrix_eye, matrix_gramschmidt, matrix_hermitian_mul,
        matrix_inv, matrix_linsolve, matrix_ludecomp_crout, matrix_ludecomp_doolittle, matrix_mul, matrix_mul_hermitian,
        matrix_mul_transpose, matrix_qrdecomp_gramschmidt_f32, matrix_qrdecomp_gramschmidt_c32, matrix_transpose_mul,
    };
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
        matrix_add(&MATRIXF_DATA_ADD_X, &MATRIXF_DATA_ADD_Y, &mut z, 5, 4);

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
        matrix_aug(&MATRIXF_DATA_AUG_X, 5, 4, &MATRIXF_DATA_AUG_Y, 5, 3, &mut z, 5, 7).unwrap();

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
        matrix_cgsolve(&MATRIXF_DATA_CGSOLVE_A, 8, &MATRIXF_DATA_CGSOLVE_B, &mut x, None).unwrap();

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
        matrix_chol(&MATRIXF_DATA_CHOL_A, 4, &mut l).unwrap();

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
        matrix_gramschmidt(&MATRIXF_DATA_GRAMSCHMIDT_A, 4, 3, &mut v).unwrap();

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
        matrix_inv(&mut y, 5, 5).unwrap();

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
            &MATRIXF_DATA_LINSOLVE_A,
            5,
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
            &MATRIXF_DATA_LINSOLVE_A,
            5,
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
            &MATRIXF_DATA_LINSOLVE_A,
            5,
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
        matrix_ludecomp_crout(&MATRIXF_DATA_LUDECOMP_A, 8, 8, &mut l, &mut u, &mut p).unwrap();

        // multiply LU
        matrix_mul(&l, 8, 8, &u, 8, 8, &mut lu_test, 8, 8).unwrap();

        for r in 0..8 {
            for c in 0..8 {
                if r < c {
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c), 0.0, epsilon = tol);
                } else if r == c {
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c), 1.0, epsilon = tol);
                } else {
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c), 0.0, epsilon = tol);
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
        matrix_ludecomp_doolittle(&MATRIXF_DATA_LUDECOMP_A, 8, 8, &mut l, &mut u, &mut p).unwrap();

        // multiply LU
        matrix_mul(&l, 8, 8, &u, 8, 8, &mut lu_test, 8, 8).unwrap();

        for r in 0..8 {
            for c in 0..8 {
                if r < c {
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c), 0.0, epsilon = tol);
                } else if r == c {
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c), 1.0, epsilon = tol);
                } else {
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c), 0.0, epsilon = tol);
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
        matrix_mul(&MATRIXF_DATA_MUL_X, 5, 4, &MATRIXF_DATA_MUL_Y, 4, 3, &mut z, 5, 3).unwrap();

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
        matrix_qrdecomp_gramschmidt_f32(&MATRIXF_DATA_QRDECOMP_A, 4, 4, &mut q, &mut r).unwrap();

        // compute Q*R
        matrix_mul(&q, 4, 4, &r, 4, 4, &mut qr_test, 4, 4).unwrap();

        // compute Q*Q^T
        matrix_mul_transpose(&q, 4, 4, &mut qqt_test);

        // ensure Q*R = A
        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXF_DATA_QRDECOMP_A[i], qr_test[i], epsilon = tol);
        }

        // ensure Q*Q = I(4)
        let mut i4 = [0.0f32; 16];
        matrix_eye(&mut i4, 4);
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
        matrix_mul_transpose(&MATRIXF_DATA_TRANSMUL_X, 5, 4, &mut xxt);
        matrix_mul_hermitian(&MATRIXF_DATA_TRANSMUL_X, 5, 4, &mut xxh);
        matrix_transpose_mul(&MATRIXF_DATA_TRANSMUL_X, 5, 4, &mut xtx);
        matrix_hermitian_mul(&MATRIXF_DATA_TRANSMUL_X, 5, 4, &mut xhx);

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

        matrix_add(&MATRIXCF_DATA_ADD_X, &MATRIXCF_DATA_ADD_Y, &mut z, 5, 4);

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

        matrix_aug(&MATRIXCF_DATA_AUG_X, 5, 4, &MATRIXCF_DATA_AUG_Y, 5, 3, &mut z, 5, 7).unwrap();

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

        matrix_chol(&MATRIXCF_DATA_CHOL_A, 4, &mut l).unwrap();

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

        matrix_inv(&mut y, 5, 5).unwrap();

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
        matrix_linsolve(&MATRIXCF_DATA_LINSOLVE_A, 5, &MATRIXCF_DATA_LINSOLVE_B, &mut x, None).unwrap();

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

        matrix_ludecomp_crout(&MATRIXCF_DATA_LUDECOMP_A, 8, 8, &mut l, &mut u, &mut p).unwrap();
        matrix_mul(&l, 8, 8, &u, 8, 8, &mut lu_test, 8, 8).unwrap();

        for r in 0..8 {
            for c in 0..8 {
                let i = r * 8 + c;
                if r < c {
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c).re, 0.0, epsilon = tol);
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c).im, 0.0, epsilon = tol);
                } else if r == c {
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c).re, 1.0, epsilon = tol);
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c).im, 0.0, epsilon = tol);
                }
                if r > c {
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c).re, 0.0, epsilon = tol);
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c).im, 0.0, epsilon = tol);
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
        matrix_ludecomp_doolittle(&MATRIXCF_DATA_LUDECOMP_A, 8, 8, &mut l, &mut u, &mut p).unwrap();

        let mut lu_test = [Complex::new(0.0, 0.0); 64];
        matrix_mul(&l, 8, 8, &u, 8, 8, &mut lu_test, 8, 8).unwrap();

        // ensure L has 1's along diagonal and 0's above
        // ensure U has 0's below diagonal
        for r in 0..8 {
            for c in 0..8 {
                if r < c {
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c).re, 0.0, epsilon = tol);
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c).im, 0.0, epsilon = tol);
                } else if r == c {
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c).re, 1.0, epsilon = tol);
                    assert_abs_diff_eq!(matrix_access(&l, 8, 8, r, c).im, 0.0, epsilon = tol);
                } else if r > c {
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c).re, 0.0, epsilon = tol);
                    assert_abs_diff_eq!(matrix_access(&u, 8, 8, r, c).im, 0.0, epsilon = tol);
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

        matrix_mul(&MATRIXCF_DATA_MUL_X, 5, 4, &MATRIXCF_DATA_MUL_Y, 4, 3, &mut z, 5, 3).unwrap();

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
        matrix_qrdecomp_gramschmidt_c32(&MATRIXCF_DATA_QRDECOMP_A, 4, 4, &mut q, &mut r).unwrap();

        // compute Q*R
        matrix_mul(&q, 4, 4, &r, 4, 4, &mut qr_test, 4, 4).unwrap();

        // compute Q*Q^T
        matrix_mul_transpose(&q, 4, 4, &mut qqt_test);

        // ensure Q*R = A
        for i in 0..16 {
            assert_abs_diff_eq!(MATRIXCF_DATA_QRDECOMP_A[i].re, qr_test[i].re, epsilon = tol);
            assert_abs_diff_eq!(MATRIXCF_DATA_QRDECOMP_A[i].im, qr_test[i].im, epsilon = tol);
        }

        // ensure Q*Q^T = I(4)
        let mut i4 = [Complex::new(0.0, 0.0); 16];
        matrix_eye(&mut i4, 4);
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
        matrix_mul_transpose(&MATRIXCF_DATA_TRANSMUL_X, 5, 4, &mut xxt);
        matrix_mul_hermitian(&MATRIXCF_DATA_TRANSMUL_X, 5, 4, &mut xxh);
        matrix_transpose_mul(&MATRIXCF_DATA_TRANSMUL_X, 5, 4, &mut xtx);
        matrix_hermitian_mul(&MATRIXCF_DATA_TRANSMUL_X, 5, 4, &mut xhx);

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
}
