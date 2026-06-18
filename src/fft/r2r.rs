// Real-to-real FFT transforms (DCT/DST)
//
// This module provides DCT (Discrete Cosine Transform) and DST (Discrete Sine Transform)
// implementations using rustdct. The transform types follow FFTW naming conventions:
//
// DCT Types:
//   REDFT00 = DCT-I
//   REDFT10 = DCT-II  (the "DCT", used in JPEG)
//   REDFT01 = DCT-III (the "IDCT", inverse of DCT-II)
//   REDFT11 = DCT-IV
//
// DST Types:
//   RODFT00 = DST-I
//   RODFT10 = DST-II
//   RODFT01 = DST-III
//   RODFT11 = DST-IV

use std::sync::Arc;

use rustdct::DctPlanner;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FftR2rKind {
    Redft00, // DCT-I
    Redft10, // DCT-II
    Redft01, // DCT-III
    Redft11, // DCT-IV
    Rodft00, // DST-I
    Rodft10, // DST-II
    Rodft01, // DST-III
    Rodft11, // DST-IV
}

#[derive(Clone)]
enum DctDst {
    Dct1(Arc<dyn rustdct::Dct1<f32>>),
    Dct2(Arc<dyn rustdct::Dct2<f32>>),
    Dct3(Arc<dyn rustdct::Dct3<f32>>),
    Dct4(Arc<dyn rustdct::Dct4<f32>>),
    Dst1(Arc<dyn rustdct::Dst1<f32>>),
    Dst2(Arc<dyn rustdct::Dst2<f32>>),
    Dst3(Arc<dyn rustdct::Dst3<f32>>),
    Dst4(Arc<dyn rustdct::Dst4<f32>>),
}

#[derive(Clone)]
pub struct FftR2r {
    n: usize,
    kind: FftR2rKind,
    transform: DctDst,
}

impl FftR2r {
    pub fn new(n: usize, kind: FftR2rKind) -> Self {
        let mut planner = DctPlanner::new();

        let transform = match kind {
            FftR2rKind::Redft00 => DctDst::Dct1(planner.plan_dct1(n)),
            FftR2rKind::Redft10 => DctDst::Dct2(planner.plan_dct2(n)),
            FftR2rKind::Redft01 => DctDst::Dct3(planner.plan_dct3(n)),
            FftR2rKind::Redft11 => DctDst::Dct4(planner.plan_dct4(n)),
            FftR2rKind::Rodft00 => DctDst::Dst1(planner.plan_dst1(n)),
            FftR2rKind::Rodft10 => DctDst::Dst2(planner.plan_dst2(n)),
            FftR2rKind::Rodft01 => DctDst::Dst3(planner.plan_dst3(n)),
            FftR2rKind::Rodft11 => DctDst::Dst4(planner.plan_dst4(n)),
        };

        Self { n, kind, transform }
    }

    pub fn run(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(y.len(), self.n);

        y.copy_from_slice(x);

        match &self.transform {
            DctDst::Dct1(dct) => dct.process_dct1(y),
            DctDst::Dct2(dct) => dct.process_dct2(y),
            DctDst::Dct3(dct) => dct.process_dct3(y),
            DctDst::Dct4(dct) => dct.process_dct4(y),
            DctDst::Dst1(dst) => dst.process_dst1(y),
            DctDst::Dst2(dst) => dst.process_dst2(y),
            DctDst::Dst3(dst) => dst.process_dst3(y),
            DctDst::Dst4(dst) => dst.process_dst4(y),
        }

        // Apply scaling factor of 2 to match FFTW/liquid-dsp convention
        for yi in y.iter_mut() {
            *yi *= 2.0;
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn kind(&self) -> FftR2rKind {
        self.kind
    }
}

impl std::fmt::Debug for FftR2r {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FftR2r {{ n: {}, kind: {:?} }}", self.n, self.kind)
    }
}

pub fn fft_r2r_run(x: &[f32], y: &mut [f32], kind: FftR2rKind) {
    let fft = FftR2r::new(x.len(), kind);
    fft.run(x, y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    fn fft_r2r_test(x: &[f32], expected: &[f32], kind: FftR2rKind) {
        let n = x.len();
        let tol = 1e-4;

        let mut y = vec![0.0f32; n];
        fft_r2r_run(x, &mut y, kind);

        for i in 0..n {
            assert_abs_diff_eq!(y[i], expected[i], epsilon = tol);
        }
    }

    include!("test_data_r2r.rs");

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT00_n8)]
    fn test_fft_r2r_redft00_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_REDFT00_Y8, FftR2rKind::Redft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT10_n8)]
    fn test_fft_r2r_redft10_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_REDFT10_Y8, FftR2rKind::Redft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT01_n8)]
    fn test_fft_r2r_redft01_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_REDFT01_Y8, FftR2rKind::Redft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT11_n8)]
    fn test_fft_r2r_redft11_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_REDFT11_Y8, FftR2rKind::Redft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT00_n8)]
    fn test_fft_r2r_rodft00_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_RODFT00_Y8, FftR2rKind::Rodft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT10_n8)]
    fn test_fft_r2r_rodft10_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_RODFT10_Y8, FftR2rKind::Rodft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT01_n8)]
    fn test_fft_r2r_rodft01_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_RODFT01_Y8, FftR2rKind::Rodft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT11_n8)]
    fn test_fft_r2r_rodft11_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_RODFT11_Y8, FftR2rKind::Rodft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT00_n32)]
    fn test_fft_r2r_redft00_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_REDFT00_Y32, FftR2rKind::Redft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT10_n32)]
    fn test_fft_r2r_redft10_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_REDFT10_Y32, FftR2rKind::Redft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT01_n32)]
    fn test_fft_r2r_redft01_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_REDFT01_Y32, FftR2rKind::Redft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT11_n32)]
    fn test_fft_r2r_redft11_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_REDFT11_Y32, FftR2rKind::Redft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT00_n32)]
    fn test_fft_r2r_rodft00_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_RODFT00_Y32, FftR2rKind::Rodft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT10_n32)]
    fn test_fft_r2r_rodft10_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_RODFT10_Y32, FftR2rKind::Rodft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT01_n32)]
    fn test_fft_r2r_rodft01_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_RODFT01_Y32, FftR2rKind::Rodft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT11_n32)]
    fn test_fft_r2r_rodft11_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_RODFT11_Y32, FftR2rKind::Rodft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT00_n27)]
    fn test_fft_r2r_redft00_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_REDFT00_Y27, FftR2rKind::Redft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT10_n27)]
    fn test_fft_r2r_redft10_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_REDFT10_Y27, FftR2rKind::Redft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT01_n27)]
    fn test_fft_r2r_redft01_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_REDFT01_Y27, FftR2rKind::Redft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT11_n27)]
    fn test_fft_r2r_redft11_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_REDFT11_Y27, FftR2rKind::Redft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT00_n27)]
    fn test_fft_r2r_rodft00_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_RODFT00_Y27, FftR2rKind::Rodft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT10_n27)]
    fn test_fft_r2r_rodft10_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_RODFT10_Y27, FftR2rKind::Rodft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT01_n27)]
    fn test_fft_r2r_rodft01_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_RODFT01_Y27, FftR2rKind::Rodft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT11_n27)]
    fn test_fft_r2r_rodft11_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_RODFT11_Y27, FftR2rKind::Rodft11);
    }
}