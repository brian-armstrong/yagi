// Real-to-real FFT transforms (DCT/DST).

use std::sync::Arc;

use rustdct::DctPlanner;

/// Kind of real-to-real transform.
///
/// Variant names and definitions follow FFTW's real-to-real transforms. The
/// transforms are unnormalized. Applying a kind's inverse partner to its
/// output returns the input scaled by `2 * N` for a length `N`, except that
/// the DCT-I scale is `2 * (N - 1)` and the DST-I scale is `2 * (N + 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[doc(alias = "FftR2rKind")]
pub enum RealToRealFftKind {
    /// DCT-I, its own inverse partner.
    Redft00,
    /// DCT-II.
    Redft10,
    /// DCT-III, the inverse partner of [`Self::Redft10`].
    Redft01,
    /// DCT-IV, its own inverse partner.
    Redft11,
    /// DST-I, its own inverse partner.
    Rodft00,
    /// DST-II.
    Rodft10,
    /// DST-III, the inverse partner of [`Self::Rodft10`].
    Rodft01,
    /// DST-IV, its own inverse partner.
    Rodft11,
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

/// A reusable real-to-real DCT or DST plan.
///
/// The transform has a fixed length and [`RealToRealFftKind`], and is
/// unnormalized. See [`RealToRealFftKind`] for the scale of a round trip.
#[derive(Clone)]
#[doc(alias = "FftR2r")]
pub struct RealToRealFft {
    n: usize,
    kind: RealToRealFftKind,
    transform: DctDst,
}

impl RealToRealFft {
    /// Creates a real-to-real transform plan for `size` samples.
    ///
    /// # Panics
    ///
    /// Panics when `size` is one and `kind` is
    /// [`RealToRealFftKind::Redft00`]. Planning a zero-length transform can
    /// also panic for some kinds.
    pub fn new(size: usize, kind: RealToRealFftKind) -> Self {
        let mut planner = DctPlanner::new();

        let transform = match kind {
            RealToRealFftKind::Redft00 => DctDst::Dct1(planner.plan_dct1(size)),
            RealToRealFftKind::Redft10 => DctDst::Dct2(planner.plan_dct2(size)),
            RealToRealFftKind::Redft01 => DctDst::Dct3(planner.plan_dct3(size)),
            RealToRealFftKind::Redft11 => DctDst::Dct4(planner.plan_dct4(size)),
            RealToRealFftKind::Rodft00 => DctDst::Dst1(planner.plan_dst1(size)),
            RealToRealFftKind::Rodft10 => DctDst::Dst2(planner.plan_dst2(size)),
            RealToRealFftKind::Rodft01 => DctDst::Dst3(planner.plan_dst3(size)),
            RealToRealFftKind::Rodft11 => DctDst::Dst4(planner.plan_dst4(size)),
        };

        Self { n: size, kind, transform }
    }

    /// Transforms `input` and writes the result to `output`.
    ///
    /// # Panics
    ///
    /// Panics unless both slices have this plan's length.
    pub fn run(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.n);
        assert_eq!(output.len(), self.n);

        output.copy_from_slice(input);

        match &self.transform {
            DctDst::Dct1(dct) => dct.process_dct1(output),
            DctDst::Dct2(dct) => dct.process_dct2(output),
            DctDst::Dct3(dct) => dct.process_dct3(output),
            DctDst::Dct4(dct) => dct.process_dct4(output),
            DctDst::Dst1(dst) => dst.process_dst1(output),
            DctDst::Dst2(dst) => dst.process_dst2(output),
            DctDst::Dst3(dst) => dst.process_dst3(output),
            DctDst::Dst4(dst) => dst.process_dst4(output),
        }

        // Apply scaling factor of 2 to match FFTW/liquid-dsp convention
        for yi in output.iter_mut() {
            *yi *= 2.0;
        }
    }

    /// Returns the transform length.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns whether the transform length is zero.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the kind of transform this plan computes.
    pub fn kind(&self) -> RealToRealFftKind {
        self.kind
    }
}

impl std::fmt::Debug for RealToRealFft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RealToRealFft {{ n: {}, kind: {:?} }}", self.n, self.kind)
    }
}

/// Transforms one real slice without retaining a plan.
///
/// The transform length is `x.len()`. Reuse [`RealToRealFft`] when
/// transforming several slices with the same length and kind.
///
/// # Panics
///
/// Panics when `y.len() != x.len()`, or when `x` has length one with
/// [`RealToRealFftKind::Redft00`]. Zero-length inputs can also panic during
/// planning.
pub fn fft_r2r_run(x: &[f32], y: &mut [f32], kind: RealToRealFftKind) {
    let fft = RealToRealFft::new(x.len(), kind);
    fft.run(x, y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    fn fft_r2r_test(x: &[f32], expected: &[f32], kind: RealToRealFftKind) {
        let n = x.len();
        let tol = 1e-4;

        let mut y = vec![0.0f32; n];
        fft_r2r_run(x, &mut y, kind);

        for i in 0..n {
            assert_abs_diff_eq!(y[i], expected[i], epsilon = tol);
        }
    }

    include!("test_data_real_to_real.rs");

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT00_n8)]
    fn test_fft_r2r_redft00_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_REDFT00_Y8, RealToRealFftKind::Redft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT10_n8)]
    fn test_fft_r2r_redft10_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_REDFT10_Y8, RealToRealFftKind::Redft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT01_n8)]
    fn test_fft_r2r_redft01_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_REDFT01_Y8, RealToRealFftKind::Redft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT11_n8)]
    fn test_fft_r2r_redft11_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_REDFT11_Y8, RealToRealFftKind::Redft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT00_n8)]
    fn test_fft_r2r_rodft00_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_RODFT00_Y8, RealToRealFftKind::Rodft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT10_n8)]
    fn test_fft_r2r_rodft10_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_RODFT10_Y8, RealToRealFftKind::Rodft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT01_n8)]
    fn test_fft_r2r_rodft01_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_RODFT01_Y8, RealToRealFftKind::Rodft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT11_n8)]
    fn test_fft_r2r_rodft11_n8() {
        fft_r2r_test(&FFTDATA_R2R_X8, &FFTDATA_R2R_RODFT11_Y8, RealToRealFftKind::Rodft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT00_n32)]
    fn test_fft_r2r_redft00_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_REDFT00_Y32, RealToRealFftKind::Redft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT10_n32)]
    fn test_fft_r2r_redft10_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_REDFT10_Y32, RealToRealFftKind::Redft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT01_n32)]
    fn test_fft_r2r_redft01_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_REDFT01_Y32, RealToRealFftKind::Redft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT11_n32)]
    fn test_fft_r2r_redft11_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_REDFT11_Y32, RealToRealFftKind::Redft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT00_n32)]
    fn test_fft_r2r_rodft00_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_RODFT00_Y32, RealToRealFftKind::Rodft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT10_n32)]
    fn test_fft_r2r_rodft10_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_RODFT10_Y32, RealToRealFftKind::Rodft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT01_n32)]
    fn test_fft_r2r_rodft01_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_RODFT01_Y32, RealToRealFftKind::Rodft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT11_n32)]
    fn test_fft_r2r_rodft11_n32() {
        fft_r2r_test(&FFTDATA_R2R_X32, &FFTDATA_R2R_RODFT11_Y32, RealToRealFftKind::Rodft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT00_n27)]
    fn test_fft_r2r_redft00_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_REDFT00_Y27, RealToRealFftKind::Redft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT10_n27)]
    fn test_fft_r2r_redft10_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_REDFT10_Y27, RealToRealFftKind::Redft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT01_n27)]
    fn test_fft_r2r_redft01_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_REDFT01_Y27, RealToRealFftKind::Redft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_REDFT11_n27)]
    fn test_fft_r2r_redft11_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_REDFT11_Y27, RealToRealFftKind::Redft11);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT00_n27)]
    fn test_fft_r2r_rodft00_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_RODFT00_Y27, RealToRealFftKind::Rodft00);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT10_n27)]
    fn test_fft_r2r_rodft10_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_RODFT10_Y27, RealToRealFftKind::Rodft10);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT01_n27)]
    fn test_fft_r2r_rodft01_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_RODFT01_Y27, RealToRealFftKind::Rodft01);
    }

    #[test]
    #[autotest_annotate(autotest_fft_r2r_RODFT11_n27)]
    fn test_fft_r2r_rodft11_n27() {
        fft_r2r_test(&FFTDATA_R2R_X27, &FFTDATA_R2R_RODFT11_Y27, RealToRealFftKind::Rodft11);
    }
}
