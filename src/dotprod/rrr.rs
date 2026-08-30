// Real-Real-Real dot product: [f32] · [f32] -> f32

use super::DotProd;

#[cfg(feature = "simd")]
use std::simd::f32x4;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use std::simd::{f32x8, f32x16, StdFloat};
#[cfg(feature = "simd")]
use std::sync::OnceLock;

#[cfg(feature = "simd")]
use super::reduce::reduce_sum_sse_f32x4;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use super::reduce::{reduce_sum_avx2_f32x8, reduce_sum_avx512_f32x16};

#[cfg(feature = "simd")]
type DotProdRrrFn = unsafe fn(&[f32], &[f32]) -> f32;
#[cfg(feature = "simd")]
static DOTPROD_RRR: OnceLock<DotProdRrrFn> = OnceLock::new();

#[cfg(feature = "simd")]
const DOTPROD_RRR_WIDE_CUTOFF: usize = 32;

#[cfg(feature = "simd")]
macro_rules! plan_dotprod_rrr_const_f32x4 {
    ($len:expr; $($n:literal),+ $(,)?) => {
        match $len {
            $(
                $n => Some(dotprod_rrr_const_f32x4::<$n> as DotProdRrrFn),
            )+
            _ => None,
        }
    };
}

impl DotProd<f32> for [f32] {
    type Output = f32;

    #[cfg(not(feature = "simd"))]
    fn dotprod(&self, other: &[f32]) -> f32 {
        assert_eq!(self.len(), other.len(), "Slices must have equal length");
        dotprod_rrr_scalar(self, other)
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[f32]) -> f32 {
        assert_eq!(self.len(), other.len(), "Slices must have equal length");
        if self.len() < DOTPROD_RRR_WIDE_CUTOFF {
            return unsafe { dotprod_rrr_128(self, other) };
        }
        let f = DOTPROD_RRR.get_or_init(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    return dotprod_rrr_avx512;
                }
                if is_x86_feature_detected!("avx2") {
                    return dotprod_rrr_avx2;
                }
            }
            dotprod_rrr_128
        });
        unsafe { f(self, other) }
    }

    #[cfg(feature = "simd")]
    fn plan(len: usize) -> super::DotProdKernel<f32, f32, f32> {
        // use the const 128 impl for lengths up to 47.
        // at 48, the wider dynamic paths catch up
        if let Some(f) = plan_dotprod_rrr_const_f32x4!(
            len;
             1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47,
        ) {
            return f;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                return dotprod_rrr_avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return dotprod_rrr_avx2;
            }
        }
        dotprod_rrr_128
    }
}

impl DotProd<f32> for std::collections::VecDeque<f32> {
    type Output = f32;

    fn dotprod(&self, other: &[f32]) -> f32 {
        let (l, r) = self.as_slices();
        let split_idx = l.len();
        let l_sum = l.dotprod(&other[..split_idx]);
        let r_sum = r.dotprod(&other[split_idx..]);
        l_sum + r_sum
    }
}

#[cfg(not(feature = "simd"))]
fn dotprod_rrr_scalar_wide(a: &[f32], b: &[f32]) -> (f32, usize) {
    let chunks = a.len() / 2;
    if chunks == 0 {
        return (0.0, 0);
    }

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;

    // two concurrent sums exposes a little ILP
    for i in 0..chunks {
        let base = i * 2;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
    }

    (sum0 + sum1, chunks * 2)
}

#[cfg(not(feature = "simd"))]
fn dotprod_rrr_scalar(a: &[f32], b: &[f32]) -> f32 {
    let (sum, n) = dotprod_rrr_scalar_wide(a, b);
    sum + a[n..].iter().zip(&b[n..]).map(|(a, b)| a * b).sum::<f32>()
}

#[cfg(feature = "simd")]
unsafe fn dotprod_rrr_scalar(a: &[f32], b: &[f32]) -> f32 {
    // the SIMD scalar tail doesn't run on many elements, so stay narrow
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(feature = "simd")]
unsafe fn dotprod_rrr_const_f32x4<const N: usize>(a: &[f32], b: &[f32]) -> f32 {
    // this method is const on N so that the loops below disappear
    // SIMD uses this for various small N where this is small and fast 
    debug_assert_eq!(a.len(), N);
    debug_assert_eq!(b.len(), N);
    unsafe {
        std::hint::assert_unchecked(a.len() == N);
        std::hint::assert_unchecked(b.len() == N);
    }

    let mut sum = f32x4::splat(0.0);
    let mut i = 0;
    while i + 4 <= N {
        let av = f32x4::from_array(*(a.as_ptr().add(i) as *const [f32; 4]));
        let bv = f32x4::from_array(*(b.as_ptr().add(i) as *const [f32; 4]));
        sum += av * bv;
        i += 4;
    }

    let lanes = sum.to_array();
    let mut result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < N {
        result += a[i] * b[i];
        i += 1;
    }
    result
}

#[cfg(feature = "simd")]
unsafe fn dotprod_rrr_128(a: &[f32], b: &[f32]) -> f32 {
    // generic f32x4 dotprod that will work nicely on many archs
    let (sum, n) = dotprod_rrr_128_f32x4_wide(a, b);
    sum + dotprod_rrr_scalar(&a[n..], &b[n..])
}

/// 4x-unrolled f32x4: 16 elements per iteration.
#[cfg(feature = "simd")]
#[inline]
unsafe fn dotprod_rrr_128_f32x4_wide(a: &[f32], b: &[f32]) -> (f32, usize) {
    let chunks = a.len() / 16;
    if chunks == 0 {
        return (0.0, 0);
    }

    let mut sum0 = f32x4::splat(0.0);
    let mut sum1 = f32x4::splat(0.0);
    let mut sum2 = f32x4::splat(0.0);
    let mut sum3 = f32x4::splat(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let base = i * 16;
        let a0 = f32x4::from_array(*(a_ptr.add(base) as *const [f32; 4]));
        let b0 = f32x4::from_array(*(b_ptr.add(base) as *const [f32; 4]));
        let a1 = f32x4::from_array(*(a_ptr.add(base + 4) as *const [f32; 4]));
        let b1 = f32x4::from_array(*(b_ptr.add(base + 4) as *const [f32; 4]));
        let a2 = f32x4::from_array(*(a_ptr.add(base + 8) as *const [f32; 4]));
        let b2 = f32x4::from_array(*(b_ptr.add(base + 8) as *const [f32; 4]));
        let a3 = f32x4::from_array(*(a_ptr.add(base + 12) as *const [f32; 4]));
        let b3 = f32x4::from_array(*(b_ptr.add(base + 12) as *const [f32; 4]));

        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }

    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;
    (reduce_sum_sse_f32x4(sum0), chunks * 16)
}

/// Single f32x4: 4 elements per iteration.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
unsafe fn dotprod_rrr_128_f32x4_narrow(a: &[f32], b: &[f32]) -> (f32, usize) {
    let chunks = a.len() / 4;
    if chunks == 0 {
        return (0.0, 0);
    }

    let mut sum = f32x4::splat(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let base = i * 4;
        let av = f32x4::from_array(*(a_ptr.add(base) as *const [f32; 4]));
        let bv = f32x4::from_array(*(b_ptr.add(base) as *const [f32; 4]));
        sum += av * bv;
    }

    (reduce_sum_sse_f32x4(sum), chunks * 4)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn dotprod_rrr_avx2(a: &[f32], b: &[f32]) -> f32 {
    let (s0, n0) = dotprod_rrr_avx2_f32x8_wide(a, b);
    let (a, b) = (&a[n0..], &b[n0..]);
    let (s1, n1) = dotprod_rrr_128_f32x4_narrow(a, b);
    s0 + s1 + dotprod_rrr_scalar(&a[n1..], &b[n1..])
}

/// 4x-unrolled f32x8: 32 elements per iteration.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dotprod_rrr_avx2_f32x8_wide(a: &[f32], b: &[f32]) -> (f32, usize) {
    let chunks = a.len() / 32;
    if chunks == 0 {
        return (0.0, 0);
    }

    let mut sum0 = f32x8::splat(0.0);
    let mut sum1 = f32x8::splat(0.0);
    let mut sum2 = f32x8::splat(0.0);
    let mut sum3 = f32x8::splat(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let base = i * 32;
        let a0 = f32x8::from_array(*(a_ptr.add(base) as *const [f32; 8]));
        let b0 = f32x8::from_array(*(b_ptr.add(base) as *const [f32; 8]));
        let a1 = f32x8::from_array(*(a_ptr.add(base + 8) as *const [f32; 8]));
        let b1 = f32x8::from_array(*(b_ptr.add(base + 8) as *const [f32; 8]));
        let a2 = f32x8::from_array(*(a_ptr.add(base + 16) as *const [f32; 8]));
        let b2 = f32x8::from_array(*(b_ptr.add(base + 16) as *const [f32; 8]));
        let a3 = f32x8::from_array(*(a_ptr.add(base + 24) as *const [f32; 8]));
        let b3 = f32x8::from_array(*(b_ptr.add(base + 24) as *const [f32; 8]));

        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }

    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;
    (reduce_sum_avx2_f32x8(sum0), chunks * 32)
}

/// Single f32x8: 8 elements per iteration.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dotprod_rrr_avx2_f32x8_narrow(a: &[f32], b: &[f32]) -> (f32, usize) {
    let chunks = a.len() / 8;
    if chunks == 0 {
        return (0.0, 0);
    }

    let mut sum = f32x8::splat(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let base = i * 8;
        let av = f32x8::from_array(*(a_ptr.add(base) as *const [f32; 8]));
        let bv = f32x8::from_array(*(b_ptr.add(base) as *const [f32; 8]));
        sum += av * bv;
    }

    (reduce_sum_avx2_f32x8(sum), chunks * 8)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_rrr_avx512(a: &[f32], b: &[f32]) -> f32 {
    let (s0, n0) = dotprod_rrr_avx512_f32x16_wide(a, b);
    let (a, b) = (&a[n0..], &b[n0..]);
    let (s1, n1) = dotprod_rrr_avx2_f32x8_narrow(a, b);
    s0 + s1 + dotprod_rrr_scalar(&a[n1..], &b[n1..])
}

/// 4x-unrolled f32x16: 64 elements per iteration.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn dotprod_rrr_avx512_f32x16_wide(a: &[f32], b: &[f32]) -> (f32, usize) {
    let chunks = a.len() / 64;
    if chunks == 0 {
        return (0.0, 0);
    }

    let mut sum0 = f32x16::splat(0.0);
    let mut sum1 = f32x16::splat(0.0);
    let mut sum2 = f32x16::splat(0.0);
    let mut sum3 = f32x16::splat(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let base = i * 64;
        let a0 = f32x16::from_array(*(a_ptr.add(base) as *const [f32; 16]));
        let b0 = f32x16::from_array(*(b_ptr.add(base) as *const [f32; 16]));
        let a1 = f32x16::from_array(*(a_ptr.add(base + 16) as *const [f32; 16]));
        let b1 = f32x16::from_array(*(b_ptr.add(base + 16) as *const [f32; 16]));
        let a2 = f32x16::from_array(*(a_ptr.add(base + 32) as *const [f32; 16]));
        let b2 = f32x16::from_array(*(b_ptr.add(base + 32) as *const [f32; 16]));
        let a3 = f32x16::from_array(*(a_ptr.add(base + 48) as *const [f32; 16]));
        let b3 = f32x16::from_array(*(b_ptr.add(base + 48) as *const [f32; 16]));

        sum0 = a0.mul_add(b0, sum0);
        sum1 = a1.mul_add(b1, sum1);
        sum2 = a2.mul_add(b2, sum2);
        sum3 = a3.mul_add(b3, sum3);
    }

    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;
    (reduce_sum_avx512_f32x16(sum0), chunks * 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;
    #[cfg(feature = "simd")]
    use crate::random::randnf;

    #[test]
    fn test_dotprod_rrr() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(a.dotprod(&b), 32.0);
    }

    #[test]
    #[should_panic(expected = "Slices must have equal length")]
    fn test_dotprod_rrr_mismatched_lengths() {
        [1.0f32; 32].dotprod(&[1.0f32; 31]);
    }

    #[test]
    fn test_dotprod_rrr_boundary_lengths() {
        const TOL: f32 = 1e-3;

        // Test specific lengths that hit tier boundaries and cleanup paths
        // SSE: <16 scalar, >=16 uses f32x4 (processes 16 per iteration)
        // AVX2: <16 scalar, <32 f32x4, >=32 f32x8 (processes 32 per iteration)
        // AVX-512: <16 scalar, <64 f32x4, >=64 f32x16 (processes 64 per iteration)
        let test_sizes = [
            1, 2, 3, 4,           // tiny scalar
            15, 16, 17,           // scalar/f32x4 boundary
            31, 32, 33,           // f32x4/f32x8 boundary (AVX2)
            63, 64, 65,           // f32x4/f32x16 boundary (AVX-512)
            127, 128, 129,        // 2x f32x16 + cleanup
            255, 256, 257,        // 4x f32x16 + cleanup
            511, 512, 513,        // 8x f32x16 + cleanup
        ];

        for &n in &test_sizes {
            let h: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

            let expected: f32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let result = h.dotprod(&x);

            assert_abs_diff_eq!(result, expected, epsilon = TOL);
        }
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_basic)]
    fn test_dotprod_rrrf_basic() {
        const TOL: f32 = 1e-6;
        let h: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

        let x0 = vec![0.0; 16];
        assert_abs_diff_eq!(h.dotprod(&x0), 0.0, epsilon = TOL);

        let x1 = vec![1.0; 16];
        assert_abs_diff_eq!(h.dotprod(&x1), 0.0, epsilon = TOL);

        let x2: Vec<f32> = (0..16).map(|i| (i % 2) as f32).collect();
        assert_abs_diff_eq!(h.dotprod(&x2), -8.0, epsilon = TOL);

        let x3: Vec<f32> = (0..16).map(|i| 1.0 - (i % 2) as f32).collect();
        assert_abs_diff_eq!(h.dotprod(&x3), 8.0, epsilon = TOL);

        assert_abs_diff_eq!(h.dotprod(&h), 16.0, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_uneven)]
    fn test_dotprod_rrrf_uneven() {
        const TOL: f32 = 1e-6;
        let h: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let x = vec![1.0; 16];

        assert_abs_diff_eq!(h[..1].dotprod(&x[..1]), 1.0, epsilon = TOL);
        assert_abs_diff_eq!(h[..2].dotprod(&x[..2]), 0.0, epsilon = TOL);
        assert_abs_diff_eq!(h[..3].dotprod(&x[..3]), 1.0, epsilon = TOL);
        assert_abs_diff_eq!(h[..11].dotprod(&x[..11]), 1.0, epsilon = TOL);
        assert_abs_diff_eq!(h[..13].dotprod(&x[..13]), 1.0, epsilon = TOL);
        assert_abs_diff_eq!(h[..15].dotprod(&x[..15]), 1.0, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_rand01)]
    fn test_dotprod_rrrf_rand01() {
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: Vec<f32> = vec![
            -0.050565,   -0.952580,    0.274320,    1.232400,
             1.268200,    0.565770,    0.800830,    0.923970,
             0.517060,   -0.530340,   -0.378550,   -1.127100,
             1.123100,   -1.006000,   -1.483800,   -0.062007,
        ];

        #[rustfmt::skip]
        let x: Vec<f32> = vec![
            -0.384280,   -0.812030,    0.156930,    1.919500,
             0.564580,   -0.123610,   -0.138640,    0.004984,
            -1.100200,   -0.497620,    0.089977,   -1.745500,
             0.463640,    0.592100,    1.150000,   -1.225400,
        ];

        let test = 3.66411513609863;
        assert_abs_diff_eq!(h.dotprod(&x), test, epsilon = TOL);
    }

    #[test]
    fn test_dotprod_rrrf_lengths() {
        const TOL: f32 = 2e-6;

        #[rustfmt::skip]
        let x: Vec<f32> = vec![
             0.03117498,  -1.54311769,  -0.58759073,  -0.73882202,
             0.86592259,  -0.26669417,  -0.70153724,  -1.24555787,
            -1.09272288,  -1.41984975,  -1.40299260,   0.95861481,
            -0.67361246,   2.05305710,   1.26576873,  -0.77474848,
            -0.93143252,  -1.05724660,   0.21455006,   1.07554168,
            -0.46703810,   0.68878404,  -1.11900266,  -0.52016966,
             0.61400744,  -0.46506142,  -0.16801031,   0.48237303,
             0.51286055,  -0.57239385,  -0.64462740,  -0.75596668,
             1.95612355,  -0.47917908,   0.52384983,
        ];

        #[rustfmt::skip]
        let h: Vec<f32> = vec![
            -0.12380948,   0.88417134,   2.27373797,  -2.61506417,
             0.35022002,   0.07481393,   0.52984228,  -0.65542307,
            -2.14893606,   0.62466395,   0.07330391,  -1.28014856,
             0.16347776,   0.21238151,   0.05462232,  -0.60290942,
            -1.27658956,   3.05114996,   1.34789601,  -1.22098592,
             1.70899633,  -0.41002037,   3.08009931,  -1.39895771,
            -0.50875066,   0.25817865,   1.08668549,   0.05494174,
            -1.05337166,   1.26772604,   1.00369204,  -0.55129338,
             1.01828299,   0.76014664,  -0.15605569,
        ];

        assert_abs_diff_eq!(h[..32].dotprod(&x[..32]), -7.99577847, epsilon = TOL);
        assert_abs_diff_eq!(h[..33].dotprod(&x[..33]), -6.00389114, epsilon = TOL);
        assert_abs_diff_eq!(h[..34].dotprod(&x[..34]), -6.36813751, epsilon = TOL);
        assert_abs_diff_eq!(h[..35].dotprod(&x[..35]), -6.44988725, epsilon = TOL);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_rrr_avx512_direct() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| randnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: f64 = h.iter().zip(x.iter()).map(|(&a, &b)| a as f64 * b as f64).sum();
            let y_avx512 = unsafe { dotprod_rrr_avx512(&h, &x) };

            assert_abs_diff_eq!(y_avx512, y_test as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_rrr_avx2_direct() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| randnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: f64 = h.iter().zip(x.iter()).map(|(&a, &b)| a as f64 * b as f64).sum();
            let y_avx2 = unsafe { dotprod_rrr_avx2(&h, &x) };

            assert_abs_diff_eq!(y_avx2, y_test as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_dotprod_rrr_128_direct() {
        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| randnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: f64 = h.iter().zip(x.iter()).map(|(&a, &b)| a as f64 * b as f64).sum();
            let y_sse = unsafe { dotprod_rrr_128(&h, &x) };

            assert_abs_diff_eq!(y_sse, y_test as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_dotprod_rrr_scalar_direct() {
        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| randnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: f64 = h.iter().zip(x.iter()).map(|(&a, &b)| a as f64 * b as f64).sum();
            let y_scalar = unsafe { dotprod_rrr_scalar(&h, &x) };

            assert_abs_diff_eq!(y_scalar, y_test as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
        }
    }
}
