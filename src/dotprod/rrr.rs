// Real-Real-Real dot product: [f32] · [f32] -> f32

use super::DotProd;

#[cfg(feature = "simd")]
use std::simd::{f32x4, f32x8, f32x16, StdFloat};
#[cfg(feature = "simd")]
use std::sync::OnceLock;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use super::reduce::{reduce_sum_sse_f32x4, reduce_sum_avx2_f32x8, reduce_sum_avx512_f32x16};

#[cfg(feature = "simd")]
type DotProdRrrFn = unsafe fn(&[f32], &[f32]) -> f32;
#[cfg(feature = "simd")]
static DOTPROD_RRR: OnceLock<DotProdRrrFn> = OnceLock::new();

impl DotProd<f32> for [f32] {
    type Output = f32;

    #[cfg(not(feature = "simd"))]
    fn dotprod(&self, other: &[f32]) -> f32 {
        self.iter().zip(other).map(|(a, b)| a * b).sum()
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[f32]) -> f32 {
        let f = DOTPROD_RRR.get_or_init(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    return dotprod_rrr_avx512;
                }
                if is_x86_feature_detected!("avx2") {
                    return dotprod_rrr_avx2;
                }
                if is_x86_feature_detected!("sse") {
                    return dotprod_rrr_sse;
                }
            }
            dotprod_rrr_scalar
        });
        unsafe { f(self, other) }
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

// Scalar fallback
#[cfg(feature = "simd")]
unsafe fn dotprod_rrr_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "sse")]
unsafe fn dotprod_rrr_sse(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 16 {
        return dotprod_rrr_scalar(a, b);
    }
    dotprod_rrr_sse_f32x4(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "sse")]
unsafe fn dotprod_rrr_sse_f32x4(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = f32x4::splat(0.0);
    let mut sum1 = f32x4::splat(0.0);
    let mut sum2 = f32x4::splat(0.0);
    let mut sum3 = f32x4::splat(0.0);

    let chunks = a.len() / 16;
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
    let mut result = reduce_sum_sse_f32x4(sum0);

    for (a, b) in a[chunks * 16..].iter().zip(&b[chunks * 16..]) {
        result += a * b;
    }
    result
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn dotprod_rrr_avx2(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 16 {
        return dotprod_rrr_scalar(a, b);
    } else if a.len() < 32 {
        return dotprod_rrr_sse_f32x4(a, b);
    }
    dotprod_rrr_avx2_f32x8(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn dotprod_rrr_avx2_f32x8(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = f32x8::splat(0.0);
    let mut sum1 = f32x8::splat(0.0);
    let mut sum2 = f32x8::splat(0.0);
    let mut sum3 = f32x8::splat(0.0);

    let chunks = a.len() / 32;
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
    let mut result = reduce_sum_avx2_f32x8(sum0);

    for (a, b) in a[chunks * 32..].iter().zip(&b[chunks * 32..]) {
        result += a * b;
    }
    result
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_rrr_avx512(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 16 {
        return dotprod_rrr_scalar(a, b);
    } else if a.len() < 64 {
        return dotprod_rrr_sse_f32x4(a, b);
    }
    dotprod_rrr_avx512_f32x16(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_rrr_avx512_f32x16(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = f32x16::splat(0.0);
    let mut sum1 = f32x16::splat(0.0);
    let mut sum2 = f32x16::splat(0.0);
    let mut sum3 = f32x16::splat(0.0);

    let chunks = a.len() / 64;
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
    let mut result = reduce_sum_avx512_f32x16(sum0);

    for (a, b) in a[chunks * 64..].iter().zip(&b[chunks * 64..]) {
        result += a * b;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::Rng;
    use test_macro::autotest_annotate;

    // Direct tests for each SIMD tier
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_rrr_avx2_direct() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = rand::thread_rng();
        const TOL: f32 = 1e-3;

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
            let x: Vec<f32> = (0..n).map(|_| rng.gen()).collect();

            let y_test: f32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let y_avx2 = unsafe { dotprod_rrr_avx2(&h, &x) };

            assert_abs_diff_eq!(y_avx2, y_test, epsilon = TOL);
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_rrr_sse_direct() {
        if !is_x86_feature_detected!("sse") {
            return;
        }
        let mut rng = rand::thread_rng();
        const TOL: f32 = 1e-3;

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
            let x: Vec<f32> = (0..n).map(|_| rng.gen()).collect();

            let y_test: f32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let y_sse = unsafe { dotprod_rrr_sse(&h, &x) };

            assert_abs_diff_eq!(y_sse, y_test, epsilon = TOL);
        }
    }

    #[test]
    fn test_dotprod_rrr() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(a.dotprod(&b), 32.0);
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
    #[autotest_annotate(autotest_dotprod_rrrf_rand02)]
    fn test_dotprod_rrrf_rand02() {
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: Vec<f32> = vec![
             2.595300,    1.243600,   -0.818550,   -1.439800,
             0.055795,   -1.476000,    0.445900,    0.325460,
            -3.451200,    0.058528,   -0.246990,    0.476290,
            -0.598780,   -0.885250,    0.464660,   -0.610140,
        ];

        #[rustfmt::skip]
        let x: Vec<f32> = vec![
            -0.917010,   -1.278200,   -0.533190,    2.309200,
             0.592980,    0.964820,    0.183220,   -0.082864,
             0.057171,   -1.186500,   -0.738260,    0.356960,
            -0.144000,   -1.435200,   -0.893420,    1.657800,
        ];

        let test = -8.17832326680587;
        assert_abs_diff_eq!(h.dotprod(&x), test, epsilon = TOL);

        let test_rev = 4.56839328512000;
        assert_abs_diff_eq!(h.iter().rev().cloned().collect::<Vec<f32>>().dotprod(&x), test_rev, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_struct_lengths)]
    fn test_dotprod_rrrf_struct_lengths() {
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

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_struct_vs_ordinal)]
    fn test_dotprod_rrrf_struct_vs_ordinal() {
        const TOL: f32 = 1e-4;
        let mut rng = rand::thread_rng();

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
            let x: Vec<f32> = (0..n).map(|_| rng.gen()).collect();

            let y_test: f32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let y_struct = h.dotprod(&x);

            assert_abs_diff_eq!(y_struct, y_test, epsilon = TOL);
        }
    }
}
