// Complex-Real-Complex dot product: [Complex<f32>] · [f32] -> Complex<f32>
// Also handles rcc: [f32] · [Complex<f32>] -> Complex<f32> via delegation

use num_complex::Complex;

use super::DotProd;

#[cfg(feature = "simd")]
use std::simd::{f32x4, f32x8, f32x16};
#[cfg(feature = "simd")]
use std::sync::OnceLock;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use super::reduce::{
    reduce_sum_complex_sse_f32x4, reduce_sum_complex_avx2_f32x8, reduce_sum_complex_avx512_f32x16,
};

#[cfg(feature = "simd")]
type DotProdCrcFn = unsafe fn(&[Complex<f32>], &[f32]) -> Complex<f32>;
#[cfg(feature = "simd")]
static DOTPROD_CRC: OnceLock<DotProdCrcFn> = OnceLock::new();

impl DotProd<Complex<f32>> for [f32] {
    type Output = Complex<f32>;

    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        other.dotprod(self)
    }
}

impl DotProd<f32> for [Complex<f32>] {
    type Output = Complex<f32>;

    #[cfg(not(feature = "simd"))]
    fn dotprod(&self, other: &[f32]) -> Complex<f32> {
        assert_eq!(self.len(), other.len(), "Slices must have equal length");
        self.iter().zip(other).map(|(a, b)| a * b).sum()
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[f32]) -> Complex<f32> {
        assert_eq!(self.len(), other.len(), "Slices must have equal length");
        let f = DOTPROD_CRC.get_or_init(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    return dotprod_crc_avx512;
                }
                if is_x86_feature_detected!("avx2") {
                    return dotprod_crc_avx2;
                }
                if is_x86_feature_detected!("sse") {
                    return dotprod_crc_sse;
                }
            }
            dotprod_crc_scalar
        });
        unsafe { f(self, other) }
    }
}

impl DotProd<Complex<f32>> for std::collections::VecDeque<f32> {
    type Output = Complex<f32>;

    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        let (l, r) = self.as_slices();
        let split_idx = l.len();
        let l_sum = l.dotprod(&other[..split_idx]);
        let r_sum = r.dotprod(&other[split_idx..]);
        l_sum + r_sum
    }
}

impl DotProd<f32> for std::collections::VecDeque<Complex<f32>> {
    type Output = Complex<f32>;

    fn dotprod(&self, other: &[f32]) -> Complex<f32> {
        let (l, r) = self.as_slices();
        let split_idx = l.len();
        let l_sum = l.dotprod(&other[..split_idx]);
        let r_sum = r.dotprod(&other[split_idx..]);
        l_sum + r_sum
    }
}

// Scalar fallback
#[cfg(feature = "simd")]
unsafe fn dotprod_crc_scalar(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "sse")]
unsafe fn dotprod_crc_sse(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    // 8 complex = 16 floats for the main loop (4 accumulators × 2 complex each)
    if a.len() < 8 {
        return dotprod_crc_scalar(a, b);
    }
    dotprod_crc_sse_f32x4(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "sse")]
unsafe fn dotprod_crc_sse_f32x4(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    let mut sum0 = f32x4::splat(0.0);
    let mut sum1 = f32x4::splat(0.0);
    let mut sum2 = f32x4::splat(0.0);
    let mut sum3 = f32x4::splat(0.0);

    let chunks = a.len() / 8;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let a_base = i * 16; // 8 complex × 2 floats
        let b_base = i * 8;

        // Load complex values (pairs of re, im)
        let a0 = f32x4::from_array(*(a_ptr.add(a_base) as *const [f32; 4]));
        let a1 = f32x4::from_array(*(a_ptr.add(a_base + 4) as *const [f32; 4]));
        let a2 = f32x4::from_array(*(a_ptr.add(a_base + 8) as *const [f32; 4]));
        let a3 = f32x4::from_array(*(a_ptr.add(a_base + 12) as *const [f32; 4]));

        // Duplicate real coefficients: [b0, b0, b1, b1] etc
        let bp = b_ptr.add(b_base);
        let b0 = f32x4::from_array([*bp, *bp, *bp.add(1), *bp.add(1)]);
        let b1 = f32x4::from_array([*bp.add(2), *bp.add(2), *bp.add(3), *bp.add(3)]);
        let b2 = f32x4::from_array([*bp.add(4), *bp.add(4), *bp.add(5), *bp.add(5)]);
        let b3 = f32x4::from_array([*bp.add(6), *bp.add(6), *bp.add(7), *bp.add(7)]);

        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }

    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;
    let mut result = reduce_sum_complex_sse_f32x4(sum0);

    // Handle remainder
    let a_rem = &a[chunks * 8..];
    let b_rem = &b[chunks * 8..];
    for (a, b) in a_rem.iter().zip(b_rem) {
        result += a * b;
    }
    result
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn dotprod_crc_avx2(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    if a.len() < 8 {
        return dotprod_crc_scalar(a, b);
    } else if a.len() < 16 {
        return dotprod_crc_sse_f32x4(a, b);
    }
    dotprod_crc_avx2_f32x8(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn dotprod_crc_avx2_f32x8(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    let mut sum0 = f32x8::splat(0.0);
    let mut sum1 = f32x8::splat(0.0);
    let mut sum2 = f32x8::splat(0.0);
    let mut sum3 = f32x8::splat(0.0);

    let chunks = a.len() / 16;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let a_base = i * 32; // 16 complex × 2 floats
        let b_base = i * 16;

        let a0 = f32x8::from_array(*(a_ptr.add(a_base) as *const [f32; 8]));
        let a1 = f32x8::from_array(*(a_ptr.add(a_base + 8) as *const [f32; 8]));
        let a2 = f32x8::from_array(*(a_ptr.add(a_base + 16) as *const [f32; 8]));
        let a3 = f32x8::from_array(*(a_ptr.add(a_base + 24) as *const [f32; 8]));

        let bp = b_ptr.add(b_base);
        let b0 = f32x8::from_array([
            *bp, *bp, *bp.add(1), *bp.add(1),
            *bp.add(2), *bp.add(2), *bp.add(3), *bp.add(3),
        ]);
        let b1 = f32x8::from_array([
            *bp.add(4), *bp.add(4), *bp.add(5), *bp.add(5),
            *bp.add(6), *bp.add(6), *bp.add(7), *bp.add(7),
        ]);
        let b2 = f32x8::from_array([
            *bp.add(8), *bp.add(8), *bp.add(9), *bp.add(9),
            *bp.add(10), *bp.add(10), *bp.add(11), *bp.add(11),
        ]);
        let b3 = f32x8::from_array([
            *bp.add(12), *bp.add(12), *bp.add(13), *bp.add(13),
            *bp.add(14), *bp.add(14), *bp.add(15), *bp.add(15),
        ]);

        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }

    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;
    let mut result = reduce_sum_complex_avx2_f32x8(sum0);

    let a_rem = &a[chunks * 16..];
    let b_rem = &b[chunks * 16..];
    for (a, b) in a_rem.iter().zip(b_rem) {
        result += a * b;
    }
    result
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_crc_avx512(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    if a.len() < 8 {
        return dotprod_crc_scalar(a, b);
    } else if a.len() < 32 {
        return dotprod_crc_sse_f32x4(a, b);
    }
    dotprod_crc_avx512_f32x16(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_crc_avx512_f32x16(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    let mut sum0 = f32x16::splat(0.0);
    let mut sum1 = f32x16::splat(0.0);
    let mut sum2 = f32x16::splat(0.0);
    let mut sum3 = f32x16::splat(0.0);

    let chunks = a.len() / 32;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let a_base = i * 64; // 32 complex × 2 floats
        let b_base = i * 32;

        let a0 = f32x16::from_array(*(a_ptr.add(a_base) as *const [f32; 16]));
        let a1 = f32x16::from_array(*(a_ptr.add(a_base + 16) as *const [f32; 16]));
        let a2 = f32x16::from_array(*(a_ptr.add(a_base + 32) as *const [f32; 16]));
        let a3 = f32x16::from_array(*(a_ptr.add(a_base + 48) as *const [f32; 16]));

        let bp = b_ptr.add(b_base);
        let b0 = f32x16::from_array([
            *bp, *bp, *bp.add(1), *bp.add(1),
            *bp.add(2), *bp.add(2), *bp.add(3), *bp.add(3),
            *bp.add(4), *bp.add(4), *bp.add(5), *bp.add(5),
            *bp.add(6), *bp.add(6), *bp.add(7), *bp.add(7),
        ]);
        let b1 = f32x16::from_array([
            *bp.add(8), *bp.add(8), *bp.add(9), *bp.add(9),
            *bp.add(10), *bp.add(10), *bp.add(11), *bp.add(11),
            *bp.add(12), *bp.add(12), *bp.add(13), *bp.add(13),
            *bp.add(14), *bp.add(14), *bp.add(15), *bp.add(15),
        ]);
        let b2 = f32x16::from_array([
            *bp.add(16), *bp.add(16), *bp.add(17), *bp.add(17),
            *bp.add(18), *bp.add(18), *bp.add(19), *bp.add(19),
            *bp.add(20), *bp.add(20), *bp.add(21), *bp.add(21),
            *bp.add(22), *bp.add(22), *bp.add(23), *bp.add(23),
        ]);
        let b3 = f32x16::from_array([
            *bp.add(24), *bp.add(24), *bp.add(25), *bp.add(25),
            *bp.add(26), *bp.add(26), *bp.add(27), *bp.add(27),
            *bp.add(28), *bp.add(28), *bp.add(29), *bp.add(29),
            *bp.add(30), *bp.add(30), *bp.add(31), *bp.add(31),
        ]);

        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }

    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;
    let mut result = reduce_sum_complex_avx512_f32x16(sum0);

    let a_rem = &a[chunks * 32..];
    let b_rem = &b[chunks * 32..];
    for (a, b) in a_rem.iter().zip(b_rem) {
        result += a * b;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;
    #[cfg(feature = "simd")]
    use crate::random::{crandnf, randnf};

    type Cf32 = Complex<f32>;

    #[cfg(feature = "simd")]
    type Cf64 = Complex<f64>;

    #[test]
    #[autotest_annotate(autotest_dotprod_crcf_rand01)]
    fn test_dotprod_crcf_rand01() {
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: [f32; 16] = [
             5.5375e-02,  -6.5857e-01,  -1.7657e+00,   7.7444e-01,
             8.0730e-01,  -5.1340e-01,  -9.3437e-02,  -5.6301e-01,
            -6.6480e-01,  -2.1673e+00,   9.0269e-01,   3.5284e+00,
            -9.7835e-01,  -6.9512e-01,  -1.2958e+00,   1.1628e+00,
        ];

        #[rustfmt::skip]
        let x: [Cf32; 16] = [
            Cf32::new( 1.3164e+00,  5.4161e-01),  Cf32::new( 1.8295e-01, -9.0284e-02),
            Cf32::new( 1.3487e+00, -1.8148e+00),  Cf32::new(-7.4696e-01, -4.1792e-01),
            Cf32::new(-9.0551e-01, -4.4294e-01),  Cf32::new( 6.0591e-01, -1.5383e+00),
            Cf32::new(-7.5393e-01, -3.5691e-01),  Cf32::new(-4.5733e-01,  1.1926e-01),
            Cf32::new(-1.4744e-01, -4.7676e-02),  Cf32::new(-1.2422e+00, -2.0213e+00),
            Cf32::new( 3.3208e-02, -1.3756e+00),  Cf32::new(-4.8573e-01,  1.0977e+00),
            Cf32::new( 1.5053e+00,  2.1141e-01),  Cf32::new(-8.4062e-01, -1.0211e+00),
            Cf32::new(-1.3932e+00, -4.8491e-01),  Cf32::new(-1.4234e+00,  2.0333e-01),
        ];

        let test = Cf32::new(-3.35346556487224, 11.78023318618137);
        let y = h.dotprod(&x);
        assert_abs_diff_eq!(y.re, test.re, epsilon = TOL);
        assert_abs_diff_eq!(y.im, test.im, epsilon = TOL);

        let test_rev = Cf32::new(3.655541203500000, 4.26531912591000);
        let y_rev = h.iter().rev().copied().collect::<Vec<f32>>().dotprod(&x);
        assert_abs_diff_eq!(y_rev.re, test_rev.re, epsilon = TOL);
        assert_abs_diff_eq!(y_rev.im, test_rev.im, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_crcf_rand02)]
    fn test_dotprod_crcf_rand02() {
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: [f32; 16] = [
             4.7622e-01,   7.1453e-01,  -7.1370e-01,  -1.6457e-01,
            -1.1573e-01,   6.4114e-01,  -1.0688e+00,  -1.6761e+00,
            -1.0376e+00,  -1.0991e+00,  -2.4161e-01,   4.6065e-01,
            -1.0403e+00,  -1.1424e-01,  -1.2371e+00,  -7.9723e-01,
        ];

        #[rustfmt::skip]
        let x: [Cf32; 16] = [
            Cf32::new(-8.3558e-01,  3.0504e-01),  Cf32::new(-6.3004e-01,  2.4680e-01),
            Cf32::new( 9.6908e-01,  1.2978e+00),  Cf32::new(-2.0587e+00,  9.5385e-01),
            Cf32::new( 2.5692e-01, -1.7314e+00),  Cf32::new(-1.2237e+00, -6.2139e-02),
            Cf32::new( 5.0300e-02, -9.2092e-01),  Cf32::new(-1.8816e-01,  7.0746e-02),
            Cf32::new(-2.4177e+00,  8.3177e-01),  Cf32::new( 1.6871e-01, -8.5129e-02),
            Cf32::new( 6.5203e-01,  2.0739e-02),  Cf32::new(-1.2331e-01, -9.7920e-01),
            Cf32::new( 8.2352e-01,  9.1093e-01),  Cf32::new( 1.5161e+00, -9.1865e-01),
            Cf32::new(-2.0892e+00,  2.7759e-02),  Cf32::new(-2.5188e-01,  2.5568e-01),
        ];

        let test = Cf32::new(2.11053363855085, -2.04167493441477);
        let y = h.dotprod(&x);
        assert_abs_diff_eq!(y.re, test.re, epsilon = TOL);
        assert_abs_diff_eq!(y.im, test.im, epsilon = TOL);
    }

    #[test]
    fn test_dotprod_crc_rand() {
        const TOL: f32 = 1e-3;

        #[rustfmt::skip]
        let h: [f32; 16] = [
             5.5375e-02,  -6.5857e-01,  -1.7657e+00,   7.7444e-01,
             8.0730e-01,  -5.1340e-01,  -9.3437e-02,  -5.6301e-01,
            -6.6480e-01,  -2.1673e+00,   9.0269e-01,   3.5284e+00,
            -9.7835e-01,  -6.9512e-01,  -1.2958e+00,   1.1628e+00,
        ];

        #[rustfmt::skip]
        let x: [Cf32; 16] = [
            Cf32::new( 1.3164e+00,  5.4161e-01),  Cf32::new( 1.8295e-01, -9.0284e-02),
            Cf32::new( 1.3487e+00, -1.8148e+00),  Cf32::new(-7.4696e-01, -4.1792e-01),
            Cf32::new(-9.0551e-01, -4.4294e-01),  Cf32::new( 6.0591e-01, -1.5383e+00),
            Cf32::new(-7.5393e-01, -3.5691e-01),  Cf32::new(-4.5733e-01,  1.1926e-01),
            Cf32::new(-1.4744e-01, -4.7676e-02),  Cf32::new(-1.2422e+00, -2.0213e+00),
            Cf32::new( 3.3208e-02, -1.3756e+00),  Cf32::new(-4.8573e-01,  1.0977e+00),
            Cf32::new( 1.5053e+00,  2.1141e-01),  Cf32::new(-8.4062e-01, -1.0211e+00),
            Cf32::new(-1.3932e+00, -4.8491e-01),  Cf32::new(-1.4234e+00,  2.0333e-01),
        ];

        let test = Cf32::new(-3.35346556487224, 11.78023318618137);
        let y = x.dotprod(&h);
        assert_abs_diff_eq!(y.re, test.re, epsilon = TOL);
        assert_abs_diff_eq!(y.im, test.im, epsilon = TOL);
    }

    #[test]
    #[should_panic(expected = "Slices must have equal length")]
    fn test_dotprod_crc_mismatched_lengths() {
        [Cf32::new(1.0, 1.0); 32].dotprod(&[1.0f32; 31]);
    }

    #[test]
    fn test_dotprod_crc_boundary_lengths() {
        const TOL: f32 = 1e-3;

        // Complex: each takes 2 floats, so thresholds are half of rrr
        // SSE: <8 scalar, >=8 uses f32x4 (processes 8 complex per iteration)
        // AVX2: <8 scalar, <16 f32x4, >=16 f32x8 (processes 16 complex per iteration)
        // AVX-512: <8 scalar, <32 f32x4, >=32 f32x16 (processes 32 complex per iteration)
        let test_sizes = [
            1, 2, 3, 4,           // tiny scalar
            7, 8, 9,              // scalar/f32x4 boundary
            15, 16, 17,           // f32x4/f32x8 boundary (AVX2)
            31, 32, 33,           // f32x4/f32x16 boundary (AVX-512)
            63, 64, 65,           // 2x f32x16 + cleanup
            127, 128, 129,        // 4x f32x16 + cleanup
            255, 256, 257,        // 8x f32x16 + cleanup
        ];

        for &n in &test_sizes {
            let x: Vec<Cf32> = (0..n)
                .map(|i| Cf32::new((i as f32 * 0.1).sin(), (i as f32 * 0.15).cos()))
                .collect();
            let h: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

            let expected: Cf32 = x.iter().zip(h.iter()).map(|(&a, &b)| a * b).sum();
            let result = x.dotprod(&h);

            assert_abs_diff_eq!(result.re, expected.re, epsilon = TOL);
            assert_abs_diff_eq!(result.im, expected.im, epsilon = TOL);
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_crc_avx512_direct() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        for n in 1..=512 {
            let h: Vec<Cf32> = (0..n).map(|_| crandnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: Cf64 = h.iter().zip(x.iter()).map(|(&a, &b)| Cf64::new(a.re as f64, a.im as f64) * b as f64).sum();
            let y_avx512 = unsafe { dotprod_crc_avx512(&h, &x) };

            assert_abs_diff_eq!(y_avx512.re, y_test.re as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
            assert_abs_diff_eq!(y_avx512.im, y_test.im as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_crc_avx2_direct() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        for n in 1..=512 {
            let h: Vec<Cf32> = (0..n).map(|_| crandnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: Cf64 = h.iter().zip(x.iter()).map(|(&a, &b)| Cf64::new(a.re as f64, a.im as f64) * b as f64).sum();
            let y_avx2 = unsafe { dotprod_crc_avx2(&h, &x) };

            assert_abs_diff_eq!(y_avx2.re, y_test.re as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
            assert_abs_diff_eq!(y_avx2.im, y_test.im as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_crc_sse_direct() {
        if !is_x86_feature_detected!("sse") {
            return;
        }

        for n in 1..=512 {
            let h: Vec<Cf32> = (0..n).map(|_| crandnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: Cf64 = h.iter().zip(x.iter()).map(|(&a, &b)| Cf64::new(a.re as f64, a.im as f64) * b as f64).sum();
            let y_sse = unsafe { dotprod_crc_sse(&h, &x) };

            assert_abs_diff_eq!(y_sse.re, y_test.re as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
            assert_abs_diff_eq!(y_sse.im, y_test.im as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_dotprod_crc_scalar_direct() {
        for n in 1..=512 {
            let h: Vec<Cf32> = (0..n).map(|_| crandnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: Cf64 = h.iter().zip(x.iter()).map(|(&a, &b)| Cf64::new(a.re as f64, a.im as f64) * b as f64).sum();
            let y_scalar = unsafe { dotprod_crc_scalar(&h, &x) };

            assert_abs_diff_eq!(y_scalar.re, y_test.re as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
            assert_abs_diff_eq!(y_scalar.im, y_test.im as f32, epsilon = 2.0 * n as f32 * f32::EPSILON);
        }
    }
}
