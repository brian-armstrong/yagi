// Complex-Real-Complex dot product: [Complex<f32>] · [f32] -> Complex<f32>
// Also handles rcc: [f32] · [Complex<f32>] -> Complex<f32> via delegation

use num_complex::Complex;

use super::DotProd;

#[cfg(feature = "simd")]
use std::simd::f32x4;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use std::simd::{f32x8, f32x16};
#[cfg(feature = "simd")]
use std::sync::OnceLock;

#[cfg(feature = "simd")]
use super::crc_block::plan_dotprod_crc_block_f32x4;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use super::crc_block::plan_dotprod_crc_block_avx512;
#[cfg(feature = "simd")]
use super::reduce::reduce_sum_complex_sse_f32x4;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use super::reduce::{reduce_sum_complex_avx2_f32x8, reduce_sum_complex_avx512_f32x16};

#[cfg(feature = "simd")]
type DotProdCrcFn = unsafe fn(&[Complex<f32>], &[f32]) -> Complex<f32>;
#[cfg(feature = "simd")]
static DOTPROD_CRC: OnceLock<DotProdCrcFn> = OnceLock::new();

#[cfg(feature = "simd")]
macro_rules! plan_dotprod_crc_const_f32x4 {
    ($len:expr; $($n:literal),+ $(,)?) => {
        match $len {
            $(
                $n => Some(dotprod_crc_const_f32x4::<$n> as DotProdCrcFn),
            )+
            _ => None,
        }
    };
}

#[cfg(feature = "simd")]
const DOTPROD_CRC_WIDE_CUTOFF: usize = 10;

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
        dotprod_crc_scalar(self, other)
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[f32]) -> Complex<f32> {
        assert_eq!(self.len(), other.len(), "Slices must have equal length");
        if self.len() < DOTPROD_CRC_WIDE_CUTOFF {
            return unsafe { dotprod_crc_128(self, other) };
        }
        let f = DOTPROD_CRC.get_or_init(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    return dotprod_crc_avx512;
                }
                if is_x86_feature_detected!("avx2") {
                    return dotprod_crc_avx2;
                }
            }
            dotprod_crc_128
        });
        unsafe { f(self, other) }
    }

    #[cfg(feature = "simd")]
    fn plan(len: usize) -> super::DotProdKernel<Complex<f32>, f32, Complex<f32>> {
        if let Some(f) = plan_dotprod_crc_const_f32x4!(
            len;
              1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
             11, 12, 13, 14, 15,
        ) {
            return f;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                return dotprod_crc_avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return dotprod_crc_avx2;
            }
        }
        dotprod_crc_128
    }

    #[cfg(feature = "simd")]
    fn plan_block(h: &[f32]) -> Option<super::DotProdBlockPlan<[Complex<f32>], f32, Complex<f32>>> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if h.len() >= 13 && is_x86_feature_detected!("avx512f") {
            return plan_dotprod_crc_block_avx512(h);
        }

        if h.len() <= 32 {
            plan_dotprod_crc_block_f32x4(h)
        } else {
            None
        }
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

#[cfg(not(feature = "simd"))]
fn dotprod_crc_scalar_wide(a: &[Complex<f32>], b: &[f32]) -> (Complex<f32>, usize) {
    let chunks = a.len() / 2;
    if chunks == 0 {
        return (Complex::new(0.0, 0.0), 0);
    }

    let mut re0 = 0.0f32;
    let mut im0 = 0.0f32;
    let mut re1 = 0.0f32;
    let mut im1 = 0.0f32;

    for i in 0..chunks {
        let k = i * 2;
        re0 += a[k].re * b[k];
        im0 += a[k].im * b[k];
        re1 += a[k + 1].re * b[k + 1];
        im1 += a[k + 1].im * b[k + 1];
    }

    (Complex::new(re0 + re1, im0 + im1), chunks * 2)
}

#[cfg(not(feature = "simd"))]
fn dotprod_crc_scalar(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    let (sum, n) = dotprod_crc_scalar_wide(a, b);
    sum + a[n..]
        .iter()
        .zip(&b[n..])
        .map(|(a, b)| a * b)
        .sum::<Complex<f32>>()
}

// Scalar fallback
#[cfg(feature = "simd")]
unsafe fn dotprod_crc_scalar(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(feature = "simd")]
unsafe fn dotprod_crc_const_f32x4<const N: usize>(
    // this method is const on N so that the loops below disappear
    // SIMD uses this for various small N where this is small and fast 
    a: &[Complex<f32>],
    b: &[f32],
) -> Complex<f32> {
    debug_assert_eq!(a.len(), N);
    debug_assert_eq!(b.len(), N);
    unsafe {
        std::hint::assert_unchecked(a.len() == N);
        std::hint::assert_unchecked(b.len() == N);
    }

    let mut sum = f32x4::splat(0.0);
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr();
    let mut i = 0;
    while i + 2 <= N {
        let av = f32x4::from_array(*(a_ptr.add(i * 2) as *const [f32; 4]));
        let b0 = *b_ptr.add(i);
        let b1 = *b_ptr.add(i + 1);
        sum += av * f32x4::from_array([b0, b0, b1, b1]);
        i += 2;
    }

    let lanes = sum.to_array();
    let mut result = Complex::new(lanes[0] + lanes[2], lanes[1] + lanes[3]);
    while i < N {
        result += a[i] * b[i];
        i += 1;
    }
    result
}

#[cfg(feature = "simd")]
pub(super) unsafe fn dotprod_crc_128(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    let (s0, n0) = dotprod_crc_sse_f32x4_wide(a, b);
    let (a, b) = (&a[n0..], &b[n0..]);
    let (s1, n1) = dotprod_crc_sse_f32x4_narrow(a, b);
    s0 + s1 + dotprod_crc_scalar(&a[n1..], &b[n1..])
}

/// 4x-unrolled f32x4: 8 elements per iteration.
#[cfg(feature = "simd")]
#[inline]
unsafe fn dotprod_crc_sse_f32x4_wide(
    a: &[Complex<f32>],
    b: &[f32],
) -> (Complex<f32>, usize) {
    // 8 complex = 16 floats per iteration (4 accumulators x 2 complex each)
    let chunks = a.len() / 8;
    if chunks == 0 {
        return (Complex::new(0.0, 0.0), 0);
    }

    let mut sum0 = f32x4::splat(0.0);
    let mut sum1 = f32x4::splat(0.0);
    let mut sum2 = f32x4::splat(0.0);
    let mut sum3 = f32x4::splat(0.0);

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
    (reduce_sum_complex_sse_f32x4(sum0), chunks * 8)
}

/// Single f32x4: 2 complex elements per iteration
#[cfg(feature = "simd")]
#[inline]
unsafe fn dotprod_crc_sse_f32x4_narrow(
    a: &[Complex<f32>],
    b: &[f32],
) -> (Complex<f32>, usize) {
    let chunks = a.len() / 2;
    if chunks == 0 {
        return (Complex::new(0.0, 0.0), 0);
    }

    let mut sum = f32x4::splat(0.0);
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let av = f32x4::from_array(*(a_ptr.add(i * 4) as *const [f32; 4]));
        let bp = b_ptr.add(i * 2);
        let bv = f32x4::from_array([*bp, *bp, *bp.add(1), *bp.add(1)]);
        sum += av * bv;
    }

    (reduce_sum_complex_sse_f32x4(sum), chunks * 2)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dotprod_crc_avx2(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    let (s0, n0) = dotprod_crc_avx2_f32x8_wide(a, b);
    let (a, b) = (&a[n0..], &b[n0..]);
    let (s1, n1) = dotprod_crc_sse_f32x4_narrow(a, b);
    s0 + s1 + dotprod_crc_scalar(&a[n1..], &b[n1..])
}

/// 4x-unrolled f32x8: 16 elements per iteration.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dotprod_crc_avx2_f32x8_wide(
    a: &[Complex<f32>],
    b: &[f32],
) -> (Complex<f32>, usize) {
    let chunks = a.len() / 16;
    if chunks == 0 {
        return (Complex::new(0.0, 0.0), 0);
    }

    let mut sum0 = f32x8::splat(0.0);
    let mut sum1 = f32x8::splat(0.0);
    let mut sum2 = f32x8::splat(0.0);
    let mut sum3 = f32x8::splat(0.0);

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
    (reduce_sum_complex_avx2_f32x8(sum0), chunks * 16)
}

/// Single f32x8: 4 complex elements per iteration.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dotprod_crc_avx2_f32x8_narrow(
    a: &[Complex<f32>],
    b: &[f32],
) -> (Complex<f32>, usize) {
    let chunks = a.len() / 4;
    if chunks == 0 {
        return (Complex::new(0.0, 0.0), 0);
    }

    let mut sum = f32x8::splat(0.0);
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let av = f32x8::from_array(*(a_ptr.add(i * 8) as *const [f32; 8]));
        let bp = b_ptr.add(i * 4);
        let bv = f32x8::from_array([
            *bp, *bp, *bp.add(1), *bp.add(1),
            *bp.add(2), *bp.add(2), *bp.add(3), *bp.add(3),
        ]);
        sum += av * bv;
    }

    (reduce_sum_complex_avx2_f32x8(sum), chunks * 4)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn dotprod_crc_avx512(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    let (s0, n0) = dotprod_crc_avx512_f32x16_wide(a, b);
    let (a, b) = (&a[n0..], &b[n0..]);
    let (s1, n1) = dotprod_crc_avx2_f32x8_narrow(a, b);
    s0 + s1 + dotprod_crc_scalar(&a[n1..], &b[n1..])
}


/// 4x-unrolled f32x8: 16 elements per iteration.
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn dotprod_crc_avx512_f32x16_wide(
    a: &[Complex<f32>],
    b: &[f32],
) -> (Complex<f32>, usize) {
    let chunks = a.len() / 32;
    if chunks == 0 {
        return (Complex::new(0.0, 0.0), 0);
    }

    let mut sum0 = f32x16::splat(0.0);
    let mut sum1 = f32x16::splat(0.0);
    let mut sum2 = f32x16::splat(0.0);
    let mut sum3 = f32x16::splat(0.0);

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
    (reduce_sum_complex_avx512_f32x16(sum0), chunks * 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[cfg(feature = "simd")]
    use crate::random::{crandnf, randnf};

    type Cf32 = Complex<f32>;

    #[cfg(feature = "simd")]
    type Cf64 = Complex<f64>;

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

    #[cfg(feature = "simd")]
    #[test]
    fn test_dotprod_crc_128_direct() {
        for n in 1..=512 {
            let h: Vec<Cf32> = (0..n).map(|_| crandnf()).collect();
            let x: Vec<f32> = (0..n).map(|_| randnf()).collect();

            let y_test: Cf64 = h.iter().zip(x.iter()).map(|(&a, &b)| Cf64::new(a.re as f64, a.im as f64) * b as f64).sum();
            let y_sse = unsafe { dotprod_crc_128(&h, &x) };

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
