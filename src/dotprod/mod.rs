// Dotprod module
// Current state:
// - Dotprod ready to use (+autotests)
// - sumsq missing

use num_complex::Complex;

#[cfg(feature = "simd")]
use std::simd::num::SimdFloat;
#[cfg(feature = "simd")]
use std::simd::{f32x8, simd_swizzle};

pub trait DotProd<Rhs> {
    type Output;

    fn dotprod(&self, other: &[Rhs]) -> Self::Output;
}

impl DotProd<f32> for [f32] {
    type Output = f32;

    #[cfg(not(feature = "simd"))]
    fn dotprod(&self, other: &[f32]) -> f32 {
        self.iter().zip(other).map(|(a, b)| a * b).sum()
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[f32]) -> f32 {
        dotprod_rrr_simd8(self, other)
    }
}

impl DotProd<Complex<f32>> for [f32] {
    type Output = Complex<f32>;

    #[cfg(not(feature = "simd"))]
    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        self.iter().zip(other).map(|(a, b)| a * b).sum()
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        dotprod_rcc_simd8(self, other)
    }
}

impl DotProd<f32> for [Complex<f32>] {
    type Output = Complex<f32>;

    #[cfg(not(feature = "simd"))]
    fn dotprod(&self, other: &[f32]) -> Complex<f32> {
        self.iter().zip(other).map(|(a, b)| a * b).sum()
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[f32]) -> Complex<f32> {
        dotprod_crc_simd8(self, other)
    }
}

impl DotProd<Complex<f32>> for [Complex<f32>] {
    type Output = Complex<f32>;

    #[cfg(not(feature = "simd"))]
    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        self.iter().zip(other).map(|(a, b)| a * b).sum()
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        dotprod_ccc_simd8(self, other)
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

impl DotProd<Complex<f32>> for std::collections::VecDeque<Complex<f32>> {
    type Output = Complex<f32>;

    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        let (l, r) = self.as_slices();
        let split_idx = l.len();
        let l_sum = l.dotprod(&other[..split_idx]);
        let r_sum = r.dotprod(&other[split_idx..]);
        l_sum + r_sum
    }
}

#[cfg(feature = "simd")]
fn dotprod_rrr_simd8(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");

    let mut sum = f32x8::splat(0.0);
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let start = i * 8;
        let a_chunk = f32x8::from_slice(&a[start..start + 8]);
        let b_chunk = f32x8::from_slice(&b[start..start + 8]);
        sum += a_chunk * b_chunk;
    }

    let mut result = sum.reduce_sum();

    // Handle remaining elements
    for i in (chunks * 8)..a.len() {
        result += a[i] * b[i];
    }

    result
}

#[cfg(feature = "simd")]
fn dotprod_crc_simd8(a: &[Complex<f32>], b: &[f32]) -> Complex<f32> {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");

    const CHUNK_SIZE: usize = 8 / 2;
    let mut sum_re = f32x8::splat(0.0);
    let mut sum_im = f32x8::splat(0.0);
    let chunks = a.len() / CHUNK_SIZE; // We process 4 complex numbers at a time

    let a_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len() * 2)
    };

    for i in 0..chunks {
        let a_start = i * CHUNK_SIZE * 2;
        let b_start = i * CHUNK_SIZE;
        let a_chunk = f32x8::from_slice(&a_floats[a_start..a_start + 8]);
        let b_chunk = f32x8::from_slice(&[b[b_start], 0.0, b[b_start + 1], 0.0, b[b_start + 2], 0.0, b[b_start + 3], 0.0]);

        let prod_a = a_chunk * b_chunk;
        let prod_b = simd_swizzle!(a_chunk, [1, 0, 3, 2, 5, 4, 7, 6]) * b_chunk;
        let prod_c = prod_a * f32x8::from_slice(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

        sum_re += prod_c;
        sum_im += prod_b;
    }

    let mut result = Complex::new(sum_re.reduce_sum(), sum_im.reduce_sum());

    // Handle remaining elements
    for i in (chunks * CHUNK_SIZE)..a.len() {
        result += a[i] * b[i];
    }

    result
}

#[cfg(feature = "simd")]
fn dotprod_rcc_simd8(a: &[f32], b: &[Complex<f32>]) -> Complex<f32> {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");

    const CHUNK_SIZE: usize = 8 / 2;
    let mut sum_re = f32x8::splat(0.0);
    let mut sum_im = f32x8::splat(0.0);
    let chunks = a.len() / CHUNK_SIZE; // We process 4 complex numbers at a time

    for i in 0..chunks {
        let start = i * CHUNK_SIZE;
        let a_chunk = f32x8::from_slice(&[a[start], 0.0, a[start + 1], 0.0, a[start + 2], 0.0, a[start + 3], 0.0]);
        let b_chunk = f32x8::from_slice(&[
            b[start].re,
            b[start].im,
            b[start + 1].re,
            b[start + 1].im,
            b[start + 2].re,
            b[start + 2].im,
            b[start + 3].re,
            b[start + 3].im,
        ]);

        let prod_a = a_chunk * b_chunk;
        let prod_b = simd_swizzle!(a_chunk, [1, 0, 3, 2, 5, 4, 7, 6]) * b_chunk;
        let prod_c = prod_a * f32x8::from_slice(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

        sum_re += prod_c;
        sum_im += prod_b;
    }

    let mut result = Complex::new(sum_re.reduce_sum(), sum_im.reduce_sum());

    // Handle remaining elements
    for i in (chunks * CHUNK_SIZE)..a.len() {
        result += a[i] * b[i];
    }

    result
}

#[cfg(feature = "simd")]
fn dotprod_ccc_simd8(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    assert_eq!(a.len(), b.len(), "Slices must have equal length");

    const CHUNK_SIZE: usize = 8 / 2;
    let mut sum_re = f32x8::splat(0.0);
    let mut sum_im = f32x8::splat(0.0);
    let chunks = a.len() / CHUNK_SIZE; // We process 4 complex numbers at a time

    for i in 0..chunks {
        let start = i * CHUNK_SIZE;
        let a_chunk = f32x8::from_slice(&[
            a[start].re,
            a[start].im,
            a[start + 1].re,
            a[start + 1].im,
            a[start + 2].re,
            a[start + 2].im,
            a[start + 3].re,
            a[start + 3].im,
        ]);
        let b_chunk = f32x8::from_slice(&[
            b[start].re,
            b[start].im,
            b[start + 1].re,
            b[start + 1].im,
            b[start + 2].re,
            b[start + 2].im,
            b[start + 3].re,
            b[start + 3].im,
        ]);

        let prod_a = a_chunk * b_chunk;
        let prod_b = simd_swizzle!(a_chunk, [1, 0, 3, 2, 5, 4, 7, 6]) * b_chunk;
        let prod_c = prod_a * f32x8::from_slice(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

        sum_re += prod_c;
        sum_im += prod_b;
    }

    let mut result = Complex::new(sum_re.reduce_sum(), sum_im.reduce_sum());

    // Handle remaining elements
    for i in (chunks * CHUNK_SIZE)..a.len() {
        result += a[i] * b[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::Rng;
    use test_macro::autotest_annotate;

    type Cf32 = Complex<f32>;

    // TODO we may want to consider removing the "struct" series of tests
    //   since we don't have that

    #[test]
    fn test_dotprod_rrr() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(a.dotprod(&b), 32.0);
    }

    #[test]
    fn test_dotprod_ccc() {
        let a = vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)];
        let b = vec![Complex::new(4.0, -4.0), Complex::new(5.0, -5.0), Complex::new(6.0, -6.0)];
        assert_eq!(a.dotprod(&b), Complex::new(64.0, 0.0));
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_basic)]
    fn test_dotprod_rrrf_basic() {
        const TOL: f32 = 1e-6;
        let h: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

        let x0 = vec![0.0; 16];
        assert_relative_eq!(h.dotprod(&x0), 0.0, epsilon = TOL);

        let x1 = vec![1.0; 16];
        assert_relative_eq!(h.dotprod(&x1), 0.0, epsilon = TOL);

        let x2: Vec<f32> = (0..16).map(|i| (i % 2) as f32).collect();
        assert_relative_eq!(h.dotprod(&x2), -8.0, epsilon = TOL);

        let x3: Vec<f32> = (0..16).map(|i| 1.0 - (i % 2) as f32).collect();
        assert_relative_eq!(h.dotprod(&x3), 8.0, epsilon = TOL);

        assert_relative_eq!(h.dotprod(&h), 16.0, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_uneven)]
    fn test_dotprod_rrrf_uneven() {
        const TOL: f32 = 1e-6;
        let h: Vec<f32> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let x = vec![1.0; 16];

        assert_relative_eq!(h[..1].dotprod(&x[..1]), 1.0, epsilon = TOL);
        assert_relative_eq!(h[..2].dotprod(&x[..2]), 0.0, epsilon = TOL);
        assert_relative_eq!(h[..3].dotprod(&x[..3]), 1.0, epsilon = TOL);
        assert_relative_eq!(h[..11].dotprod(&x[..11]), 1.0, epsilon = TOL);
        assert_relative_eq!(h[..13].dotprod(&x[..13]), 1.0, epsilon = TOL);
        assert_relative_eq!(h[..15].dotprod(&x[..15]), 1.0, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_rand01)]
    fn test_dotprod_rrrf_rand01() {
        const TOL: f32 = 1e-3;

        // Format the following array with 4 columns, fixed width, 12 spaces per column
        #[rustfmt::skip]
        let h: Vec<f32> = vec![
            -0.050565,   -0.952580,    0.274320,    1.232400,
             1.268200,    0.565770,    0.800830,    0.923970,
             0.517060,   -0.530340,   -0.378550,   -1.127100,
             1.123100,   -1.006000,   -1.483800,   -0.062007,
        ];

        // Format the following array with 4 columns, fixed width, 12 spaces per column
        #[rustfmt::skip]
        let x: Vec<f32> = vec![
            -0.384280,   -0.812030,    0.156930,    1.919500,
             0.564580,   -0.123610,   -0.138640,    0.004984,
            -1.100200,   -0.497620,    0.089977,   -1.745500,
             0.463640,    0.592100,    1.150000,   -1.225400,
        ];

        let test = 3.66411513609863;
        assert_relative_eq!(h.dotprod(&x), test, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_rand02)]
    fn test_dotprod_rrrf_rand02() {
        const TOL: f32 = 1e-3;

        // Format the following array with 4 columns, fixed width, 12 spaces per column
        #[rustfmt::skip]
        let h: Vec<f32> = vec![
             2.595300,    1.243600,   -0.818550,   -1.439800,
             0.055795,   -1.476000,    0.445900,    0.325460,
            -3.451200,    0.058528,   -0.246990,    0.476290,
            -0.598780,   -0.885250,    0.464660,   -0.610140,
        ];

        // Format the following array with 4 columns, fixed width, 12 spaces per column
        #[rustfmt::skip]
        let x: Vec<f32> = vec![
            -0.917010,   -1.278200,   -0.533190,    2.309200,
             0.592980,    0.964820,    0.183220,   -0.082864,
             0.057171,   -1.186500,   -0.738260,    0.356960,
            -0.144000,   -1.435200,   -0.893420,    1.657800,
        ];

        let test = -8.17832326680587;
        assert_relative_eq!(h.dotprod(&x), test, epsilon = TOL);

        let test_rev = 4.56839328512000;
        assert_relative_eq!(h.iter().rev().cloned().collect::<Vec<f32>>().dotprod(&x), test_rev, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_struct_lengths)]
    fn test_dotprod_rrrf_struct_lengths() {
        const TOL: f32 = 2e-6;

        // Format the following array with 4 columns, fixed width, 12 spaces per column
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

        // Format the following array with 4 columns, fixed width, 12 spaces per column
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

        assert_relative_eq!(h[..32].dotprod(&x[..32]), -7.99577847, epsilon = TOL);
        assert_relative_eq!(h[..33].dotprod(&x[..33]), -6.00389114, epsilon = TOL);
        assert_relative_eq!(h[..34].dotprod(&x[..34]), -6.36813751, epsilon = TOL);
        assert_relative_eq!(h[..35].dotprod(&x[..35]), -6.44988725, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_rrrf_struct_vs_ordinal)]
    fn test_dotprod_rrrf_struct_vs_ordinal() {
        // note that since we don't have structured dotproduct, this is just a big random test
        const TOL: f32 = 1e-4;
        let mut rng = rand::thread_rng();

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
            let x: Vec<f32> = (0..n).map(|_| rng.gen()).collect();

            let y_test: f32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let y_struct = h.dotprod(&x);

            assert_relative_eq!(y_struct, y_test, epsilon = TOL);
        }
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_crcf_rand01)]
    fn test_dotprod_crcf_rand01() {
        const TOL: f32 = 1e-3;

        // Format the following array with 4 columns, fixed width, 12 spaces per argument
        #[rustfmt::skip]
        let h: [f32; 16] = [
             5.5375e-02,  -6.5857e-01,  -1.7657e+00,   7.7444e-01,
             8.0730e-01,  -5.1340e-01,  -9.3437e-02,  -5.6301e-01,
            -6.6480e-01,  -2.1673e+00,   9.0269e-01,   3.5284e+00,
            -9.7835e-01,  -6.9512e-01,  -1.2958e+00,   1.1628e+00,
        ];

        // Format the following array with 2 columns, fixed width, 12 spaces per argument
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
        assert_relative_eq!(y.re, test.re, epsilon = TOL);
        assert_relative_eq!(y.im, test.im, epsilon = TOL);

        let test_rev = Cf32::new(3.655541203500000, 4.26531912591000);
        let y_rev = h.iter().rev().copied().collect::<Vec<f32>>().dotprod(&x);
        assert_relative_eq!(y_rev.re, test_rev.re, epsilon = TOL);
        assert_relative_eq!(y_rev.im, test_rev.im, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_crcf_rand02)]
    fn test_dotprod_crcf_rand02() {
        const TOL: f32 = 1e-3;

        // Format the following array with 4 columns, fixed width, 12 spaces per argument
        #[rustfmt::skip]
        let h: [f32; 16] = [
             4.7622e-01,   7.1453e-01,  -7.1370e-01,  -1.6457e-01,
            -1.1573e-01,   6.4114e-01,  -1.0688e+00,  -1.6761e+00,
            -1.0376e+00,  -1.0991e+00,  -2.4161e-01,   4.6065e-01,
            -1.0403e+00,  -1.1424e-01,  -1.2371e+00,  -7.9723e-01,
        ];

        // Format the following array with 2 columns, fixed width, 12 spaces per argument
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
        assert_relative_eq!(y.re, test.re, epsilon = TOL);
        assert_relative_eq!(y.im, test.im, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_crcf_struct_vs_ordinal)]
    fn test_dotprod_crcf_struct_vs_ordinal() {
        // note another big random test
        let mut rng = rand::thread_rng();
        const TOL: f32 = 1e-4;

        for n in 1..=512 {
            let h: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
            let x: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();

            // Compute expected value (ordinal computation)
            let y_test: Cf32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();

            // Compute using dotprod
            let y_struct = h.dotprod(&x);

            assert_relative_eq!(y_struct.re, y_test.re, epsilon = TOL);
            assert_relative_eq!(y_struct.im, y_test.im, epsilon = TOL);
        }
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_cccf_rand16)]
    fn test_dotprod_cccf_rand16() {
        const TOL: f32 = 1e-3;

        // Format the following array with 2 columns, fixed width, 12 spaces per argument
        #[rustfmt::skip]
        let h: [Cf32; 16] = [
            Cf32::new( 0.17702709,  1.38978455),  Cf32::new( 0.91294148,  0.39217381),
            Cf32::new(-0.80607338,  0.76477512),  Cf32::new( 0.05099755, -0.87350051),
            Cf32::new( 0.44513826, -0.49490569),  Cf32::new( 0.14754967,  2.04349962),
            Cf32::new( 1.07246623,  1.08146290),  Cf32::new(-1.14028088,  1.83380899),
            Cf32::new( 0.38105361, -0.45591846),  Cf32::new( 0.32605401,  0.34440081),
            Cf32::new(-0.05477144,  0.60832595),  Cf32::new( 1.81667523, -1.12238075),
            Cf32::new(-0.87190497,  1.10743858),  Cf32::new( 1.30921403,  1.24438643),
            Cf32::new( 0.55524695, -1.94931519),  Cf32::new(-0.87191170,  0.91693119),
        ];

        // Format the following array with 2 columns, fixed width, 12 spaces per argument
        #[rustfmt::skip]
        let x: [Cf32; 16] = [
            Cf32::new(-2.19591953, -0.93229692),  Cf32::new( 0.17150376,  0.56165114),
            Cf32::new( 1.58354529, -0.50696037),  Cf32::new( 1.40929619,  0.87868803),
            Cf32::new(-0.75505072, -0.30867372),  Cf32::new(-0.09821367, -0.73949106),
            Cf32::new( 0.03785571,  0.72763665),  Cf32::new(-1.20262636, -0.88838102),
            Cf32::new( 0.23323685,  0.12456235),  Cf32::new( 0.34593736,  0.02529594),
            Cf32::new( 0.33669564,  0.39064649),  Cf32::new(-2.45003867, -0.54862205),
            Cf32::new(-2.64870707,  2.33444473),  Cf32::new(-0.92284477, -2.45121397),
            Cf32::new( 0.24852918, -0.62409860),  Cf32::new(-0.87039907,  0.90921212),
        ];

        let test = Cf32::new(-0.604285042605890, -12.390925785344704);
        let test_rev = Cf32::new(3.412365881765360, 6.1885320363931480);

        let y = h.dotprod(&x);
        assert_relative_eq!(y.re, test.re, epsilon = TOL);
        assert_relative_eq!(y.im, test.im, epsilon = TOL);

        let y_rev = h.iter().rev().copied().collect::<Vec<Cf32>>().dotprod(&x);
        assert_relative_eq!(y_rev.re, test_rev.re, epsilon = TOL);
        assert_relative_eq!(y_rev.im, test_rev.im, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_cccf_struct_lengths)]
    fn test_dotprod_cccf_struct_lengths() {
        const TOL: f32 = 4e-6;

        // Format the following array with 2 columns, fixed width, 12 spaces per argument
        #[rustfmt::skip]
        let h: [Cf32; 35] = [
            Cf32::new( 1.11555653,  2.30658043),  Cf32::new(-0.36133676, -0.10917327),
            Cf32::new( 0.17714505, -2.14631440),  Cf32::new( 2.20424609,  0.59063608),
            Cf32::new(-0.44699194,  0.23369318),  Cf32::new( 0.60613931,  0.21868288),
            Cf32::new(-1.18746289, -0.52159563),  Cf32::new(-0.46277775,  0.75010157),
            Cf32::new( 0.93796307,  0.28608151),  Cf32::new(-2.18699829,  0.38029319),
            Cf32::new( 0.16145611,  0.18343353),  Cf32::new(-0.62653631, -1.79037656),
            Cf32::new(-0.67042462,  0.11044084),  Cf32::new( 0.70333438,  1.78729174),
            Cf32::new(-0.32923580,  0.78514690),  Cf32::new( 0.27534332, -0.56377431),
            Cf32::new( 0.41492559,  1.37176526),  Cf32::new( 3.25368958,  2.70495218),
            Cf32::new( 1.63002035, -0.14193750),  Cf32::new( 2.22057186,  0.55056461),
            Cf32::new( 1.40896777,  0.80722903),  Cf32::new(-0.22334033, -0.14227395),
            Cf32::new(-1.48631186,  0.53610531),  Cf32::new(-1.91632185,  0.88755083),
            Cf32::new(-0.52054895, -0.35572001),  Cf32::new(-1.56515607, -0.41448794),
            Cf32::new(-0.91107117,  0.17059659),  Cf32::new(-0.77007659,  2.73381816),
            Cf32::new(-0.46645585,  0.38994666),  Cf32::new( 0.80317663, -0.41756968),
            Cf32::new( 0.26992512,  0.41828145),  Cf32::new(-0.72456446,  1.25002030),
            Cf32::new( 1.19573306,  0.98449546),  Cf32::new( 1.42491943, -0.55426305),
            Cf32::new( 1.08243614,  0.35774368),
        ];

        // Format the following array with 2 columns, fixed width, 12 spaces per argument
        #[rustfmt::skip]
        let x: [Cf32; 35] = [
            Cf32::new(-0.82466736, -1.39329228),  Cf32::new(-1.46176052, -1.96218827),
            Cf32::new(-1.28388174, -0.07152934),  Cf32::new(-0.51910014, -0.37915971),
            Cf32::new(-0.65964708, -0.98417534),  Cf32::new(-1.40213479, -0.82198463),
            Cf32::new( 0.86051446,  0.97926463),  Cf32::new( 0.26257342,  0.76586696),
            Cf32::new( 0.72174183, -1.89884636),  Cf32::new(-0.26018863,  1.06920599),
            Cf32::new( 0.57949117, -0.77431546),  Cf32::new( 0.84635184, -0.81123009),
            Cf32::new(-1.12637629, -0.42027412),  Cf32::new(-1.04214881,  0.90519721),
            Cf32::new( 0.54458433, -1.03487314),  Cf32::new(-0.17847893,  2.20358978),
            Cf32::new( 0.19642532, -0.07449796),  Cf32::new(-1.84958229,  0.13218920),
            Cf32::new(-1.49042886,  0.81610408),  Cf32::new(-0.27466940, -1.48438409),
            Cf32::new( 0.29239375,  0.72443343),  Cf32::new(-1.20243456, -2.77032750),
            Cf32::new(-0.41784260,  0.77455254),  Cf32::new( 0.37737465, -0.52426993),
            Cf32::new(-1.25500377,  1.76270122),  Cf32::new( 1.55976056, -1.18189171),
            Cf32::new(-0.05111343, -1.18849396),  Cf32::new(-1.92966664,  0.66504899),
            Cf32::new(-2.82387897,  1.41128242),  Cf32::new(-1.48171326, -0.03347470),
            Cf32::new( 0.38047273, -1.40969799),  Cf32::new( 1.71995272,  0.00298203),
            Cf32::new( 0.56040910, -0.12713027),  Cf32::new(-0.46653022, -0.65450499),
            Cf32::new( 0.15515755,  1.58944030),
        ];

        let v32 = Cf32::new(-11.5100903519506, -15.3575526884014);
        let v33 = Cf32::new(-10.7148314918614, -14.9578463360225);
        let v34 = Cf32::new(-11.7423673921916, -15.6318827515320);
        let v35 = Cf32::new(-12.1430314741466, -13.8559085000689);

        assert_relative_eq!(h[..32].dotprod(&x[..32]).re, v32.re, epsilon = TOL);
        assert_relative_eq!(h[..32].dotprod(&x[..32]).im, v32.im, epsilon = TOL);
        assert_relative_eq!(h[..33].dotprod(&x[..33]).re, v33.re, epsilon = TOL);
        assert_relative_eq!(h[..33].dotprod(&x[..33]).im, v33.im, epsilon = TOL);
        assert_relative_eq!(h[..34].dotprod(&x[..34]).re, v34.re, epsilon = TOL);
        assert_relative_eq!(h[..34].dotprod(&x[..34]).im, v34.im, epsilon = TOL);
        assert_relative_eq!(h[..35].dotprod(&x[..35]).re, v35.re, epsilon = TOL);
        assert_relative_eq!(h[..35].dotprod(&x[..35]).im, v35.im, epsilon = TOL);
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_cccf_struct_vs_ordinal)]
    fn test_dotprod_cccf_struct_vs_ordinal() {
        // note another big random test
        const TOL: f32 = 1e-3;
        let mut rng = rand::thread_rng();

        for n in 1..=512 {
            let h: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();
            let x: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();

            // Compute expected value (ordinal computation)
            let y_test: Cf32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();

            // Compute using dotprod
            let y_struct = h.dotprod(&x);

            assert_relative_eq!(y_struct.re, y_test.re, epsilon = TOL);
            assert_relative_eq!(y_struct.im, y_test.im, epsilon = TOL);
        }
    }
}
