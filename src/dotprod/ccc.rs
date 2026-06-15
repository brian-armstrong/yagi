// Complex-Complex-Complex dot product: [Complex<f32>] · [Complex<f32>] -> Complex<f32>

use num_complex::Complex;

use super::DotProd;

#[cfg(feature = "simd")]
use std::simd::{f32x4, f32x8, f32x16, simd_swizzle};
#[cfg(feature = "simd")]
use std::sync::OnceLock;

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use super::reduce::{
    reduce_sum_sse_f32x4, reduce_sum_avx2_f32x8, reduce_sum_avx512_f32x16,
};

#[cfg(feature = "simd")]
type DotProdCccFn = unsafe fn(&[Complex<f32>], &[Complex<f32>]) -> Complex<f32>;
#[cfg(feature = "simd")]
static DOTPROD_CCC: OnceLock<DotProdCccFn> = OnceLock::new();

impl DotProd<Complex<f32>> for [Complex<f32>] {
    type Output = Complex<f32>;

    #[cfg(not(feature = "simd"))]
    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        self.iter().zip(other).map(|(a, b)| a * b).sum()
    }

    #[cfg(feature = "simd")]
    fn dotprod(&self, other: &[Complex<f32>]) -> Complex<f32> {
        let f = DOTPROD_CCC.get_or_init(|| {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    return dotprod_ccc_avx512;
                }
                if is_x86_feature_detected!("avx2") {
                    return dotprod_ccc_avx2;
                }
                if is_x86_feature_detected!("sse") {
                    return dotprod_ccc_sse;
                }
            }
            dotprod_ccc_scalar
        });
        unsafe { f(self, other) }
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

// Scalar fallback
#[cfg(feature = "simd")]
unsafe fn dotprod_ccc_scalar(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "sse")]
unsafe fn dotprod_ccc_sse(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    if a.len() < 8 {
        return dotprod_ccc_scalar(a, b);
    }
    dotprod_ccc_sse_f32x4(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "sse")]
unsafe fn dotprod_ccc_sse_f32x4(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    let sign = f32x4::from_array([1.0, -1.0, 1.0, -1.0]);

    let mut sum_re0 = f32x4::splat(0.0);
    let mut sum_re1 = f32x4::splat(0.0);
    let mut sum_re2 = f32x4::splat(0.0);
    let mut sum_re3 = f32x4::splat(0.0);
    let mut sum_im0 = f32x4::splat(0.0);
    let mut sum_im1 = f32x4::splat(0.0);
    let mut sum_im2 = f32x4::splat(0.0);
    let mut sum_im3 = f32x4::splat(0.0);

    let chunks = a.len() / 8;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;

    for i in 0..chunks {
        let base = i * 16; // 8 complex × 2 floats

        let a0 = f32x4::from_array(*(a_ptr.add(base) as *const [f32; 4]));
        let b0 = f32x4::from_array(*(b_ptr.add(base) as *const [f32; 4]));
        let a1 = f32x4::from_array(*(a_ptr.add(base + 4) as *const [f32; 4]));
        let b1 = f32x4::from_array(*(b_ptr.add(base + 4) as *const [f32; 4]));
        let a2 = f32x4::from_array(*(a_ptr.add(base + 8) as *const [f32; 4]));
        let b2 = f32x4::from_array(*(b_ptr.add(base + 8) as *const [f32; 4]));
        let a3 = f32x4::from_array(*(a_ptr.add(base + 12) as *const [f32; 4]));
        let b3 = f32x4::from_array(*(b_ptr.add(base + 12) as *const [f32; 4]));

        // Real part: a.re*b.re - a.im*b.im (using sign flip)
        sum_re0 += a0 * b0 * sign;
        sum_re1 += a1 * b1 * sign;
        sum_re2 += a2 * b2 * sign;
        sum_re3 += a3 * b3 * sign;

        // Imag part: a.re*b.im + a.im*b.re (swizzle a, multiply by b)
        let a0_swap: f32x4 = simd_swizzle!(a0, [1, 0, 3, 2]);
        let a1_swap: f32x4 = simd_swizzle!(a1, [1, 0, 3, 2]);
        let a2_swap: f32x4 = simd_swizzle!(a2, [1, 0, 3, 2]);
        let a3_swap: f32x4 = simd_swizzle!(a3, [1, 0, 3, 2]);
        sum_im0 += a0_swap * b0;
        sum_im1 += a1_swap * b1;
        sum_im2 += a2_swap * b2;
        sum_im3 += a3_swap * b3;
    }

    sum_re0 += sum_re1;
    sum_re2 += sum_re3;
    sum_re0 += sum_re2;
    sum_im0 += sum_im1;
    sum_im2 += sum_im3;
    sum_im0 += sum_im2;

    let mut result = Complex::new(reduce_sum_sse_f32x4(sum_re0), reduce_sum_sse_f32x4(sum_im0));

    let a_rem = &a[chunks * 8..];
    let b_rem = &b[chunks * 8..];
    for (a, b) in a_rem.iter().zip(b_rem) {
        result += a * b;
    }
    result
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn dotprod_ccc_avx2(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    if a.len() < 8 {
        return dotprod_ccc_scalar(a, b);
    } else if a.len() < 16 {
        return dotprod_ccc_sse_f32x4(a, b);
    }
    dotprod_ccc_avx2_f32x8(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn dotprod_ccc_avx2_f32x8(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    let sign = f32x8::from_array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

    let mut sum_re0 = f32x8::splat(0.0);
    let mut sum_re1 = f32x8::splat(0.0);
    let mut sum_re2 = f32x8::splat(0.0);
    let mut sum_re3 = f32x8::splat(0.0);
    let mut sum_im0 = f32x8::splat(0.0);
    let mut sum_im1 = f32x8::splat(0.0);
    let mut sum_im2 = f32x8::splat(0.0);
    let mut sum_im3 = f32x8::splat(0.0);

    let chunks = a.len() / 16;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;

    for i in 0..chunks {
        let base = i * 32; // 16 complex × 2 floats

        let a0 = f32x8::from_array(*(a_ptr.add(base) as *const [f32; 8]));
        let b0 = f32x8::from_array(*(b_ptr.add(base) as *const [f32; 8]));
        let a1 = f32x8::from_array(*(a_ptr.add(base + 8) as *const [f32; 8]));
        let b1 = f32x8::from_array(*(b_ptr.add(base + 8) as *const [f32; 8]));
        let a2 = f32x8::from_array(*(a_ptr.add(base + 16) as *const [f32; 8]));
        let b2 = f32x8::from_array(*(b_ptr.add(base + 16) as *const [f32; 8]));
        let a3 = f32x8::from_array(*(a_ptr.add(base + 24) as *const [f32; 8]));
        let b3 = f32x8::from_array(*(b_ptr.add(base + 24) as *const [f32; 8]));

        // Real part: a.re*b.re - a.im*b.im (using sign flip)
        sum_re0 += a0 * b0 * sign;
        sum_re1 += a1 * b1 * sign;
        sum_re2 += a2 * b2 * sign;
        sum_re3 += a3 * b3 * sign;

        // Imag part: a.re*b.im + a.im*b.re (swizzle a, multiply by b)
        let a0_swap: f32x8 = simd_swizzle!(a0, [1, 0, 3, 2, 5, 4, 7, 6]);
        let a1_swap: f32x8 = simd_swizzle!(a1, [1, 0, 3, 2, 5, 4, 7, 6]);
        let a2_swap: f32x8 = simd_swizzle!(a2, [1, 0, 3, 2, 5, 4, 7, 6]);
        let a3_swap: f32x8 = simd_swizzle!(a3, [1, 0, 3, 2, 5, 4, 7, 6]);
        sum_im0 += a0_swap * b0;
        sum_im1 += a1_swap * b1;
        sum_im2 += a2_swap * b2;
        sum_im3 += a3_swap * b3;
    }

    sum_re0 += sum_re1;
    sum_re2 += sum_re3;
    sum_re0 += sum_re2;
    sum_im0 += sum_im1;
    sum_im2 += sum_im3;
    sum_im0 += sum_im2;

    let mut result = Complex::new(reduce_sum_avx2_f32x8(sum_re0), reduce_sum_avx2_f32x8(sum_im0));

    let a_rem = &a[chunks * 16..];
    let b_rem = &b[chunks * 16..];
    for (a, b) in a_rem.iter().zip(b_rem) {
        result += a * b;
    }
    result
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_ccc_avx512(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    if a.len() < 8 {
        return dotprod_ccc_scalar(a, b);
    } else if a.len() < 32 {
        return dotprod_ccc_sse_f32x4(a, b);
    }
    dotprod_ccc_avx512_f32x16(a, b)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_ccc_avx512_f32x16(a: &[Complex<f32>], b: &[Complex<f32>]) -> Complex<f32> {
    let sign = f32x16::from_array([
        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    ]);

    let mut sum_re0 = f32x16::splat(0.0);
    let mut sum_re1 = f32x16::splat(0.0);
    let mut sum_re2 = f32x16::splat(0.0);
    let mut sum_re3 = f32x16::splat(0.0);
    let mut sum_im0 = f32x16::splat(0.0);
    let mut sum_im1 = f32x16::splat(0.0);
    let mut sum_im2 = f32x16::splat(0.0);
    let mut sum_im3 = f32x16::splat(0.0);

    // Process 32 complex numbers (64 floats) per iteration
    let chunks = a.len() / 32;
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;

    for i in 0..chunks {
        let base = i * 64; // 32 complex × 2 floats

        let a0 = f32x16::from_array(*(a_ptr.add(base) as *const [f32; 16]));
        let b0 = f32x16::from_array(*(b_ptr.add(base) as *const [f32; 16]));
        let a1 = f32x16::from_array(*(a_ptr.add(base + 16) as *const [f32; 16]));
        let b1 = f32x16::from_array(*(b_ptr.add(base + 16) as *const [f32; 16]));
        let a2 = f32x16::from_array(*(a_ptr.add(base + 32) as *const [f32; 16]));
        let b2 = f32x16::from_array(*(b_ptr.add(base + 32) as *const [f32; 16]));
        let a3 = f32x16::from_array(*(a_ptr.add(base + 48) as *const [f32; 16]));
        let b3 = f32x16::from_array(*(b_ptr.add(base + 48) as *const [f32; 16]));

        // Real part: a.re*b.re - a.im*b.im (using sign flip)
        sum_re0 += a0 * b0 * sign;
        sum_re1 += a1 * b1 * sign;
        sum_re2 += a2 * b2 * sign;
        sum_re3 += a3 * b3 * sign;

        // Imag part: a.re*b.im + a.im*b.re (swizzle a, multiply by b)
        let a0_swap: f32x16 = simd_swizzle!(a0, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
        let a1_swap: f32x16 = simd_swizzle!(a1, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
        let a2_swap: f32x16 = simd_swizzle!(a2, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
        let a3_swap: f32x16 = simd_swizzle!(a3, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]);
        sum_im0 += a0_swap * b0;
        sum_im1 += a1_swap * b1;
        sum_im2 += a2_swap * b2;
        sum_im3 += a3_swap * b3;
    }

    sum_re0 += sum_re1;
    sum_re2 += sum_re3;
    sum_re0 += sum_re2;
    sum_im0 += sum_im1;
    sum_im2 += sum_im3;
    sum_im0 += sum_im2;

    let mut result = Complex::new(reduce_sum_avx512_f32x16(sum_re0), reduce_sum_avx512_f32x16(sum_im0));

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
    use approx::assert_relative_eq;
    use rand::Rng;
    use test_macro::autotest_annotate;

    type Cf32 = Complex<f32>;

    #[test]
    fn test_dotprod_ccc() {
        let a = vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0), Complex::new(3.0, 3.0)];
        let b = vec![Complex::new(4.0, -4.0), Complex::new(5.0, -5.0), Complex::new(6.0, -6.0)];
        assert_eq!(a.dotprod(&b), Complex::new(64.0, 0.0));
    }

    #[test]
    #[autotest_annotate(autotest_dotprod_cccf_rand16)]
    fn test_dotprod_cccf_rand16() {
        const TOL: f32 = 1e-3;

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
        const TOL: f32 = 1e-3;
        let mut rng = rand::thread_rng();

        for n in 1..=512 {
            let h: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();
            let x: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();

            let y_test: Cf32 = h.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let y_struct = h.dotprod(&x);

            assert_relative_eq!(y_struct.re, y_test.re, epsilon = TOL);
            assert_relative_eq!(y_struct.im, y_test.im, epsilon = TOL);
        }
    }

    #[test]
    fn test_dotprod_ccc_boundary_lengths() {
        const TOL: f32 = 1e-3;

        // Complex×Complex: same thresholds as CRC
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
            let a: Vec<Cf32> = (0..n)
                .map(|i| Cf32::new((i as f32 * 0.1).sin(), (i as f32 * 0.15).cos()))
                .collect();
            let b: Vec<Cf32> = (0..n)
                .map(|i| Cf32::new((i as f32 * 0.2).cos(), (i as f32 * 0.25).sin()))
                .collect();

            let expected: Cf32 = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum();
            let result = a.dotprod(&b);

            assert_relative_eq!(result.re, expected.re, epsilon = TOL);
            assert_relative_eq!(result.im, expected.im, epsilon = TOL);
        }
    }

    // Direct tests for each SIMD tier
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_ccc_sse_direct() {
        if !is_x86_feature_detected!("sse") {
            return;
        }
        let mut rng = rand::thread_rng();
        const TOL: f32 = 1e-3;

        for n in 1..=256 {
            let a: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();
            let b: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();

            let y_test: Cf32 = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum();
            let y_sse = unsafe { dotprod_ccc_sse(&a, &b) };

            assert_relative_eq!(y_sse.re, y_test.re, epsilon = TOL);
            assert_relative_eq!(y_sse.im, y_test.im, epsilon = TOL);
        }
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn test_dotprod_ccc_avx2_direct() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = rand::thread_rng();
        const TOL: f32 = 1e-3;

        for n in 1..=256 {
            let a: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();
            let b: Vec<Cf32> = (0..n).map(|_| Cf32::new(rng.gen(), rng.gen())).collect();

            let y_test: Cf32 = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum();
            let y_avx2 = unsafe { dotprod_ccc_avx2(&a, &b) };

            assert_relative_eq!(y_avx2.re, y_test.re, epsilon = TOL);
            assert_relative_eq!(y_avx2.im, y_test.im, epsilon = TOL);
        }
    }
}
