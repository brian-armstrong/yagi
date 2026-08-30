// Hierarchical SIMD reduction functions

use num_complex::Complex;
use std::simd::{f32x4, simd_swizzle};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::simd::{f32x8, f32x16};

// 128-bit (SSE and others, not arch-specific)
#[inline]
pub unsafe fn reduce_sum_sse_f32x4(v: f32x4) -> f32 {
    let shuffled: f32x4 = simd_swizzle!(v, [2, 3, 0, 1]);
    let sum2 = v + shuffled;
    let shuffled2: f32x4 = simd_swizzle!(sum2, [1, 0, 3, 2]);
    let sum1 = sum2 + shuffled2;
    sum1[0]
}

#[inline]
pub unsafe fn reduce_sum_complex_sse_f32x4(v: f32x4) -> Complex<f32> {
    let hi: f32x4 = simd_swizzle!(v, [2, 3, 0, 1]);
    let sum1 = v + hi;
    Complex::new(sum1[0], sum1[1])
}

// AVX2 (256-bit)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn reduce_sum_avx2_f32x8(v: f32x8) -> f32 {
    let hi: f32x8 = simd_swizzle!(v, [4, 5, 6, 7, 0, 1, 2, 3]);
    let sum4 = v + hi;
    let sum4: f32x4 = simd_swizzle!(sum4, [0, 1, 2, 3]);
    reduce_sum_sse_f32x4(sum4)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn reduce_sum_complex_avx2_f32x8(v: f32x8) -> Complex<f32> {
    let hi: f32x8 = simd_swizzle!(v, [4, 5, 6, 7, 0, 1, 2, 3]);
    let sum2 = v + hi;
    let sum2: f32x4 = simd_swizzle!(sum2, [0, 1, 2, 3]);
    reduce_sum_complex_sse_f32x4(sum2)
}

// AVX-512 (512-bit)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn reduce_sum_avx512_f32x16(v: f32x16) -> f32 {
    let hi: f32x16 = simd_swizzle!(v, [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]);
    let sum8 = v + hi;
    let sum8: f32x8 = simd_swizzle!(sum8, [0, 1, 2, 3, 4, 5, 6, 7]);
    reduce_sum_avx2_f32x8(sum8)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn reduce_sum_complex_avx512_f32x16(v: f32x16) -> Complex<f32> {
    let hi: f32x16 = simd_swizzle!(v, [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]);
    let sum4 = v + hi;
    let sum4: f32x8 = simd_swizzle!(sum4, [0, 1, 2, 3, 4, 5, 6, 7]);
    reduce_sum_complex_avx2_f32x8(sum4)
}
