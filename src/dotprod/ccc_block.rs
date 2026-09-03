use num_complex::Complex;

use super::{DotProdBlockKernel, DotProdBlockPlan};

use std::simd::{f32x4, simd_swizzle};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::simd::f32x16;

use super::reduce::reduce_sum_sse_f32x4;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::reduce::reduce_sum_complex_avx512_f32x16;

pub(super) fn plan_dotprod_ccc_block_f32x4(
    h: &[Complex<f32>],
) -> Option<DotProdBlockPlan<[Complex<f32>], Complex<f32>, Complex<f32>>> {
    let padded_len = h.len().next_multiple_of(2);
    let executor = match padded_len {
         2 => dotprod_ccc_block_f32x4::<2> as DotProdBlockKernel<[Complex<f32>], Complex<f32>, Complex<f32>>,
         4 => dotprod_ccc_block_f32x4::<4>,
         6 => dotprod_ccc_block_f32x4::<6>,
         8 => dotprod_ccc_block_f32x4::<8>,
        10 => dotprod_ccc_block_f32x4::<10>,
        12 => dotprod_ccc_block_f32x4::<12>,
        14 => dotprod_ccc_block_f32x4::<14>,
        16 => dotprod_ccc_block_f32x4::<16>,
        _ => return None,
    };

    // coefficients are zero-padded (suffix) to allow bulk SIMD operations
    let mut prepared = vec![Complex::new(0.0, 0.0); padded_len];
    prepared[..h.len()].copy_from_slice(h);
    Some(DotProdBlockPlan::new(prepared, padded_len, 2, executor))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) fn plan_dotprod_ccc_block_avx512(
    h: &[Complex<f32>],
) -> Option<DotProdBlockPlan<[Complex<f32>], Complex<f32>, Complex<f32>>> {
    let padded_len = h.len().next_multiple_of(8);
    let executor = match padded_len {
         8 => dotprod_ccc_block_avx512_register::<8> as DotProdBlockKernel<[Complex<f32>], Complex<f32>, Complex<f32>>,
        16 => dotprod_ccc_block_avx512_register::<16>,
        24 => dotprod_ccc_block_avx512_register::<24>,
        32 => dotprod_ccc_block_avx512_register::<32>,
        40 => dotprod_ccc_block_avx512_register::<40>,
        48 => dotprod_ccc_block_avx512_register::<48>,
        56 => dotprod_ccc_block_avx512_register::<56>,
        64 => dotprod_ccc_block_avx512_register::<64>,
        72.. => dotprod_ccc_block_avx512_memory,
        _ => return None,
    };

    // see `dotprod_ccc_block_avx512_register` for an explanation of this arrangement
    // duplicating the coefficients and de-interleaving them allows an extra optimization
    let mut prepared = vec![Complex::new(0.0, 0.0); padded_len * 2];
    let (real, imag) = prepared.split_at_mut(padded_len);
    for ((real, imag), &coefficient) in real.iter_mut().zip(imag).zip(h) {
        *real = Complex::new(coefficient.re, coefficient.re);
        *imag = Complex::new(-coefficient.im, coefficient.im);
    }
    Some(DotProdBlockPlan::new(prepared, padded_len, 4, executor))
}

unsafe fn dotprod_ccc_block_f32x4<const N: usize>(
    x: &[Complex<f32>],
    h: &[Complex<f32>],
    y: &mut [Complex<f32>],
) -> usize {
    // same idea as `dotprod_rrr_block_f32x4` but with the complex-complex dotprod

    // 16 coefficients, 2 outputs at a time

    debug_assert_eq!(h.len(), N);
    let safe_outputs = x.len().saturating_sub(N - 1).min(y.len());
    let bulk_outputs = safe_outputs / 2 * 2;
    if bulk_outputs == 0 {
        return 0;
    }

    std::hint::assert_unchecked(h.len() == N);
    std::hint::assert_unchecked(x.len() >= bulk_outputs + N - 1);
    std::hint::assert_unchecked(y.len() >= bulk_outputs);

    let hp = h.as_ptr() as *const f32;
    let h0 = f32x4::from_array(*(hp as *const [f32; 4]));
    macro_rules! load_coefficient {
        ($name:ident, $threshold:literal, $offset:literal) => {
            let $name = if N > $threshold {
                f32x4::from_array(*(hp.add($offset) as *const [f32; 4]))
            } else {
                f32x4::splat(0.0)
            };
        };
    }
    load_coefficient!(h1, 2, 4);
    load_coefficient!(h2, 4, 8);
    load_coefficient!(h3, 6, 12);
    load_coefficient!(h4, 8, 16);
    load_coefficient!(h5, 10, 20);
    load_coefficient!(h6, 12, 24);
    load_coefficient!(h7, 14, 28);

    let sign = f32x4::from_array([1.0, -1.0, 1.0, -1.0]);
    let xp = x.as_ptr() as *const f32;
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        let base = i * 2;
        let x0 = f32x4::from_array(*(xp.add(base) as *const [f32; 4]));
        let x1 = f32x4::from_array(*(xp.add(base + 2) as *const [f32; 4]));
        // y0 = {(x0_0 * h0_0 - x0_1 * h0_1, x0_1 * h0_0 + x0_0 * h0_1), ...}
        let mut re0 = x0 * h0 * sign;
        let mut im0 = simd_swizzle!(x0, [1, 0, 3, 2]) * h0;
        let mut re1 = x1 * h0 * sign;
        let mut im1 = simd_swizzle!(x1, [1, 0, 3, 2]) * h0;

        macro_rules! accumulate {
            ($threshold:literal, $offset:literal, $coeff:ident) => {
                if N > $threshold {
                    let x0 = f32x4::from_array(
                        *(xp.add(base + $offset) as *const [f32; 4]),
                    );
                    let x1 = f32x4::from_array(
                        *(xp.add(base + $offset + 2) as *const [f32; 4]),
                    );
                    re0 += x0 * $coeff * sign;
                    im0 += simd_swizzle!(x0, [1, 0, 3, 2]) * $coeff;
                    re1 += x1 * $coeff * sign;
                    im1 += simd_swizzle!(x1, [1, 0, 3, 2]) * $coeff;
                }
            };
        }

        accumulate!(2, 4, h1);
        accumulate!(4, 8, h2);
        accumulate!(6, 12, h3);
        accumulate!(8, 16, h4);
        accumulate!(10, 20, h5);
        accumulate!(12, 24, h6);
        accumulate!(14, 28, h7);

        *yp.add(i) = Complex::new(
            reduce_sum_sse_f32x4(re0),
            reduce_sum_sse_f32x4(im0),
        );
        *yp.add(i + 1) = Complex::new(
            reduce_sum_sse_f32x4(re1),
            reduce_sum_sse_f32x4(im1),
        );
        i += 2;
    }

    bulk_outputs
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_ccc_block_avx512_register<const N: usize>(
    x: &[Complex<f32>],
    h: &[Complex<f32>],
    y: &mut [Complex<f32>],
) -> usize {
    // this kernel prepares the cofficients in order to save work in the loop
    // we keep one region with {h0.re, h0.re, h1.re, h1.re} (duplicated re)
    // we keep another region with {-h0.im, h0.im, -h1.im, h1.im} (duplicated im, first negated)
    // each region is kept in its own register e.g. h_re0 and h_im0
    // then the output is just x0 * h_re0 + swp(x0) * h_im0, where swp() swaps re and im on x

    // 64 coefficients, 4 outputs at a time

    debug_assert_eq!(h.len(), N * 2);
    let safe_outputs = x.len().saturating_sub(N - 1).min(y.len());
    let bulk_outputs = safe_outputs / 4 * 4;
    if bulk_outputs == 0 {
        return 0;
    }

    std::hint::assert_unchecked(h.len() == N * 2);
    std::hint::assert_unchecked(x.len() >= bulk_outputs + N - 1);
    std::hint::assert_unchecked(y.len() >= bulk_outputs);

    // imaginary coefficients are stored separately from real coefficients here
    let hp = h.as_ptr() as *const f32;
    let ip = hp.add(N * 2);
    macro_rules! load_coefficients {
        ($real:ident, $imag:ident, $threshold:literal, $offset:literal) => {
            let ($real, $imag) = if N > $threshold {
                (
                    f32x16::from_array(*(hp.add($offset) as *const [f32; 16])),
                    f32x16::from_array(*(ip.add($offset) as *const [f32; 16])),
                )
            } else {
                (f32x16::splat(0.0), f32x16::splat(0.0))
            };
        };
    }
    load_coefficients!(h_re0, h_im0, 0, 0);
    load_coefficients!(h_re1, h_im1, 8, 16);
    load_coefficients!(h_re2, h_im2, 16, 32);
    load_coefficients!(h_re3, h_im3, 24, 48);
    load_coefficients!(h_re4, h_im4, 32, 64);
    load_coefficients!(h_re5, h_im5, 40, 80);
    load_coefficients!(h_re6, h_im6, 48, 96);
    load_coefficients!(h_re7, h_im7, 56, 112);

    macro_rules! swap_complex {
        ($value:expr) => {
            simd_swizzle!(
                $value,
                [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
            )
        };
    }

    let xp = x.as_ptr() as *const f32;
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        let base = i * 2;
        let x0 = f32x16::from_array(*(xp.add(base) as *const [f32; 16]));
        let x1 = f32x16::from_array(*(xp.add(base + 2) as *const [f32; 16]));
        let x2 = f32x16::from_array(*(xp.add(base + 4) as *const [f32; 16]));
        let x3 = f32x16::from_array(*(xp.add(base + 6) as *const [f32; 16]));
        let mut a0 = x0 * h_re0 + swap_complex!(x0) * h_im0;
        let mut a1 = x1 * h_re0 + swap_complex!(x1) * h_im0;
        let mut a2 = x2 * h_re0 + swap_complex!(x2) * h_im0;
        let mut a3 = x3 * h_re0 + swap_complex!(x3) * h_im0;

        macro_rules! accumulate {
            ($threshold:literal, $offset:literal, $real:ident, $imag:ident) => {
                if N > $threshold {
                    let x0 = f32x16::from_array(
                        *(xp.add(base + $offset) as *const [f32; 16]),
                    );
                    let x1 = f32x16::from_array(
                        *(xp.add(base + $offset + 2) as *const [f32; 16]),
                    );
                    let x2 = f32x16::from_array(
                        *(xp.add(base + $offset + 4) as *const [f32; 16]),
                    );
                    let x3 = f32x16::from_array(
                        *(xp.add(base + $offset + 6) as *const [f32; 16]),
                    );
                    a0 += x0 * $real + swap_complex!(x0) * $imag;
                    a1 += x1 * $real + swap_complex!(x1) * $imag;
                    a2 += x2 * $real + swap_complex!(x2) * $imag;
                    a3 += x3 * $real + swap_complex!(x3) * $imag;
                }
            };
        }

        accumulate!(8, 16, h_re1, h_im1);
        accumulate!(16, 32, h_re2, h_im2);
        accumulate!(24, 48, h_re3, h_im3);
        accumulate!(32, 64, h_re4, h_im4);
        accumulate!(40, 80, h_re5, h_im5);
        accumulate!(48, 96, h_re6, h_im6);
        accumulate!(56, 112, h_re7, h_im7);

        *yp.add(i) = reduce_sum_complex_avx512_f32x16(a0);
        *yp.add(i + 1) = reduce_sum_complex_avx512_f32x16(a1);
        *yp.add(i + 2) = reduce_sum_complex_avx512_f32x16(a2);
        *yp.add(i + 3) = reduce_sum_complex_avx512_f32x16(a3);
        i += 4;
    }

    bulk_outputs
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_ccc_block_avx512_memory(
    x: &[Complex<f32>],
    h: &[Complex<f32>],
    y: &mut [Complex<f32>],
) -> usize {
    // similar to `dotprod_rrr_block_avx512_memory`

    // reuse the duplicated coefficients concept from `dotprod_ccc_block_avx512_register`
    // no persistent register coefficients here

    let n = h.len() / 2;
    debug_assert_eq!(h.len(), n * 2);
    debug_assert_eq!(n % 8, 0);
    let safe_outputs = x.len().saturating_sub(n - 1).min(y.len());
    let bulk_outputs = safe_outputs / 4 * 4;
    if bulk_outputs == 0 {
        return 0;
    }

    std::hint::assert_unchecked(h.len() == n * 2);
    std::hint::assert_unchecked(n >= 8 && n % 8 == 0);
    std::hint::assert_unchecked(x.len() >= bulk_outputs + n - 1);
    std::hint::assert_unchecked(y.len() >= bulk_outputs);

    macro_rules! swap_complex {
        ($value:expr) => {
            simd_swizzle!(
                $value,
                [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
            )
        };
    }

    let hp = h.as_ptr() as *const f32;
    let ip = hp.add(n * 2);
    let xp = x.as_ptr() as *const f32;
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        let base = i * 2;
        let hr = f32x16::from_array(*(hp as *const [f32; 16]));
        let hi = f32x16::from_array(*(ip as *const [f32; 16]));
        let x0 = f32x16::from_array(*(xp.add(base) as *const [f32; 16]));
        let x1 = f32x16::from_array(*(xp.add(base + 2) as *const [f32; 16]));
        let x2 = f32x16::from_array(*(xp.add(base + 4) as *const [f32; 16]));
        let x3 = f32x16::from_array(*(xp.add(base + 6) as *const [f32; 16]));
        let mut a0 = x0 * hr + swap_complex!(x0) * hi;
        let mut a1 = x1 * hr + swap_complex!(x1) * hi;
        let mut a2 = x2 * hr + swap_complex!(x2) * hi;
        let mut a3 = x3 * hr + swap_complex!(x3) * hi;

        let mut j = 8;
        while j < n {
            let offset = j * 2;
            let hr = f32x16::from_array(*(hp.add(offset) as *const [f32; 16]));
            let hi = f32x16::from_array(*(ip.add(offset) as *const [f32; 16]));
            let x0 = f32x16::from_array(
                *(xp.add(base + offset) as *const [f32; 16]),
            );
            let x1 = f32x16::from_array(
                *(xp.add(base + offset + 2) as *const [f32; 16]),
            );
            let x2 = f32x16::from_array(
                *(xp.add(base + offset + 4) as *const [f32; 16]),
            );
            let x3 = f32x16::from_array(
                *(xp.add(base + offset + 6) as *const [f32; 16]),
            );
            a0 += x0 * hr + swap_complex!(x0) * hi;
            a1 += x1 * hr + swap_complex!(x1) * hi;
            a2 += x2 * hr + swap_complex!(x2) * hi;
            a3 += x3 * hr + swap_complex!(x3) * hi;
            j += 8;
        }

        *yp.add(i) = reduce_sum_complex_avx512_f32x16(a0);
        *yp.add(i + 1) = reduce_sum_complex_avx512_f32x16(a1);
        *yp.add(i + 2) = reduce_sum_complex_avx512_f32x16(a2);
        *yp.add(i + 3) = reduce_sum_complex_avx512_f32x16(a3);
        i += 4;
    }

    bulk_outputs
}
