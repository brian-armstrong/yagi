use num_complex::Complex;

use super::{DotProdBlockKernel, DotProdBlockPlan};

use std::simd::f32x4;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::simd::f32x16;

use super::reduce::reduce_sum_complex_sse_f32x4;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::reduce::reduce_sum_complex_avx512_f32x16;

fn expand_real_coefficients(h: &[f32], padded_len: usize) -> Vec<f32> {
    // by duplicating each coefficient, we can avoid some extra work in the kernel
    // like the rrr version, we will also apply padding to the end
    let mut prepared = vec![0.0; padded_len * 2];
    for (expanded, &coefficient) in prepared.chunks_exact_mut(2).zip(h) {
        expanded.fill(coefficient);
    }
    prepared
}

pub(super) fn plan_dotprod_crc_block_f32x4(
    h: &[f32],
) -> Option<DotProdBlockPlan<[Complex<f32>], f32, Complex<f32>>> {
    let padded_len = h.len().next_multiple_of(2);
    let executor = match padded_len {
         2 => dotprod_crc_block_f32x4::<2> as DotProdBlockKernel<[Complex<f32>], f32, Complex<f32>>,
         4 => dotprod_crc_block_f32x4::<4>,
         6 => dotprod_crc_block_f32x4::<6>,
         8 => dotprod_crc_block_f32x4::<8>,
        10 => dotprod_crc_block_f32x4::<10>,
        12 => dotprod_crc_block_f32x4::<12>,
        14 => dotprod_crc_block_f32x4::<14>,
        16 => dotprod_crc_block_f32x4::<16>,
        18 => dotprod_crc_block_f32x4::<18>,
        20 => dotprod_crc_block_f32x4::<20>,
        22 => dotprod_crc_block_f32x4::<22>,
        24 => dotprod_crc_block_f32x4::<24>,
        26 => dotprod_crc_block_f32x4::<26>,
        28 => dotprod_crc_block_f32x4::<28>,
        30 => dotprod_crc_block_f32x4::<30>,
        32 => dotprod_crc_block_f32x4::<32>,
        _ => return None,
    };
    let prepared = expand_real_coefficients(h, padded_len);
    Some(DotProdBlockPlan::new(prepared, padded_len, 4, executor))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) fn plan_dotprod_crc_block_avx512(
    h: &[f32],
) -> Option<DotProdBlockPlan<[Complex<f32>], f32, Complex<f32>>> {
    let padded_len = h.len().next_multiple_of(8);
    let executor = match padded_len {
        16 => dotprod_crc_block_avx512_register::<16> as DotProdBlockKernel<[Complex<f32>], f32, Complex<f32>>,
        24 => dotprod_crc_block_avx512_register::<24>,
        32 => dotprod_crc_block_avx512_register::<32>,
        40 => dotprod_crc_block_avx512_register::<40>,
        48 => dotprod_crc_block_avx512_register::<48>,
        56 => dotprod_crc_block_avx512_register::<56>,
        64 => dotprod_crc_block_avx512_register::<64>,
        72 => dotprod_crc_block_avx512_register::<72>,
        80 => dotprod_crc_block_avx512_register::<80>,
        88 => dotprod_crc_block_avx512_register::<88>,
        96 => dotprod_crc_block_avx512_register::<96>,
        104 => dotprod_crc_block_avx512_register::<104>,
        112 => dotprod_crc_block_avx512_register::<112>,
        120 => dotprod_crc_block_avx512_register::<120>,
        128 => dotprod_crc_block_avx512_register::<128>,
        136.. => dotprod_crc_block_avx512_memory,
        _ => return None,
    };
    let prepared = expand_real_coefficients(h, padded_len);
    Some(DotProdBlockPlan::new(prepared, padded_len, 4, executor))
}

unsafe fn dotprod_crc_block_f32x4<const N: usize>(
    x: &[Complex<f32>],
    h: &[f32],
    y: &mut [Complex<f32>],
) -> usize {
    // this is largely the same algorithm as `dotprod_rrr_block_f32x4`
    // for Complex<f32> x {x.re, x.im} and f32 h {h, h} (with duplication)
    // the dotprod y is {h * x.re, h * x.im}
    // so it's as if we have twice as many coefficients

    // this kernel supports up to 32 coefficients with 4 outputs per loop

    debug_assert_eq!(h.len(), N * 2);
    let safe_outputs = x.len().saturating_sub(N - 1).min(y.len());
    let bulk_outputs = safe_outputs / 4 * 4;
    if bulk_outputs == 0 {
        return 0;
    }

    std::hint::assert_unchecked(h.len() == N * 2);
    std::hint::assert_unchecked(x.len() >= bulk_outputs + N - 1);
    std::hint::assert_unchecked(y.len() >= bulk_outputs);

    let hp = h.as_ptr();
    macro_rules! load_coefficient {
        ($name:ident, $threshold:literal, $offset:literal) => {
            let $name = if N > $threshold {
                f32x4::from_array(*(hp.add($offset) as *const [f32; 4]))
            } else {
                f32x4::splat(0.0)
            };
        };
    }
    load_coefficient!(h0, 0, 0);
    load_coefficient!(h1, 2, 4);
    load_coefficient!(h2, 4, 8);
    load_coefficient!(h3, 6, 12);
    load_coefficient!(h4, 8, 16);
    load_coefficient!(h5, 10, 20);
    load_coefficient!(h6, 12, 24);
    load_coefficient!(h7, 14, 28);
    load_coefficient!(h8, 16, 32);
    load_coefficient!(h9, 18, 36);
    load_coefficient!(h10, 20, 40);
    load_coefficient!(h11, 22, 44);
    load_coefficient!(h12, 24, 48);
    load_coefficient!(h13, 26, 52);
    load_coefficient!(h14, 28, 56);
    load_coefficient!(h15, 30, 60);

    let xp = x.as_ptr() as *const f32;
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        let base = i * 2;
        let mut a0 = f32x4::from_array(*(xp.add(base) as *const [f32; 4])) * h0;
        let mut a1 = f32x4::from_array(*(xp.add(base + 2) as *const [f32; 4])) * h0;
        let mut a2 = f32x4::from_array(*(xp.add(base + 4) as *const [f32; 4])) * h0;
        let mut a3 = f32x4::from_array(*(xp.add(base + 6) as *const [f32; 4])) * h0;

        macro_rules! accumulate {
            ($threshold:literal, $offset:literal, $coeff:ident) => {
                if N > $threshold {
                    a0 += f32x4::from_array(
                        *(xp.add(base + $offset) as *const [f32; 4]),
                    ) * $coeff;
                    a1 += f32x4::from_array(
                        *(xp.add(base + $offset + 2) as *const [f32; 4]),
                    ) * $coeff;
                    a2 += f32x4::from_array(
                        *(xp.add(base + $offset + 4) as *const [f32; 4]),
                    ) * $coeff;
                    a3 += f32x4::from_array(
                        *(xp.add(base + $offset + 6) as *const [f32; 4]),
                    ) * $coeff;
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
        accumulate!(16, 32, h8);
        accumulate!(18, 36, h9);
        accumulate!(20, 40, h10);
        accumulate!(22, 44, h11);
        accumulate!(24, 48, h12);
        accumulate!(26, 52, h13);
        accumulate!(28, 56, h14);
        accumulate!(30, 60, h15);

        *yp.add(i) = reduce_sum_complex_sse_f32x4(a0);
        *yp.add(i + 1) = reduce_sum_complex_sse_f32x4(a1);
        *yp.add(i + 2) = reduce_sum_complex_sse_f32x4(a2);
        *yp.add(i + 3) = reduce_sum_complex_sse_f32x4(a3);
        i += 4;
    }

    bulk_outputs
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_crc_block_avx512_register<const N: usize>(
    x: &[Complex<f32>],
    h: &[f32],
    y: &mut [Complex<f32>],
) -> usize {
    // this is a wider version of `dotprod_crc_block_f32x4`
    // more coefficients, but still 4 outputs at a time

    // this kernel supports up to 128 coefficients and handles 4 outputs at a time

    debug_assert_eq!(h.len(), N * 2);
    let safe_outputs = x.len().saturating_sub(N - 1).min(y.len());
    let bulk_outputs = safe_outputs / 4 * 4;
    if bulk_outputs == 0 {
        return 0;
    }

    std::hint::assert_unchecked(h.len() == N * 2);
    std::hint::assert_unchecked(x.len() >= bulk_outputs + N - 1);
    std::hint::assert_unchecked(y.len() >= bulk_outputs);

    let hp = h.as_ptr();
    let h0 = f32x16::from_array(*(hp as *const [f32; 16]));
    macro_rules! load_coefficient {
        ($name:ident, $threshold:literal, $offset:literal) => {
            let $name = if N > $threshold {
                f32x16::from_array(*(hp.add($offset) as *const [f32; 16]))
            } else {
                f32x16::splat(0.0)
            };
        };
    }
    load_coefficient!(h1, 8, 16);
    load_coefficient!(h2, 16, 32);
    load_coefficient!(h3, 24, 48);
    load_coefficient!(h4, 32, 64);
    load_coefficient!(h5, 40, 80);
    load_coefficient!(h6, 48, 96);
    load_coefficient!(h7, 56, 112);
    load_coefficient!(h8, 64, 128);
    load_coefficient!(h9, 72, 144);
    load_coefficient!(h10, 80, 160);
    load_coefficient!(h11, 88, 176);
    load_coefficient!(h12, 96, 192);
    load_coefficient!(h13, 104, 208);
    load_coefficient!(h14, 112, 224);
    load_coefficient!(h15, 120, 240);

    let xp = x.as_ptr() as *const f32;
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        let base = i * 2;
        let mut a0 = f32x16::from_array(*(xp.add(base) as *const [f32; 16])) * h0;
        let mut a1 = f32x16::from_array(*(xp.add(base + 2) as *const [f32; 16])) * h0;
        let mut a2 = f32x16::from_array(*(xp.add(base + 4) as *const [f32; 16])) * h0;
        let mut a3 = f32x16::from_array(*(xp.add(base + 6) as *const [f32; 16])) * h0;

        macro_rules! accumulate {
            ($threshold:literal, $offset:literal, $coeff:ident) => {
                if N > $threshold {
                    a0 += f32x16::from_array(
                        *(xp.add(base + $offset) as *const [f32; 16]),
                    ) * $coeff;
                    a1 += f32x16::from_array(
                        *(xp.add(base + $offset + 2) as *const [f32; 16]),
                    ) * $coeff;
                    a2 += f32x16::from_array(
                        *(xp.add(base + $offset + 4) as *const [f32; 16]),
                    ) * $coeff;
                    a3 += f32x16::from_array(
                        *(xp.add(base + $offset + 6) as *const [f32; 16]),
                    ) * $coeff;
                }
            };
        }

        accumulate!(8, 16, h1);
        accumulate!(16, 32, h2);
        accumulate!(24, 48, h3);
        accumulate!(32, 64, h4);
        accumulate!(40, 80, h5);
        accumulate!(48, 96, h6);
        accumulate!(56, 112, h7);
        accumulate!(64, 128, h8);
        accumulate!(72, 144, h9);
        accumulate!(80, 160, h10);
        accumulate!(88, 176, h11);
        accumulate!(96, 192, h12);
        accumulate!(104, 208, h13);
        accumulate!(112, 224, h14);
        accumulate!(120, 240, h15);

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
unsafe fn dotprod_crc_block_avx512_memory(
    x: &[Complex<f32>],
    h: &[f32],
    y: &mut [Complex<f32>],
) -> usize {
    // see `dotprod_rrr_block_avx512_memory`
    // instead of keeping coefficients in register, load from memory every 4 outputs
    // handles multiple widths

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

    let hp = h.as_ptr();
    let xp = x.as_ptr() as *const f32;
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        let base = i * 2;
        let h = f32x16::from_array(*(hp as *const [f32; 16]));
        let mut a0 = f32x16::from_array(*(xp.add(base) as *const [f32; 16])) * h;
        let mut a1 = f32x16::from_array(*(xp.add(base + 2) as *const [f32; 16])) * h;
        let mut a2 = f32x16::from_array(*(xp.add(base + 4) as *const [f32; 16])) * h;
        let mut a3 = f32x16::from_array(*(xp.add(base + 6) as *const [f32; 16])) * h;

        macro_rules! accumulate {
            ($offset:expr) => {{
                let offset = $offset;
                let h = f32x16::from_array(*(hp.add(offset) as *const [f32; 16]));
                a0 += f32x16::from_array(
                    *(xp.add(base + offset) as *const [f32; 16]),
                ) * h;
                a1 += f32x16::from_array(
                    *(xp.add(base + offset + 2) as *const [f32; 16]),
                ) * h;
                a2 += f32x16::from_array(
                    *(xp.add(base + offset + 4) as *const [f32; 16]),
                ) * h;
                a3 += f32x16::from_array(
                    *(xp.add(base + offset + 6) as *const [f32; 16]),
                ) * h;
            }};
        }

        let mut j = 8;
        // 4-wide loop with 1-wide tail
        while j + 24 < n {
            accumulate!(j * 2);
            accumulate!((j + 8) * 2);
            accumulate!((j + 16) * 2);
            accumulate!((j + 24) * 2);
            j += 32;
        }
        while j < n {
            accumulate!(j * 2);
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
