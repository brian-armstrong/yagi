use super::{DotProdBlockKernel, DotProdBlockPlan};

use std::simd::f32x4;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::simd::f32x16;

use super::reduce::reduce_sum_sse_f32x4;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::reduce::reduce_sum_avx512_f32x16;

pub(super) fn plan_dotprod_rrr_block_f32x4(
    h: &[f32],
) -> Option<DotProdBlockPlan<[f32], f32, f32>> {
    let padded_len = h.len().next_multiple_of(4);
    let executor = match padded_len {
        4 => dotprod_rrr_block_f32x4::<4> as DotProdBlockKernel<[f32], f32, f32>,
        8 => dotprod_rrr_block_f32x4::<8>,
        12 => dotprod_rrr_block_f32x4::<12>,
        16 => dotprod_rrr_block_f32x4::<16>,
        20 => dotprod_rrr_block_f32x4::<20>,
        24 => dotprod_rrr_block_f32x4::<24>,
        28 => dotprod_rrr_block_f32x4::<28>,
        32 => dotprod_rrr_block_f32x4::<32>,
        _ => return None,
    };

    // coefficients are zero-padded (suffix) to allow bulk SIMD operations
    let mut prepared = vec![0.0; padded_len];
    prepared[..h.len()].copy_from_slice(h);
    Some(DotProdBlockPlan::new(prepared, padded_len, 4, executor))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) fn plan_dotprod_rrr_block_avx512(
    h: &[f32],
) -> Option<DotProdBlockPlan<[f32], f32, f32>> {
    let padded_len = h.len().next_multiple_of(16);
    let executor = match padded_len {
         48 => dotprod_rrr_block_avx512_register::<48> as DotProdBlockKernel<[f32], f32, f32>,
         64 => dotprod_rrr_block_avx512_register::<64>,
         80 => dotprod_rrr_block_avx512_register::<80>,
         96 => dotprod_rrr_block_avx512_register::<96>,
        112 => dotprod_rrr_block_avx512_register::<112>,
        128 => dotprod_rrr_block_avx512_register::<128>,
        144 => dotprod_rrr_block_avx512_register::<144>,
        160 => dotprod_rrr_block_avx512_register::<160>,
        176 => dotprod_rrr_block_avx512_register::<176>,
        192 => dotprod_rrr_block_avx512_register::<192>,
        208 => dotprod_rrr_block_avx512_register::<208>,
        224 => dotprod_rrr_block_avx512_register::<224>,
        240 => dotprod_rrr_block_avx512_register::<240>,
        256 => dotprod_rrr_block_avx512_register::<256>,
        272.. => dotprod_rrr_block_avx512_memory,
        _ => return None,
    };
    let mut prepared = vec![0.0; padded_len];
    prepared[..h.len()].copy_from_slice(h);
    Some(DotProdBlockPlan::new(prepared, padded_len, 4, executor))
}

unsafe fn dotprod_rrr_block_f32x4<const N: usize>(
    x: &[f32],
    h: &[f32],
    y: &mut [f32],
) -> usize {
    // this const block can load up to 32 coefficients across 8 simd registers
    // it will then process 4 outputs at a time in the loop below

    debug_assert_eq!(h.len(), N);
    let safe_outputs = x.len().saturating_sub(N - 1).min(y.len());
    let bulk_outputs = safe_outputs / 4 * 4;
    if bulk_outputs == 0 {
        return 0;
    }

    std::hint::assert_unchecked(h.len() == N);
    std::hint::assert_unchecked(x.len() >= bulk_outputs + N - 1);
    std::hint::assert_unchecked(y.len() >= bulk_outputs);


    // load all coefficients into registers. if const N is shorter than the
    // full set of 8 below (32 coeffs), some of these will disappear in the
    // compile
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
    load_coefficient!(h1, 4, 4);
    load_coefficient!(h2, 8, 8);
    load_coefficient!(h3, 12, 12);
    load_coefficient!(h4, 16, 16);
    load_coefficient!(h5, 20, 20);
    load_coefficient!(h6, 24, 24);
    load_coefficient!(h7, 28, 28);

    let xp = x.as_ptr();
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        // compute sums of 4 simultaneous sliding windows of inputs

        // accumulators (initialize with a0 = x[0] * h0, ... , a3 = x[3] * h0)
        let mut a0 = f32x4::from_array(*(xp.add(i) as *const [f32; 4])) * h0;
        let mut a1 = f32x4::from_array(*(xp.add(i + 1) as *const [f32; 4])) * h0;
        let mut a2 = f32x4::from_array(*(xp.add(i + 2) as *const [f32; 4])) * h0;
        let mut a3 = f32x4::from_array(*(xp.add(i + 3) as *const [f32; 4])) * h0;

        macro_rules! accumulate {
            ($threshold:literal, $offset:literal, $coeff:ident) => {
                if N > $threshold {
                    a0 += f32x4::from_array(
                        *(xp.add(i + $offset) as *const [f32; 4]),
                    ) * $coeff;
                    a1 += f32x4::from_array(
                        *(xp.add(i + $offset + 1) as *const [f32; 4]),
                    ) * $coeff;
                    a2 += f32x4::from_array(
                        *(xp.add(i + $offset + 2) as *const [f32; 4]),
                    ) * $coeff;
                    a3 += f32x4::from_array(
                        *(xp.add(i + $offset + 3) as *const [f32; 4]),
                    ) * $coeff;
                }
            };
        }

        // invoke the macro above as needed to create the full dot product
        // a0 += x[1] * h1, ... , a3 += x[4] * h1
        // then a0 += x[2] * h2, ... , a3 += x[5] * h2, etc
        accumulate!(4, 4, h1);
        accumulate!(8, 8, h2);
        accumulate!(12, 12, h3);
        accumulate!(16, 16, h4);
        accumulate!(20, 20, h5);
        accumulate!(24, 24, h6);
        accumulate!(28, 28, h7);

        // reduce the horizontal sums (f32x4 -> f32x1), store to y
        *yp.add(i) = reduce_sum_sse_f32x4(a0);
        *yp.add(i + 1) = reduce_sum_sse_f32x4(a1);
        *yp.add(i + 2) = reduce_sum_sse_f32x4(a2);
        *yp.add(i + 3) = reduce_sum_sse_f32x4(a3);
        i += 4;
    }

    bulk_outputs
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_rrr_block_avx512_register<const N: usize>(
    x: &[f32],
    h: &[f32],
    y: &mut [f32],
) -> usize {
    // this is just a larger version of `dotprod_rrr_block_f32x4`
    // instead of 32 (8x4) coefficients in register, we get 256 (16x16)
    // but we still handle 4 outputs at a time

    debug_assert_eq!(h.len(), N);
    let safe_outputs = x.len().saturating_sub(N - 1).min(y.len());
    let bulk_outputs = safe_outputs / 4 * 4;
    if bulk_outputs == 0 {
        return 0;
    }

    std::hint::assert_unchecked(h.len() == N);
    std::hint::assert_unchecked(x.len() >= bulk_outputs + N - 1);
    std::hint::assert_unchecked(y.len() >= bulk_outputs);

    let hp = h.as_ptr();
    macro_rules! load_coefficient {
        ($name:ident, $threshold:literal, $offset:literal) => {
            let $name = if N > $threshold {
                f32x16::from_array(*(hp.add($offset) as *const [f32; 16]))
            } else {
                f32x16::splat(0.0)
            };
        };
    }
    load_coefficient!(h0, 0, 0);
    load_coefficient!(h1, 16, 16);
    load_coefficient!(h2, 32, 32);
    load_coefficient!(h3, 48, 48);
    load_coefficient!(h4, 64, 64);
    load_coefficient!(h5, 80, 80);
    load_coefficient!(h6, 96, 96);
    load_coefficient!(h7, 112, 112);
    load_coefficient!(h8, 128, 128);
    load_coefficient!(h9, 144, 144);
    load_coefficient!(h10, 160, 160);
    load_coefficient!(h11, 176, 176);
    load_coefficient!(h12, 192, 192);
    load_coefficient!(h13, 208, 208);
    load_coefficient!(h14, 224, 224);
    load_coefficient!(h15, 240, 240);

    let xp = x.as_ptr();
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        let mut a0 = f32x16::from_array(*(xp.add(i) as *const [f32; 16])) * h0;
        let mut a1 = f32x16::from_array(*(xp.add(i + 1) as *const [f32; 16])) * h0;
        let mut a2 = f32x16::from_array(*(xp.add(i + 2) as *const [f32; 16])) * h0;
        let mut a3 = f32x16::from_array(*(xp.add(i + 3) as *const [f32; 16])) * h0;

        macro_rules! accumulate {
            ($threshold:literal, $offset:literal, $coeff:ident) => {
                if N > $threshold {
                    a0 += f32x16::from_array(
                        *(xp.add(i + $offset) as *const [f32; 16]),
                    ) * $coeff;
                    a1 += f32x16::from_array(
                        *(xp.add(i + $offset + 1) as *const [f32; 16]),
                    ) * $coeff;
                    a2 += f32x16::from_array(
                        *(xp.add(i + $offset + 2) as *const [f32; 16]),
                    ) * $coeff;
                    a3 += f32x16::from_array(
                        *(xp.add(i + $offset + 3) as *const [f32; 16]),
                    ) * $coeff;
                }
            };
        }

        accumulate!(16, 16, h1);
        accumulate!(32, 32, h2);
        accumulate!(48, 48, h3);
        accumulate!(64, 64, h4);
        accumulate!(80, 80, h5);
        accumulate!(96, 96, h6);
        accumulate!(112, 112, h7);
        accumulate!(128, 128, h8);
        accumulate!(144, 144, h9);
        accumulate!(160, 160, h10);
        accumulate!(176, 176, h11);
        accumulate!(192, 192, h12);
        accumulate!(208, 208, h13);
        accumulate!(224, 224, h14);
        accumulate!(240, 240, h15);

        *yp.add(i) = reduce_sum_avx512_f32x16(a0);
        *yp.add(i + 1) = reduce_sum_avx512_f32x16(a1);
        *yp.add(i + 2) = reduce_sum_avx512_f32x16(a2);
        *yp.add(i + 3) = reduce_sum_avx512_f32x16(a3);
        i += 4;
    }

    bulk_outputs
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dotprod_rrr_block_avx512_memory(
    x: &[f32],
    h: &[f32],
    y: &mut [f32],
) -> usize {
    // this kernel loads the coefficients from memory every 4 inputs
    // it's not const on length, so one size fits all
    // less efficient, but it also won't spill registers

    let n = h.len();
    debug_assert_eq!(n % 16, 0);
    let safe_outputs = x.len().saturating_sub(n - 1).min(y.len());
    let bulk_outputs = safe_outputs / 4 * 4;
    if bulk_outputs == 0 {
        return 0;
    }

    std::hint::assert_unchecked(n >= 16 && n % 16 == 0);
    std::hint::assert_unchecked(x.len() >= bulk_outputs + n - 1);
    std::hint::assert_unchecked(y.len() >= bulk_outputs);

    let hp = h.as_ptr();
    let xp = x.as_ptr();
    let yp = y.as_mut_ptr();
    let mut i = 0;
    while i < bulk_outputs {
        let h = f32x16::from_array(*(hp as *const [f32; 16]));
        let mut a0 = f32x16::from_array(*(xp.add(i) as *const [f32; 16])) * h;
        let mut a1 = f32x16::from_array(*(xp.add(i + 1) as *const [f32; 16])) * h;
        let mut a2 = f32x16::from_array(*(xp.add(i + 2) as *const [f32; 16])) * h;
        let mut a3 = f32x16::from_array(*(xp.add(i + 3) as *const [f32; 16])) * h;

        macro_rules! accumulate {
            ($offset:expr) => {{
                let offset = $offset;
                let h = f32x16::from_array(*(hp.add(offset) as *const [f32; 16]));
                a0 += f32x16::from_array(
                    *(xp.add(i + offset) as *const [f32; 16]),
                ) * h;
                a1 += f32x16::from_array(
                    *(xp.add(i + offset + 1) as *const [f32; 16]),
                ) * h;
                a2 += f32x16::from_array(
                    *(xp.add(i + offset + 2) as *const [f32; 16]),
                ) * h;
                a3 += f32x16::from_array(
                    *(xp.add(i + offset + 3) as *const [f32; 16]),
                ) * h;
            }};
        }

        let mut j = 16;
        // this will hopefully convince llvm to give us a 4-wide version with a 1-wide tail
        while j + 48 < n {
            accumulate!(j);
            accumulate!(j + 16);
            accumulate!(j + 32);
            accumulate!(j + 48);
            j += 64;
        }
        while j < n {
            accumulate!(j);
            j += 16;
        }

        *yp.add(i) = reduce_sum_avx512_f32x16(a0);
        *yp.add(i + 1) = reduce_sum_avx512_f32x16(a1);
        *yp.add(i + 2) = reduce_sum_avx512_f32x16(a2);
        *yp.add(i + 3) = reduce_sum_avx512_f32x16(a3);
        i += 4;
    }

    bulk_outputs
}