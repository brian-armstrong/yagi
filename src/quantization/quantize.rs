//! Uniform sign-magnitude quantization for real samples near `[-1, 1]`.
//!
//! The most significant code bit is the sign. Remaining bits encode magnitude.
//! The ADC clips magnitudes of one or more to the largest code. The DAC
//! reconstructs each code at the midpoint of its bin, so neither zero code
//! reconstructs exactly to zero. A round trip is within half a bin of the
//! input for magnitudes below one.
//!
//! `f32` has 24 bits of precision, so widths above about 24 bits do not give
//! distinct amplitude steps throughout the range.

use crate::error::{Error, Result};

// we use an f32 type, so we really only have about 24 usable bits here, but the math still
//  works up to 32 bits.
const QUANTIZER_MAX_BITS: usize = 32;

/// Quantizes one real sample to a sign-magnitude code.
///
/// `num_bits` ranges from zero through 32. A width of zero always returns
/// zero. Otherwise bit `num_bits - 1` is the sign, and the bits below it hold
/// `floor(|x| * 2^(num_bits - 1))`. Magnitudes of one or more clip to the
/// largest magnitude code. Bits above the width are zero. A negative value
/// that rounds to zero magnitude keeps its sign bit, while negative zero
/// encodes as positive zero.
///
/// # Errors
///
/// Returns an error when `num_bits` exceeds 32.
pub fn quantize_adc(x: f32, num_bits: usize) -> Result<u32> {
    if num_bits > QUANTIZER_MAX_BITS {
        return Err(Error::Range("quantize_adc(), maximum bits exceeded".into()));
    }

    if num_bits == 0 {
        return Ok(0);
    }

    let levels = 1u32 << (num_bits - 1); // magnitude levels per side
    let sign_bit = levels;

    // scale
    let neg = x < 0.0;
    // clip in float, then narrow. this is safer than clip after narrow.
    let scaled = (x.abs() * levels as f32).floor();
    let mut r = if scaled >= levels as f32 {
        // clip
        levels - 1
    } else {
        scaled as u32
    };

    // if negative set MSB to 1
    if neg {
        r |= sign_bit;
    }

    Ok(r)
}

/// Reconstructs a real sample from a sign-magnitude code.
///
/// `num_bits` ranges from zero through 32. A width of zero returns zero, and
/// bits above the width are ignored. A magnitude code `m` reconstructs to
/// `(m + 0.5) / 2^(num_bits - 1)`, the midpoint of its bin, so a zero
/// magnitude gives plus or minus half a bin.
///
/// At wide bit widths, `f32` rounding can merge adjacent values or move a
/// result away from the midpoint. The largest magnitude code can then
/// reconstruct to exactly one.
///
/// # Errors
///
/// Returns an error when `num_bits` exceeds 32.
pub fn quantize_dac(s: u32, num_bits: usize) -> Result<f32> {
    if num_bits > QUANTIZER_MAX_BITS {
        return Err(Error::Range("quantize_dac(), maximum bits exceeded".into()));
    }

    if num_bits == 0 {
        return Ok(0.0);
    }

    let levels = 1u32 << (num_bits - 1); // magnitude levels per side
    let sign_bit = levels;

    // reconstruct at the center of the step
    let r = ((s & (levels - 1)) as f32 + 0.5) / levels as f32;

    // check MSB, return negative if 1
    Ok(if s & sign_bit != 0 { -r } else { r })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_quantize_float_n8)]
    fn test_quantize_float_n8() {
        let mut x = -1.0f32;
        let num_steps = 30;
        let num_bits = 8;

        let dx = 2.0 / num_steps as f32;
        let tol = 1.0 / (1u32 << num_bits) as f32;

        for _ in 0..num_steps {
            let q = quantize_adc(x, num_bits).unwrap();

            // ensure only num_bits written to value q
            assert_eq!(q >> num_bits, 0);

            let x_hat = quantize_dac(q, num_bits).unwrap();

            // ensure original value is recovered within tolerance
            assert_abs_diff_eq!(x, x_hat, epsilon = tol);

            x += dx;
            x = if x > 1.0 { 1.0 } else { x };
        }
    }

    #[test]
    fn test_quantize_bit_range() {
        for num_bits in [33usize, 64, 100] {
            assert!(quantize_adc(0.5, num_bits).is_err(), "adc {}", num_bits);
            assert!(quantize_dac(1, num_bits).is_err(), "dac {}", num_bits);
        }

        // zero bits carries no information, but is not an error
        assert_eq!(quantize_adc(0.5, 0).unwrap(), 0);
        assert_eq!(quantize_dac(1, 0).unwrap(), 0.0);

        // the widest supported width round-trips
        let q = quantize_adc(0.5, 32).unwrap();
        assert_abs_diff_eq!(quantize_dac(q, 32).unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_quantize_sign_magnitude() {
        let num_bits = 8;
        let pos = quantize_adc(1.0, num_bits).unwrap();
        let neg = quantize_adc(-1.0, num_bits).unwrap();

        assert_eq!(pos, 127);
        assert_eq!(neg, 255);
        assert_eq!(neg, pos | 0x80);

        // negative zero is not negative, so it codes as +0
        assert_eq!(quantize_adc(-0.0, num_bits).unwrap(), 0);
        assert_eq!(quantize_adc(0.0, num_bits).unwrap(), 0);
    }

    #[test]
    fn test_quantize_clips_out_of_domain() {
        let num_bits = 8;

        // inputs beyond [-1,1] saturate rather than wrapping around
        for x in [1.0f32, 1.5, 2.0, 1e9] {
            assert_eq!(quantize_adc(x, num_bits).unwrap(), 127, "x={}", x);
            assert_eq!(quantize_adc(-x, num_bits).unwrap(), 255, "x={}", -x);
        }
    }

    #[test]
    fn test_quantize_clips_at_full_width() {
        let sign_bit = 1u32 << 31;
        let full = sign_bit - 1;
        for x in [1.0f32, 2.0, 1e9, f32::MAX] {
            assert_eq!(quantize_adc(x, 32).unwrap(), full, "x={}", x);
            assert_eq!(quantize_adc(-x, 32).unwrap(), full | sign_bit, "x={}", -x);
        }

        // the largest input must not decode smaller than a mid-scale one
        let big = quantize_dac(quantize_adc(2.0, 32).unwrap(), 32).unwrap();
        let mid = quantize_dac(quantize_adc(0.5, 32).unwrap(), 32).unwrap();
        assert!(big > mid, "saturating input decoded as {} vs {}", big, mid);
    }

    #[test]
    fn test_quantize_dac_ignores_high_bits() {
        // bits above num_bits are not part of the code word
        let num_bits = 8;
        for extra in [0u32, 0x100, 0xFF00, 0xFFFF_FF00] {
            assert_abs_diff_eq!(
                quantize_dac(0x7F | extra, num_bits).unwrap(),
                quantize_dac(0x7F, num_bits).unwrap(),
                epsilon = 0.0
            );
        }
    }
}
