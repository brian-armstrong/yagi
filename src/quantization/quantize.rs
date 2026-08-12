//
// quantize.rs
//
// uniform quantizer: 'analog' signal in [-1, 1]
//
// samples are coded sign-magnitude. the most significant of the num_bits bits
// holds the sign and the remaining bits hold the magnitude. note this leaves
// two codes for zero (+0 and -0) and no code for exactly zero on output

use crate::error::{Error, Result};

// we use an f32 type, so we really only have about 24 usable bits here, but the math still
//  works up to 32 bits.
const QUANTIZER_MAX_BITS: usize = 32;

/// quantize a sample: analog to digital converter
///   x        : input sample in [-1,1]
///   num_bits : number of bits per sample, in [0,32]
///
/// values outside [-1,1] are clipped to the largest representable magnitude.
pub fn quantize_adc(x: f32, num_bits: usize) -> Result<u32> {
    if num_bits > QUANTIZER_MAX_BITS {
        return Err(Error::Range(
            "quantize_adc(), maximum bits exceeded".into(),
        ));
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

/// reconstruct a sample: digital to analog converter
///   s        : quantized sample
///   num_bits : number of bits per sample, in [0,32]
///
/// bits above num_bits are ignored.
pub fn quantize_dac(s: u32, num_bits: usize) -> Result<f32> {
    if num_bits > QUANTIZER_MAX_BITS {
        return Err(Error::Range(
            "quantize_dac(), maximum bits exceeded".into(),
        ));
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


