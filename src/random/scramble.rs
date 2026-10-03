//! Byte and soft-bit scrambling with a fixed repeating mask.
//!
//! The mask is deterministic and provides no secrecy.

// Define the scramble masks as constants.
const SCRAMBLE_MASK0: u8 = 0xB4;
const SCRAMBLE_MASK1: u8 = 0x6A;
const SCRAMBLE_MASK2: u8 = 0x8B;
const SCRAMBLE_MASK3: u8 = 0xC5;

/// Scrambles bytes in place with a fixed four-byte mask.
///
/// The mask bytes are `B4 6A 8B C5` in hexadecimal, repeating from the first
/// byte of each call. Each input byte is XORed with its mask byte.
///
/// Applying [`unscramble_data`] to the result recovers the original bytes.
pub fn scramble_data(x: &mut [u8]) {
    // t = 4*(floor(_n/4))
    let t = x.len() & !3;

    // apply static masks
    for chunk in x[..t].as_chunks_mut::<4>().0 {
        chunk[0] ^= SCRAMBLE_MASK0;
        chunk[1] ^= SCRAMBLE_MASK1;
        chunk[2] ^= SCRAMBLE_MASK2;
        chunk[3] ^= SCRAMBLE_MASK3;
    }

    // clean up remainder of elements
    for (i, byte) in x[t..].iter_mut().enumerate() {
        *byte ^= match i {
            0 => SCRAMBLE_MASK0,
            1 => SCRAMBLE_MASK1,
            2 => SCRAMBLE_MASK2,
            3 => SCRAMBLE_MASK3,
            _ => unreachable!(),
        };
    }
}

/// Recovers bytes scrambled by [`scramble_data`].
///
/// Scrambling is an XOR, so this applies the same mask again.
pub fn unscramble_data(x: &mut [u8]) {
    // for now apply simple static mask (re-run scramble)
    scramble_data(x);
}

/// Recovers soft bits from bytes scrambled by [`scramble_data`].
///
/// `x` contains one soft decision per bit in most-significant-bit-first order.
/// Values near 0 favor a zero bit and values near 255 favor a one bit. Each
/// value whose mask bit is one becomes `255 - value`. Eight soft bits match
/// one byte of the mask, so the mask repeats every 32 values.
///
/// Only complete groups of eight soft bits are processed. A trailing group
/// shorter than eight elements is left unchanged.
pub fn unscramble_data_soft(x: &mut [u8]) {
    for (i, chunk) in x.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let mask = match i % 4 {
            0 => SCRAMBLE_MASK0,
            1 => SCRAMBLE_MASK1,
            2 => SCRAMBLE_MASK2,
            3 => SCRAMBLE_MASK3,
            _ => unreachable!(),
        };

        for (j, byte) in chunk.iter_mut().enumerate() {
            if (mask >> (7 - j)) & 0x01 != 0 {
                *byte = 255 - *byte;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    // compute basic entropy metric
    fn scramble_test_entropy(x: &[u8]) -> f32 {
        let num_ones: u32 = x.iter().map(|&byte| byte.count_ones()).sum();
        let p1 = (num_ones as f32) / (8 * x.len()) as f32 + 1e-12;
        let p0 = 1.0 - p1;
        -(p0 * p0.log2() + p1 * p1.log2())
    }

    // helper function to keep code base small
    fn scramble_test(n: usize) {
        let x = vec![0u8; n];
        let mut y = vec![0u8; n];
        let mut z = vec![0u8; n];

        // scramble input
        y.copy_from_slice(&x);
        scramble_data(&mut y);

        // unscramble result
        z.copy_from_slice(&y);
        unscramble_data(&mut z);

        // ensure data are equivalent
        assert_eq!(x, z);

        // compute entropy metric
        let h = scramble_test_entropy(&y);
        assert!(h > 0.8);
    }

    // test unscrambling of soft bits (helper function to keep code base small)
    fn scramble_soft_test(n: usize) {
        let mut msg_org = vec![0u8; n];
        let mut msg_enc = vec![0u8; n];
        let mut msg_soft = vec![0u8; 8 * n];
        let mut msg_dec = vec![0u8; n];

        // initialize data array (random)
        for i in 0..n {
            msg_org[i] = rand::random::<u8>();
        }

        // scramble input
        msg_enc.copy_from_slice(&msg_org);
        scramble_data(&mut msg_enc);

        // convert to soft bits
        // TODO use liquid-dsp unpack_soft_bits when it exists
        for i in 0..n {
            for j in 0..8 {
                msg_soft[8 * i + j] = if (msg_enc[i] >> (7 - j)) & 0x01 != 0 { 255 } else { 0 };
            }
        }

        // unscramble result
        unscramble_data_soft(&mut msg_soft);

        // unpack soft bits
        // TODO use liquid-dsp pack_soft_bits when it exists
        for i in 0..n {
            let mut sym_out = 0u8;
            for j in 0..8 {
                sym_out |= if msg_soft[8 * i + j] > 127 { 1 } else { 0 } << (7 - j);
            }
            msg_dec[i] = sym_out;
        }

        // ensure data are equivalent
        assert_eq!(msg_org, msg_dec);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_n16)]
    fn test_scramble_n16() {
        scramble_test(16);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_n64)]
    fn test_scramble_n64() {
        scramble_test(64);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_n256)]
    fn test_scramble_n256() {
        scramble_test(256);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_n11)]
    fn test_scramble_n11() {
        scramble_test(11);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_n33)]
    fn test_scramble_n33() {
        scramble_test(33);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_n277)]
    fn test_scramble_n277() {
        scramble_test(277);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_soft_n16)]
    fn test_scramble_soft_n16() {
        scramble_soft_test(16);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_soft_n64)]
    fn test_scramble_soft_n64() {
        scramble_soft_test(64);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_soft_n256)]
    fn test_scramble_soft_n256() {
        scramble_soft_test(256);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_soft_n11)]
    fn test_scramble_soft_n11() {
        scramble_soft_test(11);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_soft_n33)]
    fn test_scramble_soft_n33() {
        scramble_soft_test(33);
    }

    #[test]
    #[autotest_annotate(autotest_scramble_soft_n277)]
    fn test_scramble_soft_n277() {
        scramble_soft_test(277);
    }
}
