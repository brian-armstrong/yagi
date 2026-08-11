//
// 2/3-rate (12,8) Hamming code
//
//  bit position    1   2   3   4   5   6   7   8   9   10  11  12
//  encoded bits    P1  P2  1   P4  2   3   4   P8  5   6   7   8
//
//  parity bit  P1  x   .   x   .   x   .   x   .   x   .   x   .
//  coveratge   P2  .   x   x   .   .   x   x   .   .   x   x   .
//              P4  .   .   .   x   x   x   x   .   .   .   .   x
//              P8  .   .   .   .   .   .   .   x   x   x   x   x

use crate::utility::bits::bdotprod;

// parity bit coverage mask for encoder (collapsed version of figure
// above, stripping out parity bits P1, P2, P4, P8 and only including
// data bits 1:8)
//
// bit position     3   5   6   7   9   10  11  12
//
//  parity bit  P1  x   x   .   x   x   .   x   .   =   1101 1010
//  coverage    P2  x   .   x   x   .   x   x   .   =   1011 0110
//              P4  .   x   x   x   .   .   .   x   =   0111 0001
//              P8  .   .   .   .   x   x   x   x   =   0000 1111
//
// encoding reads ENC_GENTAB; these only back the test that checks it
#[cfg(test)]
const M1: u32 = 0xda; // 1101 1010
#[cfg(test)]
const M2: u32 = 0xb6; // 1011 0110
#[cfg(test)]
const M4: u32 = 0x71; // 0111 0001
#[cfg(test)]
const M8: u32 = 0x0f; // 0000 1111

// parity bit coverage mask for decoder; used to compute syndromes
// for decoding a received message (see first figure, above).
const S1: u32 = 0x0aaa; // .... 1010 1010 1010
const S2: u32 = 0x0666; // .... 0110 0110 0110
const S4: u32 = 0x01e1; // .... 0001 1110 0001
const S8: u32 = 0x001f; // .... 0000 0001 1111

// encoder look-up table
const ENC_GENTAB: [u16; 256] = [
    0x0000, 0x0111, 0x0c12, 0x0d03, 0x0414, 0x0505, 0x0806, 0x0917,
    0x0818, 0x0909, 0x040a, 0x051b, 0x0c0c, 0x0d1d, 0x001e, 0x010f,
    0x0d20, 0x0c31, 0x0132, 0x0023, 0x0934, 0x0825, 0x0526, 0x0437,
    0x0538, 0x0429, 0x092a, 0x083b, 0x012c, 0x003d, 0x0d3e, 0x0c2f,
    0x0540, 0x0451, 0x0952, 0x0843, 0x0154, 0x0045, 0x0d46, 0x0c57,
    0x0d58, 0x0c49, 0x014a, 0x005b, 0x094c, 0x085d, 0x055e, 0x044f,
    0x0860, 0x0971, 0x0472, 0x0563, 0x0c74, 0x0d65, 0x0066, 0x0177,
    0x0078, 0x0169, 0x0c6a, 0x0d7b, 0x046c, 0x057d, 0x087e, 0x096f,
    0x0980, 0x0891, 0x0592, 0x0483, 0x0d94, 0x0c85, 0x0186, 0x0097,
    0x0198, 0x0089, 0x0d8a, 0x0c9b, 0x058c, 0x049d, 0x099e, 0x088f,
    0x04a0, 0x05b1, 0x08b2, 0x09a3, 0x00b4, 0x01a5, 0x0ca6, 0x0db7,
    0x0cb8, 0x0da9, 0x00aa, 0x01bb, 0x08ac, 0x09bd, 0x04be, 0x05af,
    0x0cc0, 0x0dd1, 0x00d2, 0x01c3, 0x08d4, 0x09c5, 0x04c6, 0x05d7,
    0x04d8, 0x05c9, 0x08ca, 0x09db, 0x00cc, 0x01dd, 0x0cde, 0x0dcf,
    0x01e0, 0x00f1, 0x0df2, 0x0ce3, 0x05f4, 0x04e5, 0x09e6, 0x08f7,
    0x09f8, 0x08e9, 0x05ea, 0x04fb, 0x0dec, 0x0cfd, 0x01fe, 0x00ef,
    0x0e00, 0x0f11, 0x0212, 0x0303, 0x0a14, 0x0b05, 0x0606, 0x0717,
    0x0618, 0x0709, 0x0a0a, 0x0b1b, 0x020c, 0x031d, 0x0e1e, 0x0f0f,
    0x0320, 0x0231, 0x0f32, 0x0e23, 0x0734, 0x0625, 0x0b26, 0x0a37,
    0x0b38, 0x0a29, 0x072a, 0x063b, 0x0f2c, 0x0e3d, 0x033e, 0x022f,
    0x0b40, 0x0a51, 0x0752, 0x0643, 0x0f54, 0x0e45, 0x0346, 0x0257,
    0x0358, 0x0249, 0x0f4a, 0x0e5b, 0x074c, 0x065d, 0x0b5e, 0x0a4f,
    0x0660, 0x0771, 0x0a72, 0x0b63, 0x0274, 0x0365, 0x0e66, 0x0f77,
    0x0e78, 0x0f69, 0x026a, 0x037b, 0x0a6c, 0x0b7d, 0x067e, 0x076f,
    0x0780, 0x0691, 0x0b92, 0x0a83, 0x0394, 0x0285, 0x0f86, 0x0e97,
    0x0f98, 0x0e89, 0x038a, 0x029b, 0x0b8c, 0x0a9d, 0x079e, 0x068f,
    0x0aa0, 0x0bb1, 0x06b2, 0x07a3, 0x0eb4, 0x0fa5, 0x02a6, 0x03b7,
    0x02b8, 0x03a9, 0x0eaa, 0x0fbb, 0x06ac, 0x07bd, 0x0abe, 0x0baf,
    0x02c0, 0x03d1, 0x0ed2, 0x0fc3, 0x06d4, 0x07c5, 0x0ac6, 0x0bd7,
    0x0ad8, 0x0bc9, 0x06ca, 0x07db, 0x0ecc, 0x0fdd, 0x02de, 0x03cf,
    0x0fe0, 0x0ef1, 0x03f2, 0x02e3, 0x0bf4, 0x0ae5, 0x07e6, 0x06f7,
    0x07f8, 0x06e9, 0x0bea, 0x0afb, 0x03ec, 0x02fd, 0x0ffe, 0x0eef,
];

// the definition ENC_GENTAB was generated from
#[cfg(test)]
fn encode_symbol_direct(sym_dec: u8) -> u16 {
    let sym = sym_dec as u32;

    // compute parity bits
    let p1 = bdotprod(sym, M1);
    let p2 = bdotprod(sym, M2);
    let p4 = bdotprod(sym, M4);
    let p8 = bdotprod(sym, M8);

    // encode symbol by inserting parity bits with data bits to
    // make a 12-bit symbol
    let sym_enc = ((sym & 0x000f) << 0)
        | ((sym & 0x0070) << 1)
        | ((sym & 0x0080) << 2)
        | (p1 << 11)
        | (p2 << 10)
        | (p4 << 8)
        | (p8 << 4);

    sym_enc as u16
}

fn encode_symbol(sym_dec: u8) -> u16 {
    // u8 cannot exceed the 8-bit symbol width, so no bound check is needed
    ENC_GENTAB[sym_dec as usize]
}

// pub(crate) so the config autotest can check its bound
pub(crate) fn decode_symbol(mut sym_enc: u16) -> u8 {
    assert!(sym_enc < (1 << 12), "input symbol too large");

    // compute syndrome bits
    let s1 = bdotprod(sym_enc as u32, S1);
    let s2 = bdotprod(sym_enc as u32, S2);
    let s4 = bdotprod(sym_enc as u32, S4);
    let s8 = bdotprod(sym_enc as u32, S8);

    // index
    let z = (s8 << 3) | (s4 << 2) | (s2 << 1) | s1;

    // flip bit at this position; z > 12 means there are likely too many
    // errors to correct, so just pass without trying to do anything
    if z != 0 && z <= 12 {
        sym_enc ^= 1 << (12 - z);
    }

    // strip data bits (x) from encoded symbol with parity bits (.)
    //      symbol:  [..x. xxx. xxxx]
    //                0000 0000 1111     >  0x000f
    //                0000 1110 0000     >  0x00e0
    //                0010 0000 0000     >  0x0200
    let sym_dec = ((sym_enc & 0x000f) |
                   ((sym_enc & 0x00e0) >> 1) |
                   ((sym_enc & 0x0200) >> 2)) as u8;

    sym_dec
}

/// encode block of data using Hamming(12,8) encoder
///
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
///  msg_enc        :   encoded message
pub fn hamming128_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    let dec_msg_len = msg_dec.len();

    // determine if input length is odd
    let r = dec_msg_len % 2;
    let mut j = 0usize;

    let mut i = 0;
    while i < dec_msg_len - r {
        // strip two input bytes
        let s0 = msg_dec[i];
        let s1 = msg_dec[i + 1];

        // encode each byte into 12-bit symbols
        let m0 = encode_symbol(s0);
        let m1 = encode_symbol(s1);

        // append both 12-bit symbols to output (three 8-bit bytes),
        // retaining order of bits in output
        msg_enc[j] = (m0 >> 4) as u8;
        msg_enc[j + 1] = ((m0 << 4) as u8) | ((m1 >> 8) as u8);
        msg_enc[j + 2] = m1 as u8;

        i += 2;
        j += 3;
    }

    // if input length is even, encode last symbol by itself
    if r != 0 {
        // strip last input symbol, encode into 12-bit symbol
        let s0 = msg_dec[dec_msg_len - 1];
        let m0 = encode_symbol(s0);

        // append to output
        msg_enc[j] = ((m0 & 0x0ff0) >> 4) as u8;
        msg_enc[j + 1] = ((m0 & 0x000f) << 4) as u8;
    }
}

/// decode block of data using Hamming(12,8) decoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
pub fn hamming128_decode(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    let r = dec_msg_len % 2;
    let mut i = 0usize;
    let mut j = 0usize;

    while i < dec_msg_len - r {
        // strip three input symbols
        let r0 = msg_enc[j];
        let r1 = msg_enc[j + 1];
        let r2 = msg_enc[j + 2];

        // combine three 8-bit symbols into two 12-bit symbols
        let m0 = ((r0 as u16) << 4) | ((r1 as u16) >> 4);
        let m1 = (((r1 as u16) & 0x0f) << 8) | (r2 as u16);

        // decode each symbol into an 8-bit byte
        msg_dec[i] = decode_symbol(m0);
        msg_dec[i + 1] = decode_symbol(m1);

        i += 2;
        j += 3;
    }

    // if input length is even, decode last symbol by itself
    if r != 0 {
        // strip last two input bytes (last byte should only contain
        // for bits)
        let r0 = msg_enc[j];
        let r1 = msg_enc[j + 1];

        // pack into 12-bit symbol
        let m0 = ((r0 as u16) << 4) | ((r1 as u16) >> 4);

        // decode symbol into an 8-bit byte
        msg_dec[i] = decode_symbol(m0);
    }
}

// compute distance metric between an encoded symbol and received soft bits
fn compute_distance(c: u16, soft_bits: &[u8]) -> u32 {
    let mut d = 0u32;
    d += if c & 0x0800 != 0 { 255 - soft_bits[0] as u32 } else { soft_bits[0] as u32 };
    d += if c & 0x0400 != 0 { 255 - soft_bits[1] as u32 } else { soft_bits[1] as u32 };
    d += if c & 0x0200 != 0 { 255 - soft_bits[2] as u32 } else { soft_bits[2] as u32 };
    d += if c & 0x0100 != 0 { 255 - soft_bits[3] as u32 } else { soft_bits[3] as u32 };
    d += if c & 0x0080 != 0 { 255 - soft_bits[4] as u32 } else { soft_bits[4] as u32 };
    d += if c & 0x0040 != 0 { 255 - soft_bits[5] as u32 } else { soft_bits[5] as u32 };
    d += if c & 0x0020 != 0 { 255 - soft_bits[6] as u32 } else { soft_bits[6] as u32 };
    d += if c & 0x0010 != 0 { 255 - soft_bits[7] as u32 } else { soft_bits[7] as u32 };
    d += if c & 0x0008 != 0 { 255 - soft_bits[8] as u32 } else { soft_bits[8] as u32 };
    d += if c & 0x0004 != 0 { 255 - soft_bits[9] as u32 } else { soft_bits[9] as u32 };
    d += if c & 0x0002 != 0 { 255 - soft_bits[10] as u32 } else { soft_bits[10] as u32 };
    d += if c & 0x0001 != 0 { 255 - soft_bits[11] as u32 } else { soft_bits[11] as u32 };
    d
}

// soft decoding of one symbol
//
// liquid also offers an approximation searching only 17 precomputed neighbors
// of the hard decision; it is not ported, so this keeps the full coding gain
fn soft_decode_symbol(soft_bits: &[u8]) -> u8 {
    // find symbol with minimum distance from all 2^8 possible
    let mut dmin = u32::MAX;
    let mut s_hat = 0u8;

    for s in 0u8..=255 {
        let d = compute_distance(encode_symbol(s), soft_bits);
        if d < dmin {
            s_hat = s;
            dmin = d;
        }
    }

    s_hat
}

/// decode block of data using Hamming(12,8) soft decoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message [size: 8*enc_msg_len x 1]
///  msg_dec        :   decoded message [size: dec_msg_len x 1]
pub fn hamming128_decode_soft(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    let mut k = 0usize;

    for i in 0..dec_msg_len {
        msg_dec[i] = soft_decode_symbol(&msg_enc[k..]);
        k += 12;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_hamming128_codec)]
    fn test_hamming128_codec() {
        let n = 8; // input symbol size (bits)
        let k = 12; // encoded symbol size (bits)

        let mut rng = rand::thread_rng();

        for i in 0..k {
            // generate symbol
            let sym_org = (rng.gen::<u32>() % (1 << n)) as u8;

            // encoded symbol
            let sym_enc = encode_symbol(sym_org);

            // received symbol, with bit i corrupted
            let sym_rec = sym_enc ^ (1 << (k - i - 1));

            // decoded symbol
            let sym_dec = decode_symbol(sym_rec);

            assert_eq!(sym_org, sym_dec);
        }
    }

    #[test]
    #[autotest_annotate(autotest_hamming128_codec_soft)]
    fn test_hamming128_codec_soft() {
        // generate each of the 2^8=256 symbols, encode, and decode
        // using soft decoding algorithm
        for s in 0u8..=255 {
            // encode using internal method
            let c = encode_symbol(s);

            // expand soft bits
            let mut c_soft = [0u8; 12];
            for (i, bit) in c_soft.iter_mut().enumerate() {
                *bit = if c & (0x0800 >> i) != 0 { 255 } else { 0 };
            }

            // decode using internal soft decoding method
            let s_hat = soft_decode_symbol(&c_soft);

            assert_eq!(s, s_hat);
        }
    }

    #[test]
    fn test_hamming128_soft_decode_under_noise() {
        let mut rng = rand::thread_rng();

        for _ in 0..2000 {
            let s = rng.gen::<u8>();
            let c = encode_symbol(s);

            // expand to soft bits with additive noise
            let mut c_soft = [0u8; 12];
            for (i, bit) in c_soft.iter_mut().enumerate() {
                let nominal: i32 = if c & (0x0800 >> i) != 0 { 255 } else { 0 };
                let noise = rng.gen_range(-110i32..=110);
                *bit = (nominal + noise).clamp(0, 255) as u8;
            }

            assert_eq!(
                soft_decode_symbol(&c_soft),
                s,
                "soft decode failed to recover symbol {:#04x} under noise",
                s
            );
        }
    }

    #[test]
    fn test_hamming128_soft_decode_single_flip() {
        for s in 0u8..=255 {
            let c = encode_symbol(s);

            for flip in 0..12 {
                let mut c_soft = [0u8; 12];
                for (i, bit) in c_soft.iter_mut().enumerate() {
                    let set = (c & (0x0800 >> i) != 0) ^ (i == flip);
                    *bit = if set { 255 } else { 0 };
                }

                assert_eq!(
                    soft_decode_symbol(&c_soft),
                    s,
                    "soft decode failed for symbol {:#04x} with bit {} flipped",
                    s,
                    flip
                );
            }
        }
    }

    #[test]
    fn test_hamming128_gentab_matches_direct() {
        for s in 0u8..=255 {
            assert_eq!(
                ENC_GENTAB[s as usize],
                encode_symbol_direct(s),
                "table disagrees with computed parity for symbol {:#04x}",
                s
            );
        }
    }

    #[test]
    fn test_hamming128_encode_decode_symbol() {
        for s in 0u8..=255 {
            let encoded = encode_symbol(s);
            let decoded = decode_symbol(encoded);
            assert_eq!(s, decoded, "symbol {} failed roundtrip", s);
        }
    }

    #[test]
    fn test_hamming128_odd_length() {
        let msg = [0xAB, 0xCD, 0xEF];
        // 3 bytes -> 5 bytes (ceil(3 * 12 / 8) = 5)
        let mut encoded = [0u8; 5];
        let mut decoded = [0u8; 3];

        hamming128_encode(&msg, &mut encoded);
        hamming128_decode(3, &encoded, &mut decoded);

        assert_eq!(msg, decoded);
    }

}
