use crate::error::{Error, Result};

/// Pack binary array with symbol(s)
///   src     : source array
///   k       : bit index to write in src
///   b       : number of bits in input symbol
///   sym_in  : input symbol
pub fn pack_array(src: &mut [u8], k: usize, b: usize, sym_in: u8) -> Result<()> {
    let n = src.len();

    // validate input
    if k >= 8 * n {
        return Err(Error::Range(format!(
            "pack_array(), bit index {} exceeds array length {}",
            k,
            8 * n
        )));
    }
    if b > 8 {
        return Err(Error::Range(
            "pack_array(), symbol size cannot exceed 8 bits".into(),
        ));
    }

    // find base index
    let i0 = k / 8; // byte index
    let b0 = k - 8 * i0; // bit index

    // determine if index spans multiple bytes
    if b0 + b > 8 {
        // compute number of bits in each symbol
        let n0 = 8 - b0;
        let n1 = b - n0;

        // generate mask for each symbol
        let mask_0: u8 = 0xff >> (8 - n0);
        let mask_1: u8 = (0xff >> (8 - n1)) << (8 - n1);

        // shift then mask
        let sym_0 = (sym_in >> n1) & mask_0;
        let sym_1 = (sym_in << (8 - n1)) & mask_1;

        // mask and pack first byte
        src[i0] &= !mask_0; // clear relevant bits
        src[i0] |= sym_0; // set relevant bits

        // mask and pack second byte (if not exceeding array size)
        if i0 < n - 1 {
            src[i0 + 1] &= !mask_1; // clear relevant bits
            src[i0 + 1] |= sym_1; // set relevant bits
        }
    } else {
        // compute mask
        let mask_0: u8 = (0xff >> (8 - b)) << (8 - b - b0);
        let sym_0 = (sym_in << (8 - b - b0)) & mask_0;

        // shift then mask
        src[i0] &= !mask_0; // clear relevant bits
        src[i0] |= sym_0; // set relevant bits
    }
    Ok(())
}

/// Unpack symbols from binary array
///   src     : source array
///   k       : bit index to read from src
///   b       : number of bits in output symbol
/// Returns the unpacked symbol
pub fn unpack_array(src: &[u8], k: usize, b: usize) -> Result<u8> {
    let n = src.len();

    // validate input
    if k >= 8 * n {
        return Err(Error::Range(format!(
            "unpack_array(), bit index {} exceeds array length {}",
            k,
            8 * n
        )));
    }
    if b > 8 {
        return Err(Error::Range(
            "unpack_array(), symbol size cannot exceed 8 bits".into(),
        ));
    }

    // find base index
    let i0 = k / 8; // byte index
    let b0 = k - 8 * i0; // bit index

    // determine if index spans multiple bytes
    let sym_out = if b0 + b > 8 {
        // compute number of bits in each symbol
        let n0 = 8 - b0;
        let n1 = b - n0;

        // generate mask for each symbol
        let mask_0: u8 = 0xff >> (8 - n0);
        let mask_1: u8 = 0xff >> (8 - n1);

        // shift then mask
        let sym_0 = src[i0] & mask_0;
        let sym_1 = if i0 == n - 1 {
            0x00
        } else {
            (src[i0 + 1] >> (8 - n1)) & mask_1
        };

        // concatenate output symbols
        (sym_0 << n1) | sym_1
    } else {
        // compute mask (use u16 to avoid overflow when b=8)
        let mask_0: u8 = ((1u16 << b) - 1) as u8;

        // shift then mask
        (src[i0] >> (8 - b - b0)) & mask_0
    };

    Ok(sym_out)
}

/// Pack one-bit symbols into bytes (8-bit symbols)
///   sym_in      : input symbols array (one bit per element)
///   sym_out     : output symbols (packed bytes)
/// Returns the number of bytes written
pub fn pack_bytes(sym_in: &[u8], sym_out: &mut [u8]) -> Result<usize> {
    let sym_in_len = sym_in.len();
    let sym_out_len = sym_out.len();

    let req_sym_out_len = (sym_in_len + 7) / 8;
    if sym_out_len < req_sym_out_len {
        return Err(Error::Config("pack_bytes(), output too short".into()));
    }

    let mut n = 0usize; // number of bytes written to output
    let mut byte: u8 = 0;

    for (i, &bit) in sym_in.iter().enumerate() {
        byte |= bit & 0x01;

        if (i + 1) % 8 == 0 {
            sym_out[n] = byte;
            n += 1;
            byte = 0;
        } else {
            byte <<= 1;
        }
    }

    if sym_in_len % 8 != 0 {
        sym_out[n] = byte >> 1;
        n += 1;
    }

    Ok(n)
}

/// Unpack 8-bit symbols (full bytes) into one-bit symbols
///   sym_in      : input symbols array (packed bytes)
///   sym_out     : output symbols array (one bit per element)
/// Returns the number of bits written
pub fn unpack_bytes(sym_in: &[u8], sym_out: &mut [u8]) -> Result<usize> {
    let sym_in_len = sym_in.len();
    let sym_out_len = sym_out.len();

    if sym_out_len < 8 * sym_in_len {
        return Err(Error::Config("unpack_bytes(), output too short".into()));
    }

    let mut n = 0usize;

    for &byte in sym_in {
        // unpack byte into 8 one-bit symbols
        sym_out[n] = (byte >> 7) & 0x01;
        sym_out[n + 1] = (byte >> 6) & 0x01;
        sym_out[n + 2] = (byte >> 5) & 0x01;
        sym_out[n + 3] = (byte >> 4) & 0x01;
        sym_out[n + 4] = (byte >> 3) & 0x01;
        sym_out[n + 5] = (byte >> 2) & 0x01;
        sym_out[n + 6] = (byte >> 1) & 0x01;
        sym_out[n + 7] = byte & 0x01;
        n += 8;
    }

    Ok(n)
}

/// Repack bytes with arbitrary symbol sizes
///   sym_in      : input symbols array
///   sym_in_bps  : number of bits per input symbol
///   sym_out     : output symbols array
///   sym_out_bps : number of bits per output symbol
/// Returns the number of output symbols written
pub fn repack_bytes(
    sym_in: &[u8],
    sym_in_bps: usize,
    sym_out: &mut [u8],
    sym_out_bps: usize,
) -> Result<usize> {
    let sym_in_len = sym_in.len();
    let sym_out_len = sym_out.len();

    // compute number of output symbols and determine if output array
    // is sufficiently sized
    let total_bits = sym_in_len * sym_in_bps;
    let req_sym_out_len = (total_bits + (sym_out_bps - 1)) / sym_out_bps;
    if sym_out_len < req_sym_out_len {
        return Err(Error::Config(format!(
            "repack_bytes(), output too short; {} {}-bit symbols cannot be packed into {} {}-bit elements",
            sym_in_len, sym_in_bps, sym_out_len, sym_out_bps
        )));
    }

    let mut s_in: u8 = 0; // input symbol
    let mut s_out: u8 = 0; // output symbol

    let mut i_in = 0usize; // input index counter
    let mut i_out = 0usize; // output index counter
    let mut k = 0usize; // input symbol enable
    let mut n = 0usize; // output symbol enable

    for _ in 0..total_bits {
        // shift output symbol by one bit
        s_out <<= 1;

        // pop input if necessary
        if k == 0 {
            s_in = sym_in[i_in];
            i_in += 1;
        }

        // compute shift amount and append input bit at index to output symbol
        let v = sym_in_bps - k - 1;
        s_out |= (s_in >> v) & 0x01;

        // push output if available
        if n == sym_out_bps - 1 {
            sym_out[i_out] = s_out;
            i_out += 1;
            s_out = 0;
        }

        // update input/output symbol pop/push flags
        k = (k + 1) % sym_in_bps;
        n = (n + 1) % sym_out_bps;
    }

    // if uneven, push zeros into remaining output symbol
    if i_out != req_sym_out_len {
        for _ in n..sym_out_bps {
            s_out <<= 1;
        }
        sym_out[i_out] = s_out;
        i_out += 1;
    }

    Ok(i_out)
}


// Constants for lookup tables

// Format the following array with 8 columns, lowercase hex
#[rustfmt::skip]
const REVERSE_BYTE_GENTAB: [u8; 256] = [
    0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0,
    0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
    0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
    0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
    0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4,
    0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
    0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
    0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
    0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
    0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
    0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea,
    0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
    0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6,
    0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
    0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
    0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
    0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1,
    0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
    0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9,
    0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
    0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
    0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
    0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed,
    0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
    0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3,
    0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
    0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
    0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
    0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7,
    0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
    0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
    0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
];

/// Count the number of ones in an integer
pub fn count_ones(x: u32) -> u32 {
    x.count_ones() as u32
}

/// Count the number of ones in an integer, modulo 2
pub fn count_ones_mod2(x: u32) -> u32 {
    x.count_ones() & 1
}

/// Count the binary dot-product between two integers
pub fn bdotprod(x: u32, y: u32) -> u32 {
    (x & y).count_ones() & 1 as u32
}

/// Counts the number of different bits between two symbols
pub fn count_bit_errors(s1: u32, s2: u32) -> u32 {
    (s1 ^ s2).count_ones() as u32
}

/// Counts the number of different bits between two arrays of symbols
pub fn count_bit_errors_array(msg0: &[u8], msg1: &[u8]) -> u32 {
    msg0.iter()
        .zip(msg1.iter())
        .map(|(&a, &b)| (a ^ b).count_ones() as u32)
        .sum()
}

/// Print string of bits to standard output
pub fn print_bitstring(x: u32, n: u32) {
    for i in (0..n).rev() {
        print!("{}", (x >> i) & 1);
    }
}

/// Slow implementation of byte reversal
pub fn reverse_byte(x: u8) -> u8 {
    REVERSE_BYTE_GENTAB[x as usize] as u8
}

/// Reverse integer with 8 bits of data
pub fn reverse_8(x: u32) -> u32 {
    REVERSE_BYTE_GENTAB[x as usize] as u32
}

/// Reverse integer with 16 bits of data
pub fn reverse_16(x: u32) -> u32 {
    ((REVERSE_BYTE_GENTAB[(x & 0xff) as usize] as u32) << 8) |
    (REVERSE_BYTE_GENTAB[(x >> 8) as usize] as u32)
}

/// Reverse integer with 24 bits of data
pub fn reverse_24(x: u32) -> u32 {
    ((REVERSE_BYTE_GENTAB[(x & 0xff) as usize] as u32) << 16) |
    ((REVERSE_BYTE_GENTAB[((x >> 8) & 0xff) as usize] as u32) << 8) |
    (REVERSE_BYTE_GENTAB[((x >> 16) & 0xff) as usize] as u32)
}

/// Reverse integer with 32 bits of data
pub fn reverse_32(x: u32) -> u32 {
    ((REVERSE_BYTE_GENTAB[(x & 0xff) as usize] as u32) << 24) |
    ((REVERSE_BYTE_GENTAB[((x >> 8) & 0xff) as usize] as u32) << 16) |
    ((REVERSE_BYTE_GENTAB[((x >> 16) & 0xff) as usize] as u32) << 8) |
    (REVERSE_BYTE_GENTAB[(x >> 24) as usize] as u32)
}

pub fn count_leading_zeros(x: u32) -> u32 {
    x.leading_zeros() as u32
}

pub fn msb_index(x: u32) -> u32 {
    32 - x.leading_zeros() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_count_ones)]
    fn test_count_ones() {
        assert_eq!(count_ones(0x0000), 0);
        assert_eq!(count_ones(0x0001), 1);
        assert_eq!(count_ones(0x0003), 2);
        assert_eq!(count_ones(0xFFFF), 16);
        assert_eq!(count_ones(0x00FF), 8);
        assert_eq!(count_ones(0x5555), 8);
        assert_eq!(count_ones(0x0007), 3);
        assert_eq!(count_ones(0x0037), 5);
        assert_eq!(count_ones(0x0137), 6);
        assert_eq!(count_ones(0xf137), 10);
    }

    #[test]
    #[autotest_annotate(autotest_count_ones_mod2)]
    fn test_count_ones_mod2() {
        assert_eq!(count_ones_mod2(0x0000), 0);
        assert_eq!(count_ones_mod2(0x0001), 1);
        assert_eq!(count_ones_mod2(0x0003), 0);
        assert_eq!(count_ones_mod2(0xFFFF), 0);
        assert_eq!(count_ones_mod2(0x00FF), 0);
        assert_eq!(count_ones_mod2(0x5555), 0);
        assert_eq!(count_ones_mod2(0x0007), 1);
        assert_eq!(count_ones_mod2(0x0037), 1);
        assert_eq!(count_ones_mod2(0x0137), 0);
        assert_eq!(count_ones_mod2(0xf137), 0);
    }

    #[test]
    #[autotest_annotate(autotest_bdotprod)]
    fn test_bdotprod() {
        // simple checks
        assert_eq!(bdotprod(0x1111, 0x1111), 0);
        assert_eq!(bdotprod(0xffff, 0xffff), 0);
        assert_eq!(bdotprod(0xffff, 0x0000), 0);
        assert_eq!(bdotprod(0x0001, 0x0001), 1);

        // random data
        assert_eq!(bdotprod(0x4379, 0xf2dc), 1);
        assert_eq!(bdotprod(0xc9a1, 0xc99d), 0);
        assert_eq!(bdotprod(0xa8ba, 0x26d9), 0);
        assert_eq!(bdotprod(0x5235, 0x8e1b), 1);
        assert_eq!(bdotprod(0x0f85, 0xa3d1), 0);
        assert_eq!(bdotprod(0x23e0, 0x5869), 0);
        assert_eq!(bdotprod(0xc8a4, 0x32a4), 1);
        assert_eq!(bdotprod(0xe1c3, 0x000c), 0);
        assert_eq!(bdotprod(0x4039, 0x192d), 1);
        assert_eq!(bdotprod(0x2e1c, 0x55a3), 1);
        assert_eq!(bdotprod(0x5a1b, 0x0241), 0);
        assert_eq!(bdotprod(0x440c, 0x7ddb), 1);
        assert_eq!(bdotprod(0xd2e2, 0x5c98), 1);
        assert_eq!(bdotprod(0xe36c, 0x5bc9), 1);
        assert_eq!(bdotprod(0xaa96, 0xf233), 1);
        assert_eq!(bdotprod(0xab0f, 0x3912), 0);
    }

    #[test]
    #[autotest_annotate(autotest_count_leading_zeros)]
    fn test_count_leading_zeros() {
        // NOTE: this test assumes a 4-byte integer
        assert_eq!(count_leading_zeros(0x00000000), 32);

        assert_eq!(count_leading_zeros(0x00000001), 31);
        assert_eq!(count_leading_zeros(0x00000002), 30);
        assert_eq!(count_leading_zeros(0x00000004), 29);
        assert_eq!(count_leading_zeros(0x00000008), 28);

        assert_eq!(count_leading_zeros(0x00000010), 27);
        assert_eq!(count_leading_zeros(0x00000020), 26);
        assert_eq!(count_leading_zeros(0x00000040), 25);
        assert_eq!(count_leading_zeros(0x00000080), 24);

        assert_eq!(count_leading_zeros(0x00000100), 23);
        assert_eq!(count_leading_zeros(0x00000200), 22);
        assert_eq!(count_leading_zeros(0x00000400), 21);
        assert_eq!(count_leading_zeros(0x00000800), 20);

        assert_eq!(count_leading_zeros(0x00001000), 19);
        assert_eq!(count_leading_zeros(0x00002000), 18);
        assert_eq!(count_leading_zeros(0x00004000), 17);
        assert_eq!(count_leading_zeros(0x00008000), 16);

        assert_eq!(count_leading_zeros(0x00010000), 15);
        assert_eq!(count_leading_zeros(0x00020000), 14);
        assert_eq!(count_leading_zeros(0x00040000), 13);
        assert_eq!(count_leading_zeros(0x00080000), 12);

        assert_eq!(count_leading_zeros(0x00100000), 11);
        assert_eq!(count_leading_zeros(0x00200000), 10);
        assert_eq!(count_leading_zeros(0x00400000),  9);
        assert_eq!(count_leading_zeros(0x00800000),  8);

        assert_eq!(count_leading_zeros(0x01000000),  7);
        assert_eq!(count_leading_zeros(0x02000000),  6);
        assert_eq!(count_leading_zeros(0x04000000),  5);
        assert_eq!(count_leading_zeros(0x08000000),  4);

        assert_eq!(count_leading_zeros(0x10000000),  3);
        assert_eq!(count_leading_zeros(0x20000000),  2);
        assert_eq!(count_leading_zeros(0x40000000),  1);
        assert_eq!(count_leading_zeros(0x80000000),  0);
    }

    #[test]
    #[autotest_annotate(autotest_msb_index)]
    fn test_msb_index() {
        // NOTE: this test assumes a 4-byte integer
        assert_eq!(msb_index(0x00000000),  0);

        assert_eq!(msb_index(0x00000001),  1);
        assert_eq!(msb_index(0x00000002),  2);
        assert_eq!(msb_index(0x00000004),  3);
        assert_eq!(msb_index(0x00000008),  4);

        assert_eq!(msb_index(0x00000010),  5);
        assert_eq!(msb_index(0x00000020),  6);
        assert_eq!(msb_index(0x00000040),  7);
        assert_eq!(msb_index(0x00000080),  8);

        assert_eq!(msb_index(0x00000100),  9);
        assert_eq!(msb_index(0x00000200), 10);
        assert_eq!(msb_index(0x00000400), 11);
        assert_eq!(msb_index(0x00000800), 12);

        assert_eq!(msb_index(0x00001000), 13);
        assert_eq!(msb_index(0x00002000), 14);
        assert_eq!(msb_index(0x00004000), 15);
        assert_eq!(msb_index(0x00008000), 16);

        assert_eq!(msb_index(0x00010000), 17);
        assert_eq!(msb_index(0x00020000), 18);
        assert_eq!(msb_index(0x00040000), 19);
        assert_eq!(msb_index(0x00080000), 20);

        assert_eq!(msb_index(0x00100000), 21);
        assert_eq!(msb_index(0x00200000), 22);
        assert_eq!(msb_index(0x00400000), 23);
        assert_eq!(msb_index(0x00800000), 24);

        assert_eq!(msb_index(0x01000000), 25);
        assert_eq!(msb_index(0x02000000), 26);
        assert_eq!(msb_index(0x04000000), 27);
        assert_eq!(msb_index(0x08000000), 28);

        assert_eq!(msb_index(0x10000000), 29);
        assert_eq!(msb_index(0x20000000), 30);
        assert_eq!(msb_index(0x40000000), 31);
        assert_eq!(msb_index(0x80000000), 32);
    }

    #[test]
    #[autotest_annotate(autotest_pack_array)]
    fn test_pack_array() {
        // input symbols
        let sym_size: [usize; 9] = [8, 2, 3, 6, 1, 3, 3, 4, 3];
        let input: [u8; 9] = [
            0x81, // 1000 0001
            0x03, //        11
            0x05, //       101
            0x3a, //   11 1010
            0x01, //         1
            0x07, //       111
            0x06, //       110
            0x0a, //      1010
            0x04, //     10[0] <- last bit is stripped
        ];

        // output       : 1000 0001 1110 1111 0101 1111 1010 1010
        // symbol       : 0000 0000 1122 2333 3334 5556 6677 7788
        let output_test: [u8; 4] = [0x81, 0xEF, 0x5F, 0xAA];
        let mut output: [u8; 4] = [0xff, 0xff, 0xff, 0xff];

        let mut k = 0usize;
        for i in 0..9 {
            pack_array(&mut output, k, sym_size[i], input[i]).unwrap();
            k += sym_size[i];
        }

        assert_eq!(&output[..4], &output_test[..4]);
    }

    #[test]
    #[autotest_annotate(autotest_unpack_array)]
    fn test_unpack_array() {
        // input        : 1000 0001 1110 1111 0101 1111 1010 1010
        // symbol       : 0000 0000 1122 2333 3334 5556 6677 7788
        let input: [u8; 4] = [0x81, 0xEF, 0x5F, 0xAA];
        let sym_size: [usize; 9] = [8, 2, 3, 6, 1, 3, 3, 4, 3];

        // output syms
        let output_test: [u8; 9] = [
            0x81, // 1000 0001
            0x03, //        11
            0x05, //       101
            0x3a, //   11 1010
            0x01, //         1
            0x07, //       111
            0x06, //       110
            0x0a, //      1010
            0x04, //     10[0] <- last bit is implied
        ];

        let mut output: [u8; 9] = [0; 9];

        let mut k = 0usize;
        for i in 0..9 {
            output[i] = unpack_array(&input, k, sym_size[i]).unwrap();
            k += sym_size[i];
        }

        assert_eq!(&output[..9], &output_test[..9]);
    }

    #[test]
    #[autotest_annotate(autotest_repack_array)]
    fn test_repack_array() {
        use rand::Rng;

        let n = 512usize; // input/output array size
        let mut src = vec![0u8; n]; // original data array
        let mut dst = vec![0u8; n]; // repacked data array

        let mut rng = rand::thread_rng();

        // initialize input array with random data
        for i in 0..n {
            src[i] = rng.gen::<u8>();
        }

        let mut k = 0usize;
        while k < 8 * n {
            // random symbol size
            let sym_size = (rng.gen::<usize>() % 8) + 1;

            // unpack symbol from input array
            let sym = unpack_array(&src, k, sym_size).unwrap();

            // pack symbol into output array
            pack_array(&mut dst, k, sym_size, sym).unwrap();

            // update bit index counter
            k += sym_size;
        }

        assert_eq!(&src[..n], &dst[..n]);
    }

    #[test]
    #[autotest_annotate(autotest_pack_bytes_01)]
    fn test_pack_bytes_01() {
        let mut output = [0u8; 8];

        #[rustfmt::skip]
        let input: [u8; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0, // 0:   0000 0000
            1, 1, 1, 1, 1, 1, 1, 1, // 255: 1111 1111
            0, 0, 0, 0, 1, 1, 1, 1, // 15:  0000 1111
            1, 0, 1, 0, 1, 0, 1, 0, // 170: 1010 1010
        ];

        // Test packing entire array (32 elements)
        let output_test_01: [u8; 4] = [0x00, 0xFF, 0x0F, 0xAA];
        let n = pack_bytes(&input[..32], &mut output).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&output[..4], &output_test_01[..4]);

        // Test packing only 28 elements
        let output_test_02: [u8; 4] = [0x00, 0xFF, 0x0F, 0x0A];
        let n = pack_bytes(&input[..28], &mut output).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&output[..4], &output_test_02[..4]);

        // Test packing only 25 elements
        let output_test_03: [u8; 4] = [0x00, 0xFF, 0x0F, 0x01];
        let n = pack_bytes(&input[..25], &mut output).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&output[..4], &output_test_03[..4]);

        // Test packing only 24 elements (3 bytes)
        let output_test_04: [u8; 3] = [0x00, 0xFF, 0x0F];
        let n = pack_bytes(&input[..24], &mut output).unwrap();
        assert_eq!(n, 3);
        assert_eq!(&output[..3], &output_test_04[..3]);
    }

    #[test]
    #[autotest_annotate(autotest_unpack_bytes_01)]
    fn test_unpack_bytes_01() {
        let input: [u8; 5] = [0x00, 0x01, 0xFF, 0x0F, 0xAA];

        let mut output = [0u8; 64];

        #[rustfmt::skip]
        let output_test: [u8; 40] = [
            0, 0, 0, 0, 0, 0, 0, 0, // 0:   0000 0000
            0, 0, 0, 0, 0, 0, 0, 1, // 1:   0000 0001
            1, 1, 1, 1, 1, 1, 1, 1, // 255: 1111 1111
            0, 0, 0, 0, 1, 1, 1, 1, // 15:  0000 1111
            1, 0, 1, 0, 1, 0, 1, 0, // 170: 1010 1010
        ];

        // Test unpacking first 4 bytes
        let n = unpack_bytes(&input[..4], &mut output).unwrap();
        assert_eq!(n, 32);
        assert_eq!(&output[..32], &output_test[..32]);
    }

    #[test]
    #[autotest_annotate(autotest_repack_bytes_01)]
    fn test_repack_bytes_01() {
        let input: [u8; 4] = [
            0x07, // 111
            0x00, // 000
            0x06, // 110
            0x07, // 111
        ];

        let output_test: [u8; 6] = [
            0x03, // 11
            0x02, // 10
            0x00, // 00
            0x03, // 11
            0x01, // 01
            0x03, // 11
        ];

        let mut output = [0u8; 6];

        let n = repack_bytes(&input, 3, &mut output, 2).unwrap();

        assert_eq!(n, 6);
        assert_eq!(&output[..6], &output_test[..6]);
    }

    #[test]
    #[autotest_annotate(autotest_repack_bytes_02)]
    fn test_repack_bytes_02() {
        let input: [u8; 3] = [
            0x01, // 00001
            0x02, // 00010
            0x04, // 00100
        ];

        let output_test: [u8; 5] = [
            0x00, // 000
            0x02, // 010
            0x01, // 001
            0x00, // 000
            0x04, // 100
        ];

        let mut output = [0u8; 5];

        let n = repack_bytes(&input, 5, &mut output, 3).unwrap();

        assert_eq!(n, 5);
        assert_eq!(&output[..5], &output_test[..5]);
    }

    #[test]
    #[autotest_annotate(autotest_repack_bytes_03)]
    fn test_repack_bytes_03() {
        let input: [u8; 5] = [
            0x00, // 000
            0x02, // 010
            0x01, // 001
            0x00, // 000
            0x04, // 100
        ];

        let output_test: [u8; 3] = [
            0x01, // 00001
            0x02, // 00010
            0x04, // 00100
        ];

        let mut output = [0u8; 3];

        let n = repack_bytes(&input, 3, &mut output, 5).unwrap();

        assert_eq!(n, 3);
        assert_eq!(&output[..3], &output_test[..3]);
    }

    #[test]
    #[autotest_annotate(autotest_repack_bytes_04_uneven)]
    fn test_repack_bytes_04_uneven() {
        let input: [u8; 3] = [
            0x07, // 111
            0x07, // 111
            0x07, // 111(0)
        ];

        let output_test: [u8; 5] = [
            0x03, // 11
            0x03, // 11
            0x03, // 11
            0x03, // 11
            0x02, // 10
        ];

        let mut output = [0u8; 5];

        let n = repack_bytes(&input, 3, &mut output, 2).unwrap();

        assert_eq!(n, 5);
        assert_eq!(&output[..5], &output_test[..5]);
    }
}