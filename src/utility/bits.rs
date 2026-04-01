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
}