//
// bshift_array
//
// bit-wise array shifting
//

use crate::error::{Error, Result};

/// shift array to the left b bits, filling in zeros
///   src : source array
///   b   : number of bits to shift, in [0,7]
pub fn lbshift(src: &mut [u8], b: usize) -> Result<()> {
    // validate input
    if b >= 8 {
        return Err(Error::Range(
            "lbshift(), shift amount must be in [0,7]".into(),
        ));
    }
    if src.is_empty() {
        return Ok(());
    }

    // invoke circular shift left and mask last byte
    lbcircshift(src, b)?;
    src[src.len() - 1] &= 0xffu8 << b;
    Ok(())
}

/// shift array to the right b bits, filling in zeros
///   src : source array
///   b   : number of bits to shift, in [0,7]
pub fn rbshift(src: &mut [u8], b: usize) -> Result<()> {
    // validate input
    if b >= 8 {
        return Err(Error::Range(
            "rbshift(), shift amount must be in [0,7]".into(),
        ));
    }
    if src.is_empty() {
        return Ok(());
    }

    // invoke circular shift right and mask first byte
    rbcircshift(src, b)?;
    src[0] &= 0xffu8 >> b;
    Ok(())
}

/// circular shift array to the left b bits
///   src : source array
///   b   : number of bits to shift, in [0,7]
pub fn lbcircshift(src: &mut [u8], b: usize) -> Result<()> {
    // validate input
    if b >= 8 {
        return Err(Error::Range(
            "lbcircshift(), shift amount must be in [0,7]".into(),
        ));
    }
    // shifting by nothing leaves the array alone, and returning here keeps
    // shift_1 below 8 so the byte shifts stay in range
    if src.is_empty() || b == 0 {
        return Ok(());
    }

    let n = src.len();
    let shift_0 = b; // shift amount: first byte
    let shift_1 = 8 - b; // shift amount: second byte
    let mask_0 = 0xffu8 << shift_0; // bit mask: first byte
    let mask_1 = 0xffu8 >> shift_1; // bit mask: second byte

    // shift then mask
    let src_0 = src[0]; // retain first byte
    for i in 0..n {
        // strip bytes
        let byte_0 = src[i];
        let byte_1 = if i == n - 1 { src_0 } else { src[i + 1] };

        // shift then mask
        src[i] = ((byte_0 << shift_0) & mask_0) | ((byte_1 >> shift_1) & mask_1);
    }
    Ok(())
}

/// circular shift array to the right b bits
///   src : source array
///   b   : number of bits to shift, in [0,7]
pub fn rbcircshift(src: &mut [u8], b: usize) -> Result<()> {
    // validate input
    if b >= 8 {
        return Err(Error::Range(
            "rbcircshift(), shift amount must be in [0,7]".into(),
        ));
    }
    // shifting by nothing leaves the array alone, and returning here keeps
    // shift_0 below 8 so the byte shifts stay in range
    if src.is_empty() || b == 0 {
        return Ok(());
    }

    let n = src.len();
    let shift_0 = 8 - b; // shift amount: first byte
    let shift_1 = b; // shift amount: second byte
    let mask_0 = 0xffu8 << shift_0; // bit mask: first byte
    let mask_1 = 0xffu8 >> shift_1; // bit mask: second byte

    // shift then mask
    let src_n = src[n - 1]; // retain last byte
    for i in (0..n).rev() {
        // strip bytes
        let byte_0 = if i == 0 { src_n } else { src[i - 1] };
        let byte_1 = src[i];

        // shift then mask
        src[i] = ((byte_0 << shift_0) & mask_0) | ((byte_1 >> shift_1) & mask_1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_lbshift)]
    fn test_lbshift() {
        // input        : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [0]   : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [1]   : 0000 0011 1101 1110 1011 1111 0101 0100
        // output [2]   : 0000 0111 1011 1101 0111 1110 1010 1000
        // output [3]   : 0000 1111 0111 1010 1111 1101 0101 0000
        // output [4]   : 0001 1110 1111 0101 1111 1010 1010 0000
        // output [5]   : 0011 1101 1110 1011 1111 0101 0100 0000
        // output [6]   : 0111 1011 1101 0111 1110 1010 1000 0000
        // output [7]   : 1111 0111 1010 1111 1101 0101 0000 0000
        let input = [0x81u8, 0xEF, 0x5F, 0xAA];
        let expected: [[u8; 4]; 8] = [
            [0x81, 0xEF, 0x5F, 0xAA],
            [0x03, 0xDE, 0xBF, 0x54],
            [0x07, 0xBD, 0x7E, 0xA8],
            [0x0F, 0x7A, 0xFD, 0x50],
            [0x1E, 0xF5, 0xFA, 0xA0],
            [0x3D, 0xEB, 0xF5, 0x40],
            [0x7B, 0xD7, 0xEA, 0x80],
            [0xF7, 0xAF, 0xD5, 0x00],
        ];

        for (b, want) in expected.iter().enumerate() {
            let mut output = input;
            lbshift(&mut output, b).unwrap();
            assert_eq!(&output, want, "shift {}", b);
        }
    }

    #[test]
    #[autotest_annotate(autotest_rbshift)]
    fn test_rbshift() {
        // input        : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [0]   : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [1]   : 0100 0000 1111 0111 1010 1111 1101 0101
        // output [2]   : 0010 0000 0111 1011 1101 0111 1110 1010
        // output [3]   : 0001 0000 0011 1101 1110 1011 1111 0101
        // output [4]   : 0000 1000 0001 1110 1111 0101 1111 1010
        // output [5]   : 0000 0100 0000 1111 0111 1010 1111 1101
        // output [6]   : 0000 0010 0000 0111 1011 1101 0111 1110
        // output [7]   : 0000 0001 0000 0011 1101 1110 1011 1111
        let input = [0x81u8, 0xEF, 0x5F, 0xAA];
        let expected: [[u8; 4]; 8] = [
            [0x81, 0xEF, 0x5F, 0xAA],
            [0x40, 0xF7, 0xAF, 0xD5],
            [0x20, 0x7B, 0xD7, 0xEA],
            [0x10, 0x3D, 0xEB, 0xF5],
            [0x08, 0x1E, 0xF5, 0xFA],
            [0x04, 0x0F, 0x7A, 0xFD],
            [0x02, 0x07, 0xBD, 0x7E],
            [0x01, 0x03, 0xDE, 0xBF],
        ];

        for (b, want) in expected.iter().enumerate() {
            let mut output = input;
            rbshift(&mut output, b).unwrap();
            assert_eq!(&output, want, "shift {}", b);
        }
    }

    #[test]
    #[autotest_annotate(autotest_lbcircshift)]
    fn test_lbcircshift() {
        // input        : 1001 0001 1110 1111 0101 1111 1010 1010
        // output [0]   : 1001 0001 1110 1111 0101 1111 1010 1010
        // output [1]   : 0010 0011 1101 1110 1011 1111 0101 0101
        // output [2]   : 0100 0111 1011 1101 0111 1110 1010 1010
        // output [3]   : 1000 1111 0111 1010 1111 1101 0101 0100
        // output [4]   : 0001 1110 1111 0101 1111 1010 1010 1001
        // output [5]   : 0011 1101 1110 1011 1111 0101 0101 0010
        // output [6]   : 0111 1011 1101 0111 1110 1010 1010 0100
        // output [7]   : 1111 0111 1010 1111 1101 0101 0100 1000
        let input = [0x91u8, 0xEF, 0x5F, 0xAA];
        let expected: [[u8; 4]; 8] = [
            [0x91, 0xEF, 0x5F, 0xAA],
            [0x23, 0xDE, 0xBF, 0x55],
            [0x47, 0xBD, 0x7E, 0xAA],
            [0x8F, 0x7A, 0xFD, 0x54],
            [0x1E, 0xF5, 0xFA, 0xA9],
            [0x3D, 0xEB, 0xF5, 0x52],
            [0x7B, 0xD7, 0xEA, 0xA4],
            [0xF7, 0xAF, 0xD5, 0x48],
        ];

        for (b, want) in expected.iter().enumerate() {
            let mut output = input;
            lbcircshift(&mut output, b).unwrap();
            assert_eq!(&output, want, "shift {}", b);
        }
    }

    #[test]
    #[autotest_annotate(autotest_rbcircshift)]
    fn test_rbcircshift() {
        // input        : 1001 0001 1110 1111 0101 1111 1010 1010
        // output [0]   : 1001 0001 1110 1111 0101 1111 1010 1010
        // output [1]   : 0100 1000 1111 0111 1010 1111 1101 0101
        // output [2]   : 1010 0100 0111 1011 1101 0111 1110 1010
        // output [3]   : 0101 0010 0011 1101 1110 1011 1111 0101
        // output [4]   : 1010 1001 0001 1110 1111 0101 1111 1010
        // output [5]   : 0101 0100 1000 1111 0111 1010 1111 1101
        // output [6]   : 1010 1010 0100 0111 1011 1101 0111 1110
        // output [7]   : 0101 0101 0010 0011 1101 1110 1011 1111
        let input = [0x91u8, 0xEF, 0x5F, 0xAA];
        let expected: [[u8; 4]; 8] = [
            [0x91, 0xEF, 0x5F, 0xAA],
            [0x48, 0xF7, 0xAF, 0xD5],
            [0xA4, 0x7B, 0xD7, 0xEA],
            [0x52, 0x3D, 0xEB, 0xF5],
            [0xA9, 0x1E, 0xF5, 0xFA],
            [0x54, 0x8F, 0x7A, 0xFD],
            [0xAA, 0x47, 0xBD, 0x7E],
            [0x55, 0x23, 0xDE, 0xBF],
        ];

        for (b, want) in expected.iter().enumerate() {
            let mut output = input;
            rbcircshift(&mut output, b).unwrap();
            assert_eq!(&output, want, "shift {}", b);
        }
    }

    #[test]
    fn test_bshift_range() {
        for b in [8usize, 9, 100] {
            let mut x = [0x81u8, 0xEF];
            assert!(lbshift(&mut x, b).is_err(), "lbshift {}", b);
            assert!(rbshift(&mut x, b).is_err(), "rbshift {}", b);
            assert!(lbcircshift(&mut x, b).is_err(), "lbcircshift {}", b);
            assert!(rbcircshift(&mut x, b).is_err(), "rbcircshift {}", b);
        }
    }

    #[test]
    fn test_bshift_empty() {
        let mut empty: [u8; 0] = [];
        lbshift(&mut empty, 3).unwrap();
        rbshift(&mut empty, 3).unwrap();
        lbcircshift(&mut empty, 3).unwrap();
        rbcircshift(&mut empty, 3).unwrap();
    }

    #[test]
    fn test_bcircshift_round_trip() {
        let input = [0x91u8, 0xEF, 0x5F, 0xAA];

        for b in 0..8usize {
            let mut x = input;
            lbcircshift(&mut x, b).unwrap();
            rbcircshift(&mut x, b).unwrap();
            assert_eq!(x, input, "round trip {}", b);
        }
    }

    #[test]
    fn test_bcircshift_single_byte() {
        for b in 0..8usize {
            let mut x = [0b1000_0001u8];
            lbcircshift(&mut x, b).unwrap();
            assert_eq!(x[0], 0b1000_0001u8.rotate_left(b as u32), "shift {}", b);

            let mut x = [0b1000_0001u8];
            rbcircshift(&mut x, b).unwrap();
            assert_eq!(x[0], 0b1000_0001u8.rotate_right(b as u32), "shift {}", b);
        }
    }
}
