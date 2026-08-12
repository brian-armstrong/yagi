//
// shift_array
//
// byte-wise array shifting
//

/// shift array to the left b bytes, filling in zeros
///   src : source array
///   b   : number of bytes to shift
pub fn lshift(src: &mut [u8], b: usize) {
    let n = src.len();

    // shift amount exceeds buffer size; fill with zeros
    if b >= n {
        src.fill(0x00);
        return;
    }

    // move memory
    src.copy_within(b..n, 0);

    // fill remaining buffer with zeros
    src[n - b..].fill(0x00);
}

/// shift array to the right b bytes, filling in zeros
///   src : source array
///   b   : number of bytes to shift
pub fn rshift(src: &mut [u8], b: usize) {
    let n = src.len();

    // shift amount exceeds buffer size; fill with zeros
    if b >= n {
        src.fill(0x00);
        return;
    }

    // move memory
    src.copy_within(0..n - b, b);

    // fill remaining buffer with zeros
    src[..b].fill(0x00);
}

/// circular shift array to the left b bytes
///   src : source array
///   b   : number of bytes to shift
pub fn lcircshift(src: &mut [u8], b: usize) {
    // validate input
    if src.is_empty() {
        return;
    }

    // ensure 0 <= b < n
    src.rotate_left(b % src.len());
}

/// circular shift array to the right b bytes
///   src : source array
///   b   : number of bytes to shift
pub fn rcircshift(src: &mut [u8], b: usize) {
    // validate input
    if src.is_empty() {
        return;
    }

    // ensure 0 <= b < n
    src.rotate_right(b % src.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_lshift)]
    fn test_lshift() {
        // input        : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [0]   : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [1]   : 1110 1111 0101 1111 1010 1010 0000 0000
        // output [2]   : 0101 1111 1010 1010 0000 0000 0000 0000
        // output [3]   : 1010 1010 0000 0000 0000 0000 0000 0000
        // output [4]   : 0000 0000 0000 0000 0000 0000 0000 0000
        let input = [0x81u8, 0xEF, 0x5F, 0xAA];
        let expected: [[u8; 4]; 5] = [
            [0x81, 0xEF, 0x5F, 0xAA],
            [0xEF, 0x5F, 0xAA, 0x00],
            [0x5F, 0xAA, 0x00, 0x00],
            [0xAA, 0x00, 0x00, 0x00],
            [0x00, 0x00, 0x00, 0x00],
        ];

        for (b, want) in expected.iter().enumerate() {
            let mut output = input;
            lshift(&mut output, b);
            assert_eq!(&output, want, "shift {}", b);
        }
    }

    #[test]
    #[autotest_annotate(autotest_rshift)]
    fn test_rshift() {
        // input        : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [0]   : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [1]   : 0000 0000 1000 0001 1110 1111 0101 1111
        // output [2]   : 0000 0000 0000 0000 1000 0001 1110 1111
        // output [3]   : 0000 0000 0000 0000 0000 0000 1000 0001
        // output [4]   : 0000 0000 0000 0000 0000 0000 0000 0000
        let input = [0x81u8, 0xEF, 0x5F, 0xAA];
        let expected: [[u8; 4]; 5] = [
            [0x81, 0xEF, 0x5F, 0xAA],
            [0x00, 0x81, 0xEF, 0x5F],
            [0x00, 0x00, 0x81, 0xEF],
            [0x00, 0x00, 0x00, 0x81],
            [0x00, 0x00, 0x00, 0x00],
        ];

        for (b, want) in expected.iter().enumerate() {
            let mut output = input;
            rshift(&mut output, b);
            assert_eq!(&output, want, "shift {}", b);
        }
    }

    #[test]
    #[autotest_annotate(autotest_lcircshift)]
    fn test_lcircshift() {
        // input        : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [0]   : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [1]   : 1110 1111 0101 1111 1010 1010 1000 0001
        // output [2]   : 0101 1111 1010 1010 1000 0001 1110 1111
        // output [3]   : 1010 1010 1000 0001 1110 1111 0101 1111
        // output [4]   : 1000 0001 1110 1111 0101 1111 1010 1010
        let input = [0x81u8, 0xEF, 0x5F, 0xAA];
        let expected: [[u8; 4]; 5] = [
            [0x81, 0xEF, 0x5F, 0xAA],
            [0xEF, 0x5F, 0xAA, 0x81],
            [0x5F, 0xAA, 0x81, 0xEF],
            [0xAA, 0x81, 0xEF, 0x5F],
            [0x81, 0xEF, 0x5F, 0xAA],
        ];

        for (b, want) in expected.iter().enumerate() {
            let mut output = input;
            lcircshift(&mut output, b);
            assert_eq!(&output, want, "shift {}", b);
        }
    }

    #[test]
    #[autotest_annotate(autotest_rcircshift)]
    fn test_rcircshift() {
        // input        : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [0]   : 1000 0001 1110 1111 0101 1111 1010 1010
        // output [1]   : 1010 1010 1000 0001 1110 1111 0101 1111
        // output [2]   : 0101 1111 1010 1010 1000 0001 1110 1111
        // output [3]   : 1110 1111 0101 1111 1010 1010 1000 0001
        // output [4]   : 1000 0001 1110 1111 0101 1111 1010 1010
        let input = [0x81u8, 0xEF, 0x5F, 0xAA];
        let expected: [[u8; 4]; 5] = [
            [0x81, 0xEF, 0x5F, 0xAA],
            [0xAA, 0x81, 0xEF, 0x5F],
            [0x5F, 0xAA, 0x81, 0xEF],
            [0xEF, 0x5F, 0xAA, 0x81],
            [0x81, 0xEF, 0x5F, 0xAA],
        ];

        for (b, want) in expected.iter().enumerate() {
            let mut output = input;
            rcircshift(&mut output, b);
            assert_eq!(&output, want, "shift {}", b);
        }
    }

    #[test]
    fn test_shift_edge_cases() {
        let mut empty: [u8; 0] = [];
        lshift(&mut empty, 3);
        rshift(&mut empty, 3);
        lcircshift(&mut empty, 3);
        rcircshift(&mut empty, 3);

        for b in [4usize, 5, 100] {
            let mut x = [0x81u8, 0xEF, 0x5F, 0xAA];
            lshift(&mut x, b);
            assert_eq!(x, [0, 0, 0, 0], "lshift {}", b);

            let mut x = [0x81u8, 0xEF, 0x5F, 0xAA];
            rshift(&mut x, b);
            assert_eq!(x, [0, 0, 0, 0], "rshift {}", b);
        }
    }

    #[test]
    fn test_circshift_wraps() {
        let input = [0x81u8, 0xEF, 0x5F, 0xAA];

        for b in 0..12usize {
            let mut left = input;
            lcircshift(&mut left, b);
            let mut right = input;
            rcircshift(&mut right, 4 - (b % 4));
            // shifting left by b and right by n-b land in the same place
            assert_eq!(left, right, "shift {}", b);

            // and the two directions undo each other
            let mut x = input;
            lcircshift(&mut x, b);
            rcircshift(&mut x, b);
            assert_eq!(x, input, "round trip {}", b);
        }
    }
}
