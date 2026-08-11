//
// interleaver
//
// Create and initialize interleaver objects
//

/// structured interleaver object
#[derive(Clone, Debug)]
pub struct Interleaver {
    n: usize,      // number of bytes
    rows: usize,   // row dimension
    cols: usize,   // col dimension
    depth: usize,  // interleaving depth (number of permutations)
}

impl Interleaver {
    /// create interleaver of length n input/output bytes
    pub fn new(n: usize) -> Self {
        // set internal properties
        let depth = 4; // default depth to maximum

        // compute block dimensions
        let rows = 1 + (n as f32).sqrt().floor() as usize;

        let mut cols = n / rows;
        while n >= rows * cols {
            cols += 1;
        } // ensures rows * cols >= n

        Self { n, rows, cols, depth }
    }

    /// set depth (number of internal iterations)
    pub fn set_depth(&mut self, depth: usize) {
        self.depth = depth;
    }

    /// get the interleaver length in bytes
    pub fn len(&self) -> usize {
        self.n
    }

    /// check if interleaver is empty
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// execute forward interleaver (encoder)
    ///
    ///  msg_dec    :   decoded (un-interleaved) message
    ///  msg_enc    :   encoded (interleaved) message
    pub fn encode(&self, msg_dec: &[u8], msg_enc: &mut [u8]) {
        // copy data to output
        msg_enc[..self.n].copy_from_slice(&msg_dec[..self.n]);

        if self.depth > 0 {
            permute(msg_enc, self.n, self.rows, self.cols);
        }
        if self.depth > 1 {
            permute_mask(msg_enc, self.n, self.rows, self.cols + 2, 0x0f);
        }
        if self.depth > 2 {
            permute_mask(msg_enc, self.n, self.rows, self.cols + 4, 0x55);
        }
        if self.depth > 3 {
            permute_mask(msg_enc, self.n, self.rows, self.cols + 8, 0x33);
        }
    }

    /// execute forward interleaver (encoder) on soft bits
    ///
    ///  msg_dec    :   decoded (un-interleaved) message
    ///  msg_enc    :   encoded (interleaved) message
    pub fn encode_soft(&self, msg_dec: &[u8], msg_enc: &mut [u8]) {
        // copy data to output
        msg_enc[..8 * self.n].copy_from_slice(&msg_dec[..8 * self.n]);

        if self.depth > 0 {
            permute_soft(msg_enc, self.n, self.rows, self.cols);
        }
        if self.depth > 1 {
            permute_mask_soft(msg_enc, self.n, self.rows, self.cols + 2, 0x0f);
        }
        if self.depth > 2 {
            permute_mask_soft(msg_enc, self.n, self.rows, self.cols + 4, 0x55);
        }
        if self.depth > 3 {
            permute_mask_soft(msg_enc, self.n, self.rows, self.cols + 8, 0x33);
        }
    }

    /// execute reverse interleaver (decoder)
    ///
    ///  msg_enc    :   encoded (interleaved) message
    ///  msg_dec    :   decoded (un-interleaved) message
    pub fn decode(&self, msg_enc: &[u8], msg_dec: &mut [u8]) {
        // copy data to output
        msg_dec[..self.n].copy_from_slice(&msg_enc[..self.n]);

        if self.depth > 3 {
            permute_mask(msg_dec, self.n, self.rows, self.cols + 8, 0x33);
        }
        if self.depth > 2 {
            permute_mask(msg_dec, self.n, self.rows, self.cols + 4, 0x55);
        }
        if self.depth > 1 {
            permute_mask(msg_dec, self.n, self.rows, self.cols + 2, 0x0f);
        }
        if self.depth > 0 {
            permute(msg_dec, self.n, self.rows, self.cols);
        }
    }

    /// execute reverse interleaver (decoder) on soft bits
    ///
    ///  msg_enc    :   encoded (interleaved) message
    ///  msg_dec    :   decoded (un-interleaved) message
    pub fn decode_soft(&self, msg_enc: &[u8], msg_dec: &mut [u8]) {
        // copy data to output
        msg_dec[..8 * self.n].copy_from_slice(&msg_enc[..8 * self.n]);

        if self.depth > 3 {
            permute_mask_soft(msg_dec, self.n, self.rows, self.cols + 8, 0x33);
        }
        if self.depth > 2 {
            permute_mask_soft(msg_dec, self.n, self.rows, self.cols + 4, 0x55);
        }
        if self.depth > 1 {
            permute_mask_soft(msg_dec, self.n, self.rows, self.cols + 2, 0x0f);
        }
        if self.depth > 0 {
            permute_soft(msg_dec, self.n, self.rows, self.cols);
        }
    }
}

//
// internal methods
//

// permute one iteration
fn permute(x: &mut [u8], n: usize, rows: usize, cols: usize) {
    let mut row = 0usize;
    let mut col = n / 3;
    let n2 = n / 2;

    for i in 0..n2 {
        let j;
        loop {
            let candidate = row * cols + col;
            row += 1;
            if row == rows {
                col = (col + 1) % cols;
                row = 0;
            }
            if candidate < n2 {
                j = candidate;
                break;
            }
        }

        // swap indices
        x.swap(2 * j + 1, 2 * i);
    }
}

// permute one iteration (soft bit input)
fn permute_soft(x: &mut [u8], n: usize, rows: usize, cols: usize) {
    let mut row = 0usize;
    let mut col = n / 3;
    let n2 = n / 2;

    for i in 0..n2 {
        let j;
        loop {
            let candidate = row * cols + col;
            row += 1;
            if row == rows {
                col = (col + 1) % cols;
                row = 0;
            }
            if candidate < n2 {
                j = candidate;
                break;
            }
        }

        // swap soft bits at indices (8 bytes each)
        let idx_i = 8 * (2 * i);
        let idx_j = 8 * (2 * j + 1);
        for k in 0..8 {
            x.swap(idx_i + k, idx_j + k);
        }
    }
}

/// Permute one iteration with mask
fn permute_mask(x: &mut [u8], n: usize, rows: usize, cols: usize, mask: u8) {
    let mut row = 0usize;
    let mut col = n / 3;
    let n2 = n / 2;

    for i in 0..n2 {
        let j;
        loop {
            let candidate = row * cols + col;
            row += 1;
            if row == rows {
                col = (col + 1) % cols;
                row = 0;
            }
            if candidate < n2 {
                j = candidate;
                break;
            }
        }

        // swap indices, applying mask
        let tmp0 = (x[2 * i] & !mask) | (x[2 * j + 1] & mask);
        let tmp1 = (x[2 * i] & mask) | (x[2 * j + 1] & !mask);
        x[2 * i] = tmp0;
        x[2 * j + 1] = tmp1;
    }
}

// permute one iteration (soft bit input) with mask
fn permute_mask_soft(x: &mut [u8], n: usize, rows: usize, cols: usize, mask: u8) {
    let mut row = 0usize;
    let mut col = n / 3;
    let n2 = n / 2;

    for i in 0..n2 {
        let j;
        loop {
            let candidate = row * cols + col;
            row += 1;
            if row == rows {
                col = (col + 1) % cols;
                row = 0;
            }
            if candidate < n2 {
                j = candidate;
                break;
            }
        }

        // swap bits matching the mask
        for k in 0..8 {
            if (mask >> (7 - k)) & 0x01 != 0 {
                x.swap(8 * (2 * j + 1) + k, 8 * (2 * i) + k);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    fn interleaver_test_hard(n: usize) {
        let mut rng = rand::thread_rng();

        let mut x = vec![0u8; n];
        let mut y = vec![0u8; n];
        let mut z = vec![0u8; n];

        for i in 0..n {
            x[i] = rng.gen::<u8>();
        }

        // create interleaver object
        let q = Interleaver::new(n);

        q.encode(&x[..n], &mut y[..n]);
        q.decode(&y[..n], &mut z[..n]);

        assert_eq!(&x[..], &z[..]);
    }

    fn interleaver_test_soft(n: usize) {
        let mut rng = rand::thread_rng();

        let mut x = vec![0u8; 8 * n];
        let mut y = vec![0u8; 8 * n];
        let mut z = vec![0u8; 8 * n];

        for i in 0..8 * n {
            x[i] = rng.gen::<u8>();
        }

        // create interleaver object
        let q = Interleaver::new(n);

        q.encode_soft(&x[..8 * n], &mut y[..8 * n]);
        q.decode_soft(&y[..8 * n], &mut z[..8 * n]);

        assert_eq!(&x[..], &z[..]);
    }

    #[test]
    #[autotest_annotate(autotest_interleaver_hard_8)]
    fn test_interleaver_hard_8() {
        interleaver_test_hard(8);
    }

    #[test]
    #[autotest_annotate(autotest_interleaver_hard_16)]
    fn test_interleaver_hard_16() {
        interleaver_test_hard(16);
    }

    #[test]
    #[autotest_annotate(autotest_interleaver_hard_64)]
    fn test_interleaver_hard_64() {
        interleaver_test_hard(64);
    }

    #[test]
    #[autotest_annotate(autotest_interleaver_hard_256)]
    fn test_interleaver_hard_256() {
        interleaver_test_hard(256);
    }

    #[test]
    #[autotest_annotate(autotest_interleaver_soft_8)]
    fn test_interleaver_soft_8() {
        interleaver_test_soft(8);
    }

    #[test]
    #[autotest_annotate(autotest_interleaver_soft_16)]
    fn test_interleaver_soft_16() {
        interleaver_test_soft(16);
    }

    #[test]
    #[autotest_annotate(autotest_interleaver_soft_64)]
    fn test_interleaver_soft_64() {
        interleaver_test_soft(64);
    }

    #[test]
    #[autotest_annotate(autotest_interleaver_soft_256)]
    fn test_interleaver_soft_256() {
        interleaver_test_soft(256);
    }

    #[test]
    fn test_interleaver_passthrough() {
        let mut rng = rand::thread_rng();
        const N: usize = 64;

        let mut x = vec![0u8; N];
        let mut y = vec![0u8; N];
        let mut z = vec![0u8; N];

        for i in 0..N {
            x[i] = rng.gen::<u8>();
        }

        // create interleaver object
        let mut q = Interleaver::new(N);
        q.set_depth(0);

        q.encode(&x[..N], &mut y[..N]);
        q.decode(&y[..N], &mut z[..N]);

        assert_eq!(&x[..], &y[..]);
        assert_eq!(&x[..], &z[..]);
        assert_eq!(&y[..], &z[..]);
    }

    #[test]
    fn test_interleaver_depths_hard() {
        for depth in 0..=4 {
            let mut rng = rand::thread_rng();
            const N: usize = 64;

            let mut x = vec![0u8; N];
            let mut y = vec![0u8; N];
            let mut z = vec![0u8; N];

            for i in 0..N {
                x[i] = rng.gen::<u8>();
            }

            // create interleaver object
            let mut q = Interleaver::new(N);
            q.set_depth(depth);

            q.encode(&x[..N], &mut y[..N]);
            q.decode(&y[..N], &mut z[..N]);

            assert_eq!(&x[..], &z[..]);
        }
    }

    #[test]
    fn test_interleaver_depths_soft() {
        for depth in 0..=4 {
            let mut rng = rand::thread_rng();
            const N: usize = 64;

            let mut x = vec![0u8; 8 * N];
            let mut y = vec![0u8; 8 * N];
            let mut z = vec![0u8; 8 * N];

            for i in 0..N {
                x[i] = rng.gen::<u8>();
            }

            // create interleaver object
            let mut q = Interleaver::new(N);
            q.set_depth(depth);

            q.encode_soft(&x[..8 * N], &mut y[..8 * N]);
            q.decode_soft(&y[..8 * N], &mut z[..8 * N]);

            assert_eq!(&x[..], &z[..]);
        }
    }

    #[test]
    fn test_interleaver_permutation_depth_1() {
        // interleave 0,1,2,... at depth 1
        let x: Vec<u8> = (0..8).collect();
        let mut y = vec![0u8; 8];

        let mut q = Interleaver::new(8);
        q.set_depth(1);
        q.encode(&x, &mut y);

        //  i  | j | swap
        // ----+---+-----------
        //  0  | 2 | x[5] <-> x[0]
        //  1  | 0 | x[1] <-> x[2]
        //  2  | 3 | x[7] <-> x[4]
        //  3  | 1 | x[3] <-> x[6]
        assert_eq!(y, [5, 2, 1, 6, 7, 0, 3, 4]);

        // demonstrate a longer run
        let x: Vec<u8> = (0..64).collect();
        let mut y = vec![0u8; 64];

        let mut q = Interleaver::new(64);
        q.set_depth(1);
        q.encode(&x, &mut y);

        assert_eq!(
            y,
            [
                43, 20, 59, 28, 13, 36, 29, 44, 45, 52, 61, 60, 15, 4, 31, 12, 47, 22, 63, 30, 1,
                38, 17, 46, 33, 54, 49, 62, 3, 6, 19, 14, 35, 24, 51, 32, 5, 40, 21, 48, 37, 56,
                53, 0, 7, 8, 23, 16, 39, 26, 55, 34, 9, 42, 25, 50, 41, 58, 57, 2, 11, 10, 27, 18
            ]
        );
    }

    #[test]
    fn test_interleaver_permutation_all_depths() {
        // repeat the same 4 bits on top and on bottom so that the effect of depth 2 shows up in hex
        let x: Vec<u8> = (0..16).map(|i| (i << 4) | i).collect();
        let mut y = vec![0u8; 16];

        let mut q = Interleaver::new(16);
        q.set_depth(0);
        q.encode(&x, &mut y);

        // depth 0 (same as x)
        assert_eq!(y, [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]);

        q.set_depth(1);
        q.encode(&x, &mut y);

        // depth 1 interleaving
        assert_eq!(y, [0xbb, 0xaa, 0x55, 0xee, 0xdd, 0x22, 0x77, 0x66, 0xff, 0xcc, 0x11, 0x00, 0x99, 0x44, 0x33, 0x88]);

        q.set_depth(2);
        q.encode(&x, &mut y);

        // depth 2 interleaving. top bits are the same as before but the bottom 4 bits are scrambled.
        assert_eq!(y, [0xb0, 0xa5, 0x5a, 0xe7, 0xd4, 0x21, 0x7e, 0x69, 0xf8, 0xc3, 0x12, 0x0b, 0x96, 0x4d, 0x3c, 0x8f]);

        q.set_depth(3);
        q.encode(&x, &mut y);

        // depth 3 interleaves with mask 0x55, so the pattern is less obvious
        assert_eq!(y, [0xa1, 0xf4, 0x4f, 0xf2, 0x85, 0x30, 0x2f, 0x3c, 0xed, 0x96, 0x03, 0x1a, 0xc3, 0x58, 0x69, 0xde]);

        q.set_depth(4);
        q.encode(&x, &mut y);

        // full interleaving (depth 4). last stage uses mask 0x33
        assert_eq!(y, [0x92, 0xe7, 0x5c, 0xe1, 0x96, 0x03, 0x3c, 0x0f, 0xfe, 0xa5, 0x30, 0x29, 0xf0, 0x4b, 0x5a, 0xcd]);
    }


    #[test]
    fn test_interleaver_permutation_is_bijection() {
        for n in [2usize, 8, 15, 16, 63, 64, 100, 256] {
            let x: Vec<u8> = (0..n).map(|i| i as u8).collect();
            let mut y = vec![0u8; n];

            let mut q = Interleaver::new(n);
            q.set_depth(1);
            q.encode(&x, &mut y);

            // all indices present exactly once (n <= 256 keeps these distinct)
            let mut sorted = y.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, x, "n = {}", n);

            // index parity always flips, for each complete pair
            for i in 0..n / 2 {
                assert_eq!(y[2 * i] % 2, 1, "n = {}, even slot {}", n, 2 * i);
                assert_eq!(y[2 * i + 1] % 2, 0, "n = {}, odd slot {}", n, 2 * i + 1);
            }
        }
    }
}
