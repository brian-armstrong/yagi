//
// interleaver
//
// Create and initialize interleaver objects
//

/// structured interleaver object
#[derive(Clone, Debug)]
pub struct Interleaver {
    n: usize,     // number of bytes
    m: usize,     // row dimension
    nn: usize,    // col dimension (called N in liquid, but n is taken)
    depth: usize, // interleaving depth (number of permutations)
}

impl Interleaver {
    /// create interleaver of length n input/output bytes
    pub fn new(n: usize) -> Self {
        // set internal properties
        let depth = 4; // default depth to maximum

        // compute block dimensions
        let m = 1 + (n as f32).sqrt().floor() as usize;

        let mut nn = n / m;
        while n >= m * nn {
            nn += 1;
        } // ensures M*N >= n

        Self { n, m, nn, depth }
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
            permute(msg_enc, self.n, self.m, self.nn);
        }
        if self.depth > 1 {
            permute_mask(msg_enc, self.n, self.m, self.nn + 2, 0x0f);
        }
        if self.depth > 2 {
            permute_mask(msg_enc, self.n, self.m, self.nn + 4, 0x55);
        }
        if self.depth > 3 {
            permute_mask(msg_enc, self.n, self.m, self.nn + 8, 0x33);
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
            permute_soft(msg_enc, self.n, self.m, self.nn);
        }
        if self.depth > 1 {
            permute_mask_soft(msg_enc, self.n, self.m, self.nn + 2, 0x0f);
        }
        if self.depth > 2 {
            permute_mask_soft(msg_enc, self.n, self.m, self.nn + 4, 0x55);
        }
        if self.depth > 3 {
            permute_mask_soft(msg_enc, self.n, self.m, self.nn + 8, 0x33);
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
            permute_mask(msg_dec, self.n, self.m, self.nn + 8, 0x33);
        }
        if self.depth > 2 {
            permute_mask(msg_dec, self.n, self.m, self.nn + 4, 0x55);
        }
        if self.depth > 1 {
            permute_mask(msg_dec, self.n, self.m, self.nn + 2, 0x0f);
        }
        if self.depth > 0 {
            permute(msg_dec, self.n, self.m, self.nn);
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
            permute_mask_soft(msg_dec, self.n, self.m, self.nn + 8, 0x33);
        }
        if self.depth > 2 {
            permute_mask_soft(msg_dec, self.n, self.m, self.nn + 4, 0x55);
        }
        if self.depth > 1 {
            permute_mask_soft(msg_dec, self.n, self.m, self.nn + 2, 0x0f);
        }
        if self.depth > 0 {
            permute_soft(msg_dec, self.n, self.m, self.nn);
        }
    }
}

//
// internal methods
//

// permute one iteration
fn permute(x: &mut [u8], n: usize, m: usize, nn: usize) {
    let mut mm = 0usize;
    let mut col = n / 3;
    let n2 = n / 2;

    for i in 0..n2 {
        let j;
        loop {
            let candidate = mm * nn + col;
            mm += 1;
            if mm == m {
                col = (col + 1) % nn;
                mm = 0;
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
fn permute_soft(x: &mut [u8], n: usize, m: usize, nn: usize) {
    let mut mm = 0usize;
    let mut col = n / 3;
    let n2 = n / 2;

    for i in 0..n2 {
        let j;
        loop {
            let candidate = mm * nn + col;
            mm += 1;
            if mm == m {
                col = (col + 1) % nn;
                mm = 0;
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
fn permute_mask(x: &mut [u8], n: usize, m: usize, nn: usize, mask: u8) {
    let mut mm = 0usize;
    let mut col = n / 3;
    let n2 = n / 2;

    for i in 0..n2 {
        let j;
        loop {
            let candidate = mm * nn + col;
            mm += 1;
            if mm == m {
                col = (col + 1) % nn;
                mm = 0;
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
fn permute_mask_soft(x: &mut [u8], n: usize, m: usize, nn: usize, mask: u8) {
    let mut mm = 0usize;
    let mut col = n / 3;
    let n2 = n / 2;

    for i in 0..n2 {
        let j;
        loop {
            let candidate = mm * nn + col;
            mm += 1;
            if mm == m {
                col = (col + 1) % nn;
                mm = 0;
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
}
