use crate::error::{Error, Result};
use crate::utility::bits::count_ones;
use crate::sequence::msequence::MSequence;


/// binary sequence
#[derive(Debug, Clone)]
pub struct BSequence {
    s: Vec<u32>,
    num_bits: usize,
    num_bits_msb: usize,
    bit_mask_msb: u32,
}

impl BSequence {
    pub fn new(num_bits: usize) -> Self {
        // round up to the nearest 32 (number of bits in a u32)
        let s_len = (num_bits + 31) / 32;
        let num_bits_msb = if num_bits % 32 == 0 { 32 } else { num_bits % 32 };
        let bit_mask_msb = 1u32.checked_shl(num_bits_msb as u32).unwrap_or(0).wrapping_sub(1);

        let mut bs = Self {
            s: vec![0; s_len],
            num_bits,
            num_bits_msb,
            bit_mask_msb,
        };
        bs.reset();
        bs
    }

    // initialize two sequences to complementary codes.  sequences must
    // be of length at least 8 and a power of 2 (e.g. 8, 16, 32, 64,...)
    pub fn create_ccodes(qa: &mut BSequence, qb: &mut BSequence) -> Result<()> {
        if qa.num_bits != qb.num_bits {
            return Err(Error::Config("sequence lengths must match".into()));
        }
        if qa.num_bits < 8 {
            return Err(Error::Config("sequence too short".into()));
        }
        if qa.num_bits % 8 != 0 {
            return Err(Error::Config("sequence must be multiple of 8".into()));
        }

        let num_bytes = qa.num_bits / 8;
        let mut a = vec![0u8; num_bytes];
        let mut b = vec![0u8; num_bytes];

        a[num_bytes - 1] = 0xb8;  // 1011 1000
        b[num_bytes - 1] = 0xb7;  // 1011 0111

        let mut n = 1;
        while n < num_bytes {
            let i_n1 = num_bytes - n;
            let i_n0 = num_bytes - 2 * n;

            // a -> [a  b]
            // b -> [a ~b]
            let (a_start, a_end) = a.split_at_mut(i_n1);
            let (b_start, b_end) = b.split_at_mut(i_n1);
            a_start[i_n0..i_n1].copy_from_slice(&a_end[0..n]);
            b_start[i_n0..i_n1].copy_from_slice(&a_end[0..n]);

            a_end[0..n].copy_from_slice(&b_end[0..n]);
            // skip a copy from b to b (src = dst)
            // b_end[0..n].copy_from_slice(&b_end[0..n]);

            for i in 0..n {
                b[num_bytes - i - 1] ^= 0xff;
            }

            n *= 2;
        }

        qa.init(&a);
        qb.init(&b);

        Ok(())
    }

    pub fn from_msequence(ms: &mut MSequence) -> Result<Self> {
        let mut bs = BSequence::new(ms.get_length() as usize);
        bs.reset();
        for _ in 0..ms.get_length() {
            bs.push(ms.advance());
        }
        Ok(bs)
    }

    pub fn reset(&mut self) {
        self.s.fill(0);
    }

    // initialize sequence on external array
    pub fn init(&mut self, v: &[u8]) {
        let mut k = 0;
        let mut byte = 0;
        let mut mask = 0x80;
        for i in 0..self.num_bits {
            if (i % 8) == 0 {
                byte = v[k];
                k += 1;
                mask = 0x80;
            }
            self.push(if (byte & mask) != 0 { 1 } else { 0 });
            mask >>= 1;
        }
    }

    pub fn print(&self) {
        println!("<bsequence, bits={}>", self.num_bits);
    }

    // push bits in from the right
    pub fn push(&mut self, bit: u32) {
        let p = 32;
        self.s[0] = (self.s[0] << 1) & self.bit_mask_msb;

        for i in 1..self.s.len() {
            let overflow = (self.s[i] >> (p - 1)) & 1;
            self.s[i] <<= 1;
            self.s[i - 1] |= overflow;
        }

        let l = self.s.len();
        self.s[l - 1] |= bit & 1;
    }

    // circular shift (left)
    pub fn circshift(&mut self) {
        let msb_mask = 1u32.checked_shl(self.num_bits_msb as u32 - 1).unwrap_or(0);
        let b = (self.s[0] & msb_mask) >> (self.num_bits_msb - 1);
        self.push(b);
    }

    // Correlate two binary sequences together
    pub fn correlate(&self, bs2: &BSequence) -> Result<i32> {
        if self.s.len() != bs2.s.len() {
            return Err(Error::Config("binary sequences must be the same length".into()));
        }

        let mut rxy = 0;
        for (a, b) in self.s.iter().zip(bs2.s.iter()) {
            let chunk = !(*a ^ *b);
            rxy += count_ones(chunk) as i32;
        }

        rxy -= 32 - self.num_bits_msb as i32;
        Ok(rxy)
    }

    // compute the binary addition of two bit sequences
    pub fn add(&self, bs2: &BSequence, bs3: &mut BSequence) -> Result<()> {
        if self.s.len() != bs2.s.len() || self.s.len() != bs3.s.len() {
            return Err(Error::Config("binary sequences must be same length".into()));
        }

        for ((a, b), c) in self.s.iter().zip(bs2.s.iter()).zip(bs3.s.iter_mut()) {
            *c = *a ^ *b;
        }

        Ok(())
    }

    // compute the binary multiplication of two bit sequences
    pub fn mul(&self, bs2: &BSequence, bs3: &mut BSequence) -> Result<()> {
        if self.s.len() != bs2.s.len() || self.s.len() != bs3.s.len() {
            return Err(Error::Config("binary sequences must be same length".into()));
        }

        for ((a, b), c) in self.s.iter().zip(bs2.s.iter()).zip(bs3.s.iter_mut()) {
            *c = *a & *b;
        }

        Ok(())
    }

    // accumulate the 1's in a binary sequence
    pub fn accumulate(&self) -> u32 {
        self.s.iter().map(|&x| count_ones(x)).sum()
    }

    pub fn get_length(&self) -> usize {
        self.num_bits
    }

    // return the i-th bit of the sequence
    pub fn index(&self, i: usize) -> Result<u32> {
        if i >= self.num_bits {
            return Err(Error::Config(format!("invalid index {}", i)));
        }
        let k = self.s.len() - 1 - i / 32;
        Ok((self.s[k] >> (i % 32)) & 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    use crate::sequence::msequence::MSequence;

    #[test]
    #[autotest_annotate(autotest_bsequence_init)]
    fn test_bsequence_init() {
        // 1111 0000 1100 1010
        let v = [0xf0u8, 0xcau8];
    
        // create and initialize sequence
        let mut q = BSequence::new(16);
        q.init(&v);
    
        // run tests
        assert_eq!(q.index(15).unwrap(), 1);
        assert_eq!(q.index(14).unwrap(), 1);
        assert_eq!(q.index(13).unwrap(), 1);
        assert_eq!(q.index(12).unwrap(), 1);
        
        assert_eq!(q.index(11).unwrap(), 0);
        assert_eq!(q.index(10).unwrap(), 0);
        assert_eq!(q.index(9).unwrap(), 0);
        assert_eq!(q.index(8).unwrap(), 0);
        
        assert_eq!(q.index(7).unwrap(), 1);
        assert_eq!(q.index(6).unwrap(), 1);
        assert_eq!(q.index(5).unwrap(), 0);
        assert_eq!(q.index(4).unwrap(), 0);
        
        assert_eq!(q.index(3).unwrap(), 1);
        assert_eq!(q.index(2).unwrap(), 0);
        assert_eq!(q.index(1).unwrap(), 1);
        assert_eq!(q.index(0).unwrap(), 0);
    }

    #[test]
    #[autotest_annotate(autotest_bsequence_init_msequence)]
    fn test_bsequence_init_msequence() {
        // create and initialize m-sequence
        let mut ms = MSequence::create_default(4).unwrap();

        // create and initialize binary sequence on m-sequence
        let bs = BSequence::from_msequence(&mut ms).unwrap();

        assert_eq!(bs.get_length(), ms.get_length() as usize);
    }

    
    #[test]
    #[autotest_annotate(autotest_bsequence_correlate)]
    fn test_bsequence_correlate() {
        // v0   :   1111 0000 1100 1010
        // v1   :   1100 1011 0001 1110
        // sim  :   1100 0100 0010 1011 (7 similar bits)
        let v0 = [0xf0u8, 0xcau8];
        let v1 = [0xcbu8, 0x1eu8];
    
        // create and initialize sequences
        let mut q0 = BSequence::new(16);
        let mut q1 = BSequence::new(16);
        q0.init(&v0);
        q1.init(&v1);
    
        // run tests
        assert_eq!(q0.correlate(&q1).unwrap(), 7);
    }
    
    #[test]
    #[autotest_annotate(autotest_bsequence_add)]
    fn test_bsequence_add() {
        // v0   :   1111 0000 1100 1010
        // v1   :   1100 1011 0001 1110
        // sum  :   0011 1011 1101 0100
        let v0 = [0xf0u8, 0xcau8];
        let v1 = [0xcbu8, 0x1eu8];
    
        // create and initialize sequences
        let mut q0 = BSequence::new(16);
        let mut q1 = BSequence::new(16);
        q0.init(&v0);
        q1.init(&v1);
    
        // create result sequence
        let mut r = BSequence::new(16);
        q0.add(&q1, &mut r).unwrap();
    
        // run tests
        assert_eq!(r.index(15).unwrap(), 0);
        assert_eq!(r.index(14).unwrap(), 0);
        assert_eq!(r.index(13).unwrap(), 1);
        assert_eq!(r.index(12).unwrap(), 1);
        
        assert_eq!(r.index(11).unwrap(), 1);
        assert_eq!(r.index(10).unwrap(), 0);
        assert_eq!(r.index(9).unwrap(), 1);
        assert_eq!(r.index(8).unwrap(), 1);
        
        assert_eq!(r.index(7).unwrap(), 1);
        assert_eq!(r.index(6).unwrap(), 1);
        assert_eq!(r.index(5).unwrap(), 0);
        assert_eq!(r.index(4).unwrap(), 1);
        
        assert_eq!(r.index(3).unwrap(), 0);
        assert_eq!(r.index(2).unwrap(), 1);
        assert_eq!(r.index(1).unwrap(), 0);
        assert_eq!(r.index(0).unwrap(), 0);
    }
    
    #[test]
    #[autotest_annotate(autotest_bsequence_mul)]
    fn test_bsequence_mul() {
        // v0   :   1111 0000 1100 1010
        // v1   :   1100 1011 0001 1110
        // prod :   1100 0000 0000 1010
        let v0 = [0xf0u8, 0xcau8];
        let v1 = [0xcbu8, 0x1eu8];
    
        // create and initialize sequences
        let mut q0 = BSequence::new(16);
        let mut q1 = BSequence::new(16);
        q0.init(&v0);
        q1.init(&v1);
    
        // create result sequence
        let mut r = BSequence::new(16);
        q0.mul(&q1, &mut r).unwrap();
    
        // run tests
        assert_eq!(r.index(15).unwrap(), 1);
        assert_eq!(r.index(14).unwrap(), 1);
        assert_eq!(r.index(13).unwrap(), 0);
        assert_eq!(r.index(12).unwrap(), 0);
        
        assert_eq!(r.index(11).unwrap(), 0);
        assert_eq!(r.index(10).unwrap(), 0);
        assert_eq!(r.index(9).unwrap(), 0);
        assert_eq!(r.index(8).unwrap(), 0);
        
        assert_eq!(r.index(7).unwrap(), 0);
        assert_eq!(r.index(6).unwrap(), 0);
        assert_eq!(r.index(5).unwrap(), 0);
        assert_eq!(r.index(4).unwrap(), 0);
        
        assert_eq!(r.index(3).unwrap(), 1);
        assert_eq!(r.index(2).unwrap(), 0);
        assert_eq!(r.index(1).unwrap(), 1);
        assert_eq!(r.index(0).unwrap(), 0);
    }
    
    #[test]
    #[autotest_annotate(autotest_bsequence_accumulate)]
    fn test_bsequence_accumulate() {
        // 1111 0000 1100 1010 (8 total bits)
        let v = [0xf0u8, 0xcau8];
    
        // create and initialize sequence
        let mut q = BSequence::new(16);
        q.init(&v);
    
        // run tests
        assert_eq!(q.accumulate(), 8);
    }


    fn complementary_codes_test(n: usize) {
        // create and initialize codes
        let mut a = BSequence::new(n);
        let mut b = BSequence::new(n);
        BSequence::create_ccodes(&mut a, &mut b).unwrap();

        // generate test sequences
        let mut ax = BSequence::new(n);
        let mut bx = BSequence::new(n);
        BSequence::create_ccodes(&mut ax, &mut bx).unwrap();

        for i in 0..n {
            // correlate like sequences
            let raa = 2 * a.correlate(&ax).unwrap() - n as i32;
            let rbb = 2 * b.correlate(&bx).unwrap() - n as i32;

            if i == 0 {
                assert_eq!(raa + rbb, 2 * n as i32);
            } else {
                assert_eq!(raa + rbb, 0);
            }

            ax.circshift();
            bx.circshift();
        }
    }

    #[test]
    #[autotest_annotate(autotest_complementary_code_n8)]
    fn test_complementary_code_n8() { complementary_codes_test(8); }

    #[test]
    #[autotest_annotate(autotest_complementary_code_n16)]
    fn test_complementary_code_n16() { complementary_codes_test(16); }

    #[test]
    #[autotest_annotate(autotest_complementary_code_n32)]
    fn test_complementary_code_n32() { complementary_codes_test(32); }

    #[test]
    #[autotest_annotate(autotest_complementary_code_n64)]
    fn test_complementary_code_n64() { complementary_codes_test(64); }

    #[test]
    #[autotest_annotate(autotest_complementary_code_n128)]
    fn test_complementary_code_n128() { complementary_codes_test(128); }

    #[test]
    #[autotest_annotate(autotest_complementary_code_n256)]
    fn test_complementary_code_n256() { complementary_codes_test(256); }

    #[test]
    #[autotest_annotate(autotest_complementary_code_n512)]
    fn test_complementary_code_n512() { complementary_codes_test(512); }
}