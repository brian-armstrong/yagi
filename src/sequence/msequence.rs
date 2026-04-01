use crate::error::{Error, Result};
use crate::utility::bits::{bdotprod, msb_index};

const MIN_MSEQUENCE_M: u32 = 2;
const MAX_MSEQUENCE_M: u32 = 31;

// default m-sequence generators:       g (hex)            m   n
const MSEQUENCE_GENPOLY_M2: u32 = 0x00000003;   //  2   3
const MSEQUENCE_GENPOLY_M3: u32 = 0x00000006;   //  3   7
const MSEQUENCE_GENPOLY_M4: u32 = 0x0000000c;   //  4   15
const MSEQUENCE_GENPOLY_M5: u32 = 0x00000014;   //  5   31
const MSEQUENCE_GENPOLY_M6: u32 = 0x00000030;   //  6   63
const MSEQUENCE_GENPOLY_M7: u32 = 0x00000060;   //  7   127
const MSEQUENCE_GENPOLY_M8: u32 = 0x000000b8;   //  8   255
const MSEQUENCE_GENPOLY_M9: u32 = 0x00000110;   //  9   511
const MSEQUENCE_GENPOLY_M10: u32 = 0x00000240;  // 10   1,023
const MSEQUENCE_GENPOLY_M11: u32 = 0x00000500;  // 11   2,047
const MSEQUENCE_GENPOLY_M12: u32 = 0x00000e08;  // 12   4,095
const MSEQUENCE_GENPOLY_M13: u32 = 0x00001c80;  // 13   8,191
const MSEQUENCE_GENPOLY_M14: u32 = 0x00003802;  // 14   16,383
const MSEQUENCE_GENPOLY_M15: u32 = 0x00006000;  // 15   32,767
const MSEQUENCE_GENPOLY_M16: u32 = 0x0000d008;  // 16   65,535
const MSEQUENCE_GENPOLY_M17: u32 = 0x00012000;  // 17   131,071
const MSEQUENCE_GENPOLY_M18: u32 = 0x00020400;  // 18   262,143
const MSEQUENCE_GENPOLY_M19: u32 = 0x00072000;  // 19   524,287
const MSEQUENCE_GENPOLY_M20: u32 = 0x00090000;  // 20   1,048,575
const MSEQUENCE_GENPOLY_M21: u32 = 0x00140000;  // 21   2,097,151
const MSEQUENCE_GENPOLY_M22: u32 = 0x00300000;  // 22   4,194,303
const MSEQUENCE_GENPOLY_M23: u32 = 0x00420000;  // 23   8,388,607
const MSEQUENCE_GENPOLY_M24: u32 = 0x00e10000;  // 24   16,777,215
const MSEQUENCE_GENPOLY_M25: u32 = 0x01000004;  // 25   33,554,431
const MSEQUENCE_GENPOLY_M26: u32 = 0x02000023;  // 26   67,108,863
const MSEQUENCE_GENPOLY_M27: u32 = 0x04000013;  // 27   134,217,727
const MSEQUENCE_GENPOLY_M28: u32 = 0x08000004;  // 28   268,435,455
const MSEQUENCE_GENPOLY_M29: u32 = 0x10000002;  // 29   536,870,911
const MSEQUENCE_GENPOLY_M30: u32 = 0x20000029;  // 30   1,073,741,823
const MSEQUENCE_GENPOLY_M31: u32 = 0x40000004;  // 31   2,147,483,647

/// maximal-length sequence
#[derive(Debug, Clone, Copy)]
pub struct MSequence {
    m: u32,     // length generator polynomial, shift register
    g: u32,     // generator polynomial, form: { x^m + ... + 1 }
    a: u32,     // initial shift register state, default: 1
    n: u32,     // length of sequence, n = (2^m)-1
    state: u32, // shift register
}

impl MSequence {
    // create a maximal-length sequence (m-sequence) object with
    // an internal shift register length of _m bits.
    //  m      :   generator polynomial length, sequence length is (2^m)-1
    //  g      :   generator polynomial, starting with most-significant bit
    //  a      :   initial shift register state, default: 000...001
    pub fn new(m: u32, g: u32, a: u32) -> Result<Self> {
        if m > MAX_MSEQUENCE_M || m < MIN_MSEQUENCE_M {
            return Err(Error::Config(format!("m ({}) not in range", m)));
        }

        Ok(Self {
            m,
            g,
            a,
            n: (1 << m) - 1,
            state: a,
        })
    }

    pub fn create_genpoly(g: u32) -> Result<Self> {
        let t = msb_index(g);
        if t < 2 {
            return Err(Error::Config(format!("invalid generator polynomial: 0x{:x}", g)));
        }
        let m = t;
        let a = 1;
        Self::new(m, g, a)
    }

    pub fn create_default(m: u32) -> Result<Self> {
        let g = match m {
            2 => MSEQUENCE_GENPOLY_M2,
            3 => MSEQUENCE_GENPOLY_M3,
            4 => MSEQUENCE_GENPOLY_M4,
            5 => MSEQUENCE_GENPOLY_M5,
            6 => MSEQUENCE_GENPOLY_M6,
            7 => MSEQUENCE_GENPOLY_M7,
            8 => MSEQUENCE_GENPOLY_M8,
            9 => MSEQUENCE_GENPOLY_M9,
            10 => MSEQUENCE_GENPOLY_M10,
            11 => MSEQUENCE_GENPOLY_M11,
            12 => MSEQUENCE_GENPOLY_M12,
            13 => MSEQUENCE_GENPOLY_M13,
            14 => MSEQUENCE_GENPOLY_M14,
            15 => MSEQUENCE_GENPOLY_M15,
            16 => MSEQUENCE_GENPOLY_M16,
            17 => MSEQUENCE_GENPOLY_M17,
            18 => MSEQUENCE_GENPOLY_M18,
            19 => MSEQUENCE_GENPOLY_M19,
            20 => MSEQUENCE_GENPOLY_M20,
            21 => MSEQUENCE_GENPOLY_M21, 
            22 => MSEQUENCE_GENPOLY_M22,
            23 => MSEQUENCE_GENPOLY_M23,
            24 => MSEQUENCE_GENPOLY_M24,
            25 => MSEQUENCE_GENPOLY_M25,
            26 => MSEQUENCE_GENPOLY_M26,
            27 => MSEQUENCE_GENPOLY_M27,
            28 => MSEQUENCE_GENPOLY_M28,
            29 => MSEQUENCE_GENPOLY_M29,
            30 => MSEQUENCE_GENPOLY_M30,
            31 => MSEQUENCE_GENPOLY_M31,
            _ => return Err(Error::Config(format!("m ({}) not in range", m))),
        };
        Self::create_genpoly(g)
    }

    pub fn advance(&mut self) -> u32 {
        let b = bdotprod(self.state, self.g);
        self.state <<= 1;
        self.state |= b;
        self.state &= self.n;
        b
    }

    pub fn generate_symbol(&mut self, bps: u32) -> u32 {
        let mut s = 0;
        for _ in 0..bps {
            s <<= 1;
            s |= self.advance();
        }
        s
    }

    pub fn reset(&mut self) {
        self.state = self.a;
    }

    // Getter methods
    pub fn get_genpoly_length(&self) -> u32 { self.m }
    pub fn get_length(&self) -> u32 { self.n }
    pub fn get_genpoly(&self) -> u32 { self.g }
    pub fn get_state(&self) -> u32 { self.state }

    pub fn set_state(&mut self, a: u32) {
        self.state = a;
    }

    pub fn measure_period(&mut self) -> u32 {
        let s = self.get_state();
        let mut period = 0;
        for _ in 0..=self.n {
            self.advance();
            period += 1;
            if self.get_state() == s {
                break;
            }
        }
        period
    }

    pub fn genpoly_period(g: u32) -> Result<u32> {
        let mut q = MSequence::create_genpoly(g)?;
        Ok(q.measure_period())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use test_macro::autotest_annotate;

    use crate::sequence::bsequence::BSequence;

    fn msequence_test_autocorrelation(m: u32) {
        // create and initialize m-sequence
        let mut ms = MSequence::create_default(m).unwrap();
        let n = ms.get_length();

        // create and initialize first binary sequence on m-sequence
        let bs1 = BSequence::from_msequence(&mut ms).unwrap();

        // create and initialize second binary sequence on same m-sequence
        let mut bs2 = BSequence::from_msequence(&mut ms).unwrap();

        // ensure sequences are the same length
        assert_eq!(bs1.get_length(), n as usize);
        assert_eq!(bs2.get_length(), n as usize);

        // when sequences are aligned, autocorrelation is equal to length
        let mut rxy = bs1.correlate(&bs2).unwrap();
        assert_eq!(rxy, n as i32);

        // when sequences are misaligned, autocorrelation is equal to -1
        for _ in 0..n-1 {
            bs2.push(ms.advance());
            rxy = 2 * bs1.correlate(&bs2).unwrap() - n as i32;
            assert_eq!(rxy, -1);
        }
    }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m2)]
    fn test_msequence_xcorr_m2() { msequence_test_autocorrelation(2); }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m3)]
    fn test_msequence_xcorr_m3() { msequence_test_autocorrelation(3); }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m4)]
    fn test_msequence_xcorr_m4() { msequence_test_autocorrelation(4); }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m5)]
    fn test_msequence_xcorr_m5() { msequence_test_autocorrelation(5); }

    #[test] 
    #[autotest_annotate(autotest_msequence_xcorr_m6)]
    fn test_msequence_xcorr_m6() { msequence_test_autocorrelation(6); }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m7)]
    fn test_msequence_xcorr_m7() { msequence_test_autocorrelation(7); }
    
    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m8)]
    fn test_msequence_xcorr_m8() { msequence_test_autocorrelation(8); }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m9)]
    fn test_msequence_xcorr_m9() { msequence_test_autocorrelation(9); }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m10)]
    fn test_msequence_xcorr_m10() { msequence_test_autocorrelation(10); }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m11)]
    fn test_msequence_xcorr_m11() { msequence_test_autocorrelation(11); }

    #[test]
    #[autotest_annotate(autotest_msequence_xcorr_m12)]
    fn test_msequence_xcorr_m12() { msequence_test_autocorrelation(12); }

    fn msequence_test_period(m: u32) {
        // create and initialize m-sequence
        let mut q = MSequence::create_default(m).unwrap();

        // measure period and compare to expected
        let n = (1u32 << m) - 1;
        let p = q.measure_period();
        assert_eq!(p, n);
    }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m2)]
    fn test_msequence_period_m2() { msequence_test_period(2); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m3)]
    fn test_msequence_period_m3() { msequence_test_period(3); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m4)]
    fn test_msequence_period_m4() { msequence_test_period(4); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m5)]
    fn test_msequence_period_m5() { msequence_test_period(5); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m6)]
    fn test_msequence_period_m6() { msequence_test_period(6); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m7)]
    fn test_msequence_period_m7() { msequence_test_period(7); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m8)]
    fn test_msequence_period_m8() { msequence_test_period(8); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m9)]
    fn test_msequence_period_m9() { msequence_test_period(9); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m10)]
    fn test_msequence_period_m10() { msequence_test_period(10); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m11)]
    fn test_msequence_period_m11() { msequence_test_period(11); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m12)]
    fn test_msequence_period_m12() { msequence_test_period(12); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m13)]
    fn test_msequence_period_m13() { msequence_test_period(13); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m14)]
    fn test_msequence_period_m14() { msequence_test_period(14); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m15)]
    fn test_msequence_period_m15() { msequence_test_period(15); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m16)]
    fn test_msequence_period_m16() { msequence_test_period(16); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m17)]
    fn test_msequence_period_m17() { msequence_test_period(17); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m18)]
    fn test_msequence_period_m18() { msequence_test_period(18); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m19)]
    fn test_msequence_period_m19() { msequence_test_period(19); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m20)]
    fn test_msequence_period_m20() { msequence_test_period(20); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m21)]
    fn test_msequence_period_m21() { msequence_test_period(21); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m22)]
    fn test_msequence_period_m22() { msequence_test_period(22); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m23)]
    fn test_msequence_period_m23() { msequence_test_period(23); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m24)]
    fn test_msequence_period_m24() { msequence_test_period(24); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m25)]
    fn test_msequence_period_m25() { msequence_test_period(25); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m26)]
    fn test_msequence_period_m26() { msequence_test_period(26); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m27)]
    fn test_msequence_period_m27() { msequence_test_period(27); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m28)]
    fn test_msequence_period_m28() { msequence_test_period(28); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m29)]
    fn test_msequence_period_m29() { msequence_test_period(29); }
        
    #[test]
    #[autotest_annotate(autotest_msequence_period_m30)]
    fn test_msequence_period_m30() { msequence_test_period(30); }

    #[test]
    #[autotest_annotate(autotest_msequence_period_m31)]
    fn test_msequence_period_m31() { msequence_test_period(31); }

    #[test]
    #[autotest_annotate(autotest_msequence_config)]
    fn test_msequence_config() {
        // check invalid configurations
        assert!(MSequence::new(100, 0, 0).is_err());
        assert!(MSequence::create_default(32).is_err()); // too long
        assert!(MSequence::create_genpoly(0).is_err());

        // create proper object and test configurations
        let mut q = MSequence::create_genpoly(MSEQUENCE_GENPOLY_M11).unwrap();

        assert_eq!(q.get_state(), 1);
        q.set_state(0x8a);
        assert_eq!(q.get_state(), 0x8a);
    }
}