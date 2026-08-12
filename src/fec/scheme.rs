//
// forward error-correction scheme
//
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FecScheme {
    /// unknown/unsupported scheme
    Unknown,
    /// no error-correction
    None,
    /// simple repeat code, r1/3
    Rep3,
    /// simple repeat code, r1/5
    Rep5,
    /// Hamming (7,4) block code, r1/2 (really 4/7)
    Hamming74,
    /// Hamming (7,4) with extra parity bit, r1/2
    Hamming84,
    /// Hamming (12,8) block code, r2/3
    Hamming128,
    /// Golay (24,12) block code, r1/2
    Golay2412,
    /// SEC-DED (22,16) block code, r8/11
    Secded2216,
    /// SEC-DED (39,32) block code
    Secded3932,
    /// SEC-DED (72,64) block code, r8/9
    Secded7264,
    // codecs not defined internally (see http://www.ka9q.net/code/fec/)
    /// r1/2, K=7, dfree=10
    ConvV27,
    /// r1/2, K=9, dfree=12
    ConvV29,
    /// r1/3, K=9, dfree=18
    ConvV39,
    /// r1/6, K=15, dfree<=57 (Heller 1968)
    ConvV615,
    // punctured (perforated) codes
    /// r2/3, K=7, dfree=6
    ConvV27P23,
    /// r3/4, K=7, dfree=5
    ConvV27P34,
    /// r4/5, K=7, dfree=4
    ConvV27P45,
    /// r5/6, K=7, dfree=4
    ConvV27P56,
    /// r6/7, K=7, dfree=3
    ConvV27P67,
    /// r7/8, K=7, dfree=3
    ConvV27P78,
    /// r2/3, K=9, dfree=7
    ConvV29P23,
    /// r3/4, K=9, dfree=6
    ConvV29P34,
    /// r4/5, K=9, dfree=5
    ConvV29P45,
    /// r5/6, K=9, dfree=5
    ConvV29P56,
    /// r6/7, K=9, dfree=4
    ConvV29P67,
    /// r7/8, K=9, dfree=4
    ConvV29P78,
    // Reed-Solomon codes
    /// m=8, n=255, k=223
    RsM8,
}


impl FecScheme {
    /// returns fec_scheme based on input string
    pub fn from_str(s: &str) -> Self {
        match s {
            "none" => FecScheme::None,
            "rep3" => FecScheme::Rep3,
            "rep5" => FecScheme::Rep5,
            "h74" => FecScheme::Hamming74,
            "h84" => FecScheme::Hamming84,
            "h128" => FecScheme::Hamming128,
            "g2412" => FecScheme::Golay2412,
            "secded2216" => FecScheme::Secded2216,
            "secded3932" => FecScheme::Secded3932,
            "secded7264" => FecScheme::Secded7264,
            "v27" => FecScheme::ConvV27,
            "v29" => FecScheme::ConvV29,
            "v39" => FecScheme::ConvV39,
            "v615" => FecScheme::ConvV615,
            "v27p23" => FecScheme::ConvV27P23,
            "v27p34" => FecScheme::ConvV27P34,
            "v27p45" => FecScheme::ConvV27P45,
            "v27p56" => FecScheme::ConvV27P56,
            "v27p67" => FecScheme::ConvV27P67,
            "v27p78" => FecScheme::ConvV27P78,
            "v29p23" => FecScheme::ConvV29P23,
            "v29p34" => FecScheme::ConvV29P34,
            "v29p45" => FecScheme::ConvV29P45,
            "v29p56" => FecScheme::ConvV29P56,
            "v29p67" => FecScheme::ConvV29P67,
            "v29p78" => FecScheme::ConvV29P78,
            "rs8" => FecScheme::RsM8,
            _ => FecScheme::Unknown,
        }
    }

    /// short name
    pub fn short_name(&self) -> &'static str {
        match self {
            FecScheme::Unknown => "unknown",
            FecScheme::None => "none",
            FecScheme::Rep3 => "rep3",
            FecScheme::Rep5 => "rep5",
            FecScheme::Hamming74 => "h74",
            FecScheme::Hamming84 => "h84",
            FecScheme::Hamming128 => "h128",
            FecScheme::Golay2412 => "g2412",
            FecScheme::Secded2216 => "secded2216",
            FecScheme::Secded3932 => "secded3932",
            FecScheme::Secded7264 => "secded7264",
            FecScheme::ConvV27 => "v27",
            FecScheme::ConvV29 => "v29",
            FecScheme::ConvV39 => "v39",
            FecScheme::ConvV615 => "v615",
            FecScheme::ConvV27P23 => "v27p23",
            FecScheme::ConvV27P34 => "v27p34",
            FecScheme::ConvV27P45 => "v27p45",
            FecScheme::ConvV27P56 => "v27p56",
            FecScheme::ConvV27P67 => "v27p67",
            FecScheme::ConvV27P78 => "v27p78",
            FecScheme::ConvV29P23 => "v29p23",
            FecScheme::ConvV29P34 => "v29p34",
            FecScheme::ConvV29P45 => "v29p45",
            FecScheme::ConvV29P56 => "v29p56",
            FecScheme::ConvV29P67 => "v29p67",
            FecScheme::ConvV29P78 => "v29p78",
            FecScheme::RsM8 => "rs8",
        }
    }

    /// long name
    pub fn long_name(&self) -> &'static str {
        match self {
            FecScheme::Unknown => "unknown",
            FecScheme::None => "none",
            FecScheme::Rep3 => "repeat(3)",
            FecScheme::Rep5 => "repeat(5)",
            FecScheme::Hamming74 => "Hamming(7,4)",
            FecScheme::Hamming84 => "Hamming(8,4)",
            FecScheme::Hamming128 => "Hamming(12,8)",
            FecScheme::Golay2412 => "Golay(24,12)",
            FecScheme::Secded2216 => "SEC-DED(22,16)",
            FecScheme::Secded3932 => "SEC-DED(39,32)",
            FecScheme::Secded7264 => "SEC-DED(72,64)",
            FecScheme::ConvV27 => "convolutional r1/2 K=7",
            FecScheme::ConvV29 => "convolutional r1/2 K=9",
            FecScheme::ConvV39 => "convolutional r1/3 K=9",
            FecScheme::ConvV615 => "convolutional r1/6 K=15",
            FecScheme::ConvV27P23 => "convolutional r2/3 K=7 (punctured)",
            FecScheme::ConvV27P34 => "convolutional r3/4 K=7 (punctured)",
            FecScheme::ConvV27P45 => "convolutional r4/5 K=7 (punctured)",
            FecScheme::ConvV27P56 => "convolutional r5/6 K=7 (punctured)",
            FecScheme::ConvV27P67 => "convolutional r6/7 K=7 (punctured)",
            FecScheme::ConvV27P78 => "convolutional r7/8 K=7 (punctured)",
            FecScheme::ConvV29P23 => "convolutional r2/3 K=9 (punctured)",
            FecScheme::ConvV29P34 => "convolutional r3/4 K=9 (punctured)",
            FecScheme::ConvV29P45 => "convolutional r4/5 K=9 (punctured)",
            FecScheme::ConvV29P56 => "convolutional r5/6 K=9 (punctured)",
            FecScheme::ConvV29P67 => "convolutional r6/7 K=9 (punctured)",
            FecScheme::ConvV29P78 => "convolutional r7/8 K=9 (punctured)",
            FecScheme::RsM8 => "Reed-Solomon, 223/255",
        }
    }

    /// is scheme convolutional?
    pub fn is_convolutional(&self) -> bool {
        matches!(
            self,
            FecScheme::ConvV27
                | FecScheme::ConvV29
                | FecScheme::ConvV39
                | FecScheme::ConvV615
                | FecScheme::ConvV27P23
                | FecScheme::ConvV27P34
                | FecScheme::ConvV27P45
                | FecScheme::ConvV27P56
                | FecScheme::ConvV27P67
                | FecScheme::ConvV27P78
                | FecScheme::ConvV29P23
                | FecScheme::ConvV29P34
                | FecScheme::ConvV29P45
                | FecScheme::ConvV29P56
                | FecScheme::ConvV29P67
                | FecScheme::ConvV29P78
        )
    }

    /// is scheme punctured?
    pub fn is_punctured(&self) -> bool {
        matches!(
            self,
            FecScheme::ConvV27P23
                | FecScheme::ConvV27P34
                | FecScheme::ConvV27P45
                | FecScheme::ConvV27P56
                | FecScheme::ConvV27P67
                | FecScheme::ConvV27P78
                | FecScheme::ConvV29P23
                | FecScheme::ConvV29P34
                | FecScheme::ConvV29P45
                | FecScheme::ConvV29P56
                | FecScheme::ConvV29P67
                | FecScheme::ConvV29P78
        )
    }

    /// is scheme Reed-Solomon?
    pub fn is_reedsolomon(&self) -> bool {
        matches!(self, FecScheme::RsM8)
    }

    /// is scheme Hamming?
    pub fn is_hamming(&self) -> bool {
        matches!(
            self,
            FecScheme::Hamming74 | FecScheme::Hamming84 | FecScheme::Hamming128
        )
    }

    /// is scheme repeat?
    pub fn is_repeat(&self) -> bool {
        matches!(self, FecScheme::Rep3 | FecScheme::Rep5)
    }

    /// symbol width in bits
    pub fn symbol_bits(&self) -> usize {
        if self.is_reedsolomon() {
            8
        } else {
            1
        }
    }

    /// get the theoretical rate of a particular forward error-
    /// correction scheme (object-independent method)
    pub fn rate(&self) -> f32 {
        match self {
            FecScheme::Unknown => 0.0,
            FecScheme::None => 1.0,
            FecScheme::Rep3 => 1.0 / 3.0,
            FecScheme::Rep5 => 1.0 / 5.0,
            FecScheme::Hamming74 => 4.0 / 7.0,
            FecScheme::Hamming84 => 4.0 / 8.0,
            FecScheme::Hamming128 => 8.0 / 12.0,
            FecScheme::Golay2412 => 1.0 / 2.0,
            // the parity bits occupy a whole byte rather than being packed, so
            // these are the rates as implemented, not the code's 16/22 and 32/39
            FecScheme::Secded2216 => 2.0 / 3.0, // ultimately 16/22 ~ 0.72727
            FecScheme::Secded3932 => 4.0 / 5.0, // ultimately 32/39 ~ 0.82051
            FecScheme::Secded7264 => 64.0 / 72.0,
            // convolutional codes
            FecScheme::ConvV27 => 1.0 / 2.0,
            FecScheme::ConvV29 => 1.0 / 2.0,
            FecScheme::ConvV39 => 1.0 / 3.0,
            FecScheme::ConvV615 => 1.0 / 6.0,
            FecScheme::ConvV27P23 | FecScheme::ConvV29P23 => 2.0 / 3.0,
            FecScheme::ConvV27P34 | FecScheme::ConvV29P34 => 3.0 / 4.0,
            FecScheme::ConvV27P45 | FecScheme::ConvV29P45 => 4.0 / 5.0,
            FecScheme::ConvV27P56 | FecScheme::ConvV29P56 => 5.0 / 6.0,
            FecScheme::ConvV27P67 | FecScheme::ConvV29P67 => 6.0 / 7.0,
            FecScheme::ConvV27P78 | FecScheme::ConvV29P78 => 7.0 / 8.0,
            // Reed-Solomon codes
            FecScheme::RsM8 => 223.0 / 255.0,
        }
    }

    /// return the encoded message length using a particular error-
    /// correction scheme (object-independent method)
    ///
    ///  dec_msg_len    :   raw, uncoded message length
    pub fn enc_msg_len(&self, dec_msg_len: usize) -> usize {
        match self {
            FecScheme::Unknown => 0,
            FecScheme::None => dec_msg_len,
            FecScheme::Rep3 => 3 * dec_msg_len,
            FecScheme::Rep5 => 5 * dec_msg_len,
            FecScheme::Hamming74 => block_enc_msg_len(dec_msg_len, 4, 7),
            FecScheme::Hamming84 => block_enc_msg_len(dec_msg_len, 4, 8),
            FecScheme::Hamming128 => block_enc_msg_len(dec_msg_len, 8, 12),
            FecScheme::Golay2412 => block_enc_msg_len(dec_msg_len, 12, 24),
            FecScheme::Secded2216 => dec_msg_len + (dec_msg_len + 1) / 2,
            FecScheme::Secded3932 => dec_msg_len + (dec_msg_len + 3) / 4,
            FecScheme::Secded7264 => dec_msg_len + (dec_msg_len + 7) / 8,
            // convolutional codes
            FecScheme::ConvV27 => 2 * dec_msg_len + 2, // (K-1)/r=12, round up to 2 bytes
            FecScheme::ConvV29 => 2 * dec_msg_len + 2, // (K-1)/r=16, 2 bytes
            FecScheme::ConvV39 => 3 * dec_msg_len + 3, // (K-1)/r=24, 3 bytes
            FecScheme::ConvV615 => 6 * dec_msg_len + 11, // (K-1)/r=84, round up to 11 bytes

            FecScheme::ConvV27P23 => conv_punctured_enc_msg_len(dec_msg_len, 7, 2),
            FecScheme::ConvV27P34 => conv_punctured_enc_msg_len(dec_msg_len, 7, 3),
            FecScheme::ConvV27P45 => conv_punctured_enc_msg_len(dec_msg_len, 7, 4),
            FecScheme::ConvV27P56 => conv_punctured_enc_msg_len(dec_msg_len, 7, 5),
            FecScheme::ConvV27P67 => conv_punctured_enc_msg_len(dec_msg_len, 7, 6),
            FecScheme::ConvV27P78 => conv_punctured_enc_msg_len(dec_msg_len, 7, 7),
            FecScheme::ConvV29P23 => conv_punctured_enc_msg_len(dec_msg_len, 9, 2),
            FecScheme::ConvV29P34 => conv_punctured_enc_msg_len(dec_msg_len, 9, 3),
            FecScheme::ConvV29P45 => conv_punctured_enc_msg_len(dec_msg_len, 9, 4),
            FecScheme::ConvV29P56 => conv_punctured_enc_msg_len(dec_msg_len, 9, 5),
            FecScheme::ConvV29P67 => conv_punctured_enc_msg_len(dec_msg_len, 9, 6),
            FecScheme::ConvV29P78 => conv_punctured_enc_msg_len(dec_msg_len, 9, 7),

            // Reed-Solomon codes
            FecScheme::RsM8 => rs_enc_msg_len(dec_msg_len, RS_M8_NROOTS, RS_M8_KK),
        }
    }
}

/// compute encoded message length for block codes
///
///  dec_msg_len    :   decoded message length (bytes)
///  m              :   input block size (bits)
///  k              :   output block size (bits)
pub(crate) fn block_enc_msg_len(dec_msg_len: usize, m: usize, k: usize) -> usize {
    // compute total number of bits in decoded message
    let num_bits_in = dec_msg_len * 8;

    // compute total number of blocks: ceil(num_bits_in/m)
    let num_blocks = (num_bits_in + m - 1) / m;

    // compute total number of bits out
    let num_bits_out = num_blocks * k;

    // compute total number of bytes out: ceil(num_bits_out/8)
    (num_bits_out + 7) / 8
}

/// Compute encoded message length for punctured convolutional codes.
///
/// Mirrors liquid's `fec_conv_get_enc_msg_len`: the base rate-1/2 code emits
/// `n = len*8 + K - 1` bits per output stream, and puncturing at rate
/// `p/(p+1)` keeps `n + ceil(n/p)` bits in total.
pub(crate) fn conv_punctured_enc_msg_len(dec_msg_len: usize, k: usize, p: usize) -> usize {
    let num_bits_in = dec_msg_len * 8;
    let n = num_bits_in + k - 1;
    let num_bits_out = n + n.div_ceil(p);
    num_bits_out.div_ceil(8)
}

/// Number of parity bytes for the m=8 Reed-Solomon code
pub(crate) const RS_M8_NROOTS: usize = 32;
/// Message capacity of a full m=8 Reed-Solomon block
pub(crate) const RS_M8_KK: usize = 223;

/// Compute encoded message length for Reed-Solomon codes.
///
/// The message is split into `ceil(len/kk)` blocks of as-equal size as
/// possible, each carrying `nroots` parity bytes. Blocks shorter than `kk`
/// are shortened codes: the missing symbols are treated as virtual zero
/// padding that is never transmitted.
pub(crate) fn rs_enc_msg_len(dec_msg_len: usize, nroots: usize, kk: usize) -> usize {
    if dec_msg_len == 0 {
        return 0;
    }

    let num_blocks = (dec_msg_len + kk - 1) / kk;
    let dec_block_len = (dec_msg_len + num_blocks - 1) / num_blocks;

    (dec_block_len + nroots) * num_blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    // (scheme, name, conv, punctured, rs, hamming, repeat). liquid's autotests
    // spot-check a subset, but driving all of them means a newly added scheme
    // can't slip past unclassified
    const SCHEMES: &[(FecScheme, &str, bool, bool, bool, bool, bool)] = &[
        (FecScheme::None, "none", false, false, false, false, false),
        (FecScheme::Rep3, "rep3", false, false, false, false, true),
        (FecScheme::Rep5, "rep5", false, false, false, false, true),
        (FecScheme::Hamming74, "h74", false, false, false, true, false),
        (FecScheme::Hamming84, "h84", false, false, false, true, false),
        (FecScheme::Hamming128, "h128", false, false, false, true, false),
        (FecScheme::Golay2412, "g2412", false, false, false, false, false),
        (FecScheme::Secded2216, "secded2216", false, false, false, false, false),
        (FecScheme::Secded3932, "secded3932", false, false, false, false, false),
        (FecScheme::Secded7264, "secded7264", false, false, false, false, false),
        (FecScheme::ConvV27, "v27", true, false, false, false, false),
        (FecScheme::ConvV29, "v29", true, false, false, false, false),
        (FecScheme::ConvV39, "v39", true, false, false, false, false),
        (FecScheme::ConvV615, "v615", true, false, false, false, false),
        (FecScheme::ConvV27P23, "v27p23", true, true, false, false, false),
        (FecScheme::ConvV27P34, "v27p34", true, true, false, false, false),
        (FecScheme::ConvV27P45, "v27p45", true, true, false, false, false),
        (FecScheme::ConvV27P56, "v27p56", true, true, false, false, false),
        (FecScheme::ConvV27P67, "v27p67", true, true, false, false, false),
        (FecScheme::ConvV27P78, "v27p78", true, true, false, false, false),
        (FecScheme::ConvV29P23, "v29p23", true, true, false, false, false),
        (FecScheme::ConvV29P34, "v29p34", true, true, false, false, false),
        (FecScheme::ConvV29P45, "v29p45", true, true, false, false, false),
        (FecScheme::ConvV29P56, "v29p56", true, true, false, false, false),
        (FecScheme::ConvV29P67, "v29p67", true, true, false, false, false),
        (FecScheme::ConvV29P78, "v29p78", true, true, false, false, false),
        (FecScheme::RsM8, "rs8", false, false, true, false, false),
    ];

    #[test]
    #[autotest_annotate(autotest_fec_str2fec)]
    fn test_fec_str2fec() {
        // invalid case
        assert_eq!(FecScheme::from_str("invalid scheme"), FecScheme::Unknown);

        for &(scheme, name, ..) in SCHEMES {
            assert_eq!(FecScheme::from_str(name), scheme, "from_str({})", name);
            // short_name is the inverse: it must round-trip
            assert_eq!(scheme.short_name(), name, "short_name({:?})", scheme);
        }
    }

    #[test]
    #[autotest_annotate(autotest_fec_is_convolutional)]
    fn test_fec_is_convolutional() {
        for &(scheme, _, conv, ..) in SCHEMES {
            assert_eq!(scheme.is_convolutional(), conv, "{:?}", scheme);
        }
        assert!(!FecScheme::Unknown.is_convolutional());
    }

    #[test]
    #[autotest_annotate(autotest_fec_is_punctured)]
    fn test_fec_is_punctured() {
        for &(scheme, _, _, punctured, ..) in SCHEMES {
            assert_eq!(scheme.is_punctured(), punctured, "{:?}", scheme);
            // puncturing only applies to convolutional codes
            if punctured {
                assert!(scheme.is_convolutional(), "{:?}", scheme);
            }
        }
        assert!(!FecScheme::Unknown.is_punctured());
    }

    #[test]
    #[autotest_annotate(autotest_fec_is_reedsolomon)]
    fn test_fec_is_reedsolomon() {
        for &(scheme, _, _, _, rs, ..) in SCHEMES {
            assert_eq!(scheme.is_reedsolomon(), rs, "{:?}", scheme);
        }
        assert!(!FecScheme::Unknown.is_reedsolomon());
    }

    #[test]
    #[autotest_annotate(autotest_fec_is_hamming)]
    fn test_fec_is_hamming() {
        for &(scheme, _, _, _, _, hamming, _) in SCHEMES {
            assert_eq!(scheme.is_hamming(), hamming, "{:?}", scheme);
        }
        assert!(!FecScheme::Unknown.is_hamming());
    }

    // liquid declares fec_scheme_is_repeat but has no autotest for it
    #[test]
    fn test_fec_is_repeat() {
        for &(scheme, _, _, _, _, _, repeat) in SCHEMES {
            assert_eq!(scheme.is_repeat(), repeat, "{:?}", scheme);
        }
        assert!(!FecScheme::Unknown.is_repeat());
    }

    #[test]
    fn test_fec_families_disjoint() {
        for &(scheme, ..) in SCHEMES {
            let families = [
                scheme.is_convolutional(),
                scheme.is_reedsolomon(),
                scheme.is_hamming(),
                scheme.is_repeat(),
            ];
            assert!(
                families.iter().filter(|&&b| b).count() <= 1,
                "{:?} belongs to more than one family",
                scheme
            );
        }
    }

    #[test]
    fn test_fec_rate_matches_enc_msg_len() {
        for &(scheme, name, ..) in SCHEMES {
            // block codes only: the convolutional and RS lengths carry a tail
            // or parity overhead that does not vanish as the message grows
            if scheme.is_convolutional() || scheme.is_reedsolomon() {
                continue;
            }

            // pick a length that divides evenly for every block code here
            let n = 8 * 3usize;
            let measured = n as f32 / scheme.enc_msg_len(n) as f32;

            assert!(
                (measured - scheme.rate()).abs() < 1e-6,
                "{}: rate() {} disagrees with measured {}",
                name,
                scheme.rate(),
                measured
            );
        }
    }
}
