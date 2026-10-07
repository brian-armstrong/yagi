//
// forward error-correction scheme
//
use crate::error::{Error, Result};

fn length_overflow() -> Error {
    Error::Range("encoded message length overflows usize".into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Forward error-correction code.
///
/// Each scheme fixes a nominal code rate and the encoded length for each
/// message length. Create a [`Fec`](crate::fec::Fec) to encode and decode with
/// a scheme. Schemes parse from their [short names](Self::short_name) with
/// [`str::parse`], and [`Self::ALL`] lists every usable scheme.
///
/// Rates below are decoded bits over encoded bits. `K` is the constraint
/// length of a convolutional code and `dfree` its free distance.
pub enum FecScheme {
    /// Placeholder for an unrecognized scheme. It cannot encode or decode.
    Unknown,
    /// No error correction. Data passes through unchanged.
    None,
    /// Repetition code, rate 1/3.
    Rep3,
    /// Repetition code, rate 1/5.
    Rep5,
    /// Hamming (7,4) code, rate 4/7.
    Hamming74,
    /// Hamming (8,4) code, which adds a parity bit to Hamming (7,4), rate 1/2.
    Hamming84,
    /// Hamming (12,8) code, rate 2/3.
    Hamming128,
    /// Golay (24,12) code, rate 1/2. Corrects up to three bit errors in each
    /// 24-bit codeword.
    Golay2412,
    /// SEC-DED (22,16) code. Each 16-bit block carries its six parity bits in
    /// a whole byte, so the rate is 2/3.
    Secded2216,
    /// SEC-DED (39,32) code. Each 32-bit block carries its seven parity bits
    /// in a whole byte, so the rate is 4/5.
    Secded3932,
    /// SEC-DED (72,64) code, rate 8/9.
    Secded7264,
    // codecs not defined internally (see http://www.ka9q.net/code/fec/)
    /// Convolutional code, rate 1/2, K=7, dfree=10.
    ConvV27,
    /// Convolutional code, rate 1/2, K=9, dfree=12.
    ConvV29,
    /// Convolutional code, rate 1/3, K=9, dfree=18.
    ConvV39,
    /// Convolutional code, rate 1/6, K=15, dfree<=57 (Heller 1968).
    ConvV615,
    // punctured (perforated) codes
    /// Punctured convolutional code, rate 2/3, K=7, dfree=6.
    ConvV27P23,
    /// Punctured convolutional code, rate 3/4, K=7, dfree=5.
    ConvV27P34,
    /// Punctured convolutional code, rate 4/5, K=7, dfree=4.
    ConvV27P45,
    /// Punctured convolutional code, rate 5/6, K=7, dfree=4.
    ConvV27P56,
    /// Punctured convolutional code, rate 6/7, K=7, dfree=3.
    ConvV27P67,
    /// Punctured convolutional code, rate 7/8, K=7, dfree=3.
    ConvV27P78,
    /// Punctured convolutional code, rate 2/3, K=9, dfree=7.
    ConvV29P23,
    /// Punctured convolutional code, rate 3/4, K=9, dfree=5.
    ConvV29P34,
    /// Punctured convolutional code, rate 4/5, K=9, dfree=4.
    ConvV29P45,
    /// Punctured convolutional code, rate 5/6, K=9, dfree=3.
    ConvV29P56,
    /// Punctured convolutional code, rate 6/7, K=9, dfree=4.
    ConvV29P67,
    /// Punctured convolutional code, rate 7/8, K=9, dfree=3.
    ConvV29P78,
    // Reed-Solomon codes
    /// Reed-Solomon (255,223) code over bytes, rate 223/255.
    ///
    /// Each block carries 32 parity bytes and corrects up to 16 byte errors.
    /// A message longer than 223 bytes is split into blocks of nearly equal
    /// size.
    RsM8,
}

/// Number of FEC-scheme variants, including [`FecScheme::Unknown`].
pub const NUM_FEC_SCHEMES: usize = FecScheme::RsM8 as usize + 1;

impl FecScheme {
    /// Every usable FEC scheme, in declaration order.
    ///
    /// Excludes [`Self::Unknown`], which is not a scheme.
    pub const ALL: [FecScheme; NUM_FEC_SCHEMES - 1] = [
        FecScheme::None,
        FecScheme::Rep3,
        FecScheme::Rep5,
        FecScheme::Hamming74,
        FecScheme::Hamming84,
        FecScheme::Hamming128,
        FecScheme::Golay2412,
        FecScheme::Secded2216,
        FecScheme::Secded3932,
        FecScheme::Secded7264,
        FecScheme::ConvV27,
        FecScheme::ConvV29,
        FecScheme::ConvV39,
        FecScheme::ConvV615,
        FecScheme::ConvV27P23,
        FecScheme::ConvV27P34,
        FecScheme::ConvV27P45,
        FecScheme::ConvV27P56,
        FecScheme::ConvV27P67,
        FecScheme::ConvV27P78,
        FecScheme::ConvV29P23,
        FecScheme::ConvV29P34,
        FecScheme::ConvV29P45,
        FecScheme::ConvV29P56,
        FecScheme::ConvV29P67,
        FecScheme::ConvV29P78,
        FecScheme::RsM8,
    ];

    /// Returns the scheme whose integer index is `index`, or `None` when
    /// `index` names no scheme.
    ///
    /// This turns a scheme stored as a number, such as in a frame header,
    /// back into the scheme. `index` 0 is [`Self::Unknown`], which this
    /// rejects as not a usable scheme.
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|s| *s as usize == index)
    }

    /// Returns the short name that [`str::parse`] accepts, such as `"v27p34"`.
    ///
    /// The name of [`Self::Unknown`] is `"unknown"`, which does not parse.
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

    fn from_short_name(s: &str) -> crate::error::Result<Self> {
        Ok(match s {
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
            _ => return Err(crate::error::Error::Config(format!("unknown FEC scheme: {}", s))),
        })
    }

    /// Returns a descriptive name for display, such as `"Hamming(7,4)"`.
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

    /// Returns whether this is a convolutional code, punctured or not.
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

    /// Returns whether this is a punctured convolutional code.
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

    /// Returns whether this is a Reed-Solomon code.
    pub fn is_reedsolomon(&self) -> bool {
        matches!(self, FecScheme::RsM8)
    }

    /// Returns whether this is a Hamming code.
    pub fn is_hamming(&self) -> bool {
        matches!(self, FecScheme::Hamming74 | FecScheme::Hamming84 | FecScheme::Hamming128)
    }

    /// Returns whether this is a repetition code.
    pub fn is_repeat(&self) -> bool {
        matches!(self, FecScheme::Rep3 | FecScheme::Rep5)
    }

    /// Returns the width, in bits, of the symbols that interleaving should
    /// keep intact.
    ///
    /// The width is 8 for Reed-Solomon, which corrects whole bytes, and 1 for
    /// every other scheme.
    pub fn symbol_bits(&self) -> usize {
        if self.is_reedsolomon() {
            8
        } else {
            1
        }
    }

    /// Returns the nominal code rate, decoded bits over encoded bits.
    ///
    /// Padding, parity bytes, and convolutional tail bits make the actual rate
    /// for a given message length differ. [`Self::enc_msg_len`] gives the
    /// exact encoded length. The rate of [`Self::Unknown`] is zero.
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

    /// Returns the encoded length in bytes for a message of `dec_msg_len`
    /// bytes.
    ///
    /// The length of [`Self::Unknown`] is zero.
    ///
    /// # Errors
    ///
    /// Returns an error when the encoded length overflows `usize`.
    pub fn enc_msg_len(&self, dec_msg_len: usize) -> Result<usize> {
        match self {
            FecScheme::Unknown => Ok(0),
            FecScheme::None => Ok(dec_msg_len),
            FecScheme::Rep3 => dec_msg_len.checked_mul(3).ok_or_else(length_overflow),
            FecScheme::Rep5 => dec_msg_len.checked_mul(5).ok_or_else(length_overflow),
            FecScheme::Hamming74 => block_enc_msg_len(dec_msg_len, 4, 7),
            FecScheme::Hamming84 => block_enc_msg_len(dec_msg_len, 4, 8),
            FecScheme::Hamming128 => block_enc_msg_len(dec_msg_len, 8, 12),
            FecScheme::Golay2412 => block_enc_msg_len(dec_msg_len, 12, 24),
            FecScheme::Secded2216 => dec_msg_len.checked_add(dec_msg_len.div_ceil(2)).ok_or_else(length_overflow),
            FecScheme::Secded3932 => dec_msg_len.checked_add(dec_msg_len.div_ceil(4)).ok_or_else(length_overflow),
            FecScheme::Secded7264 => dec_msg_len.checked_add(dec_msg_len.div_ceil(8)).ok_or_else(length_overflow),
            // convolutional codes
            FecScheme::ConvV27 | FecScheme::ConvV29 => {
                dec_msg_len.checked_mul(2).and_then(|n| n.checked_add(2)).ok_or_else(length_overflow)
            }
            FecScheme::ConvV39 => dec_msg_len.checked_mul(3).and_then(|n| n.checked_add(3)).ok_or_else(length_overflow),
            FecScheme::ConvV615 => {
                dec_msg_len.checked_mul(6).and_then(|n| n.checked_add(11)).ok_or_else(length_overflow)
            }

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

impl std::str::FromStr for FecScheme {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_short_name(s)
    }
}

/// compute encoded message length for block codes
///
///  dec_msg_len    :   decoded message length (bytes)
///  m              :   input block size (bits)
///  k              :   output block size (bits)
pub(crate) fn block_enc_msg_len(dec_msg_len: usize, m: usize, k: usize) -> Result<usize> {
    // compute total number of bits in decoded message
    let num_bits_in = (dec_msg_len as u128).checked_mul(8).ok_or_else(length_overflow)?;

    // compute total number of blocks: ceil(num_bits_in/m)
    let num_blocks = num_bits_in.div_ceil(m as u128);

    // compute total number of bits out
    let num_bits_out = num_blocks.checked_mul(k as u128).ok_or_else(length_overflow)?;

    // compute total number of bytes out: ceil(num_bits_out/8)
    num_bits_out.div_ceil(8).try_into().map_err(|_| length_overflow())
}

/// Compute encoded message length for punctured convolutional codes.
///
/// Mirrors liquid's `fec_conv_get_enc_msg_len`: the base rate-1/2 code emits
/// `n = len*8 + K - 1` bits per output stream, and puncturing at rate
/// `p/(p+1)` keeps `n + ceil(n/p)` bits in total.
pub(crate) fn conv_punctured_enc_msg_len(dec_msg_len: usize, k: usize, p: usize) -> Result<usize> {
    let num_bits_in = (dec_msg_len as u128).checked_mul(8).ok_or_else(length_overflow)?;
    let n = num_bits_in.checked_add(k as u128 - 1).ok_or_else(length_overflow)?;
    let num_bits_out = n.checked_add(n.div_ceil(p as u128)).ok_or_else(length_overflow)?;
    num_bits_out.div_ceil(8).try_into().map_err(|_| length_overflow())
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
pub(crate) fn rs_enc_msg_len(dec_msg_len: usize, nroots: usize, kk: usize) -> Result<usize> {
    if dec_msg_len == 0 {
        return Ok(0);
    }

    let num_blocks = dec_msg_len.div_ceil(kk);
    let dec_block_len = dec_msg_len.div_ceil(num_blocks);

    dec_block_len
        .checked_add(nroots)
        .and_then(|n| n.checked_mul(num_blocks))
        .ok_or_else(length_overflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;

    #[test]
    fn encoded_lengths_reject_overflow_without_rejecting_representable_lengths() {
        for scheme in FecScheme::ALL {
            if scheme != FecScheme::None {
                assert!(scheme.enc_msg_len(usize::MAX).is_err(), "{scheme:?}");
            }
        }
        assert_eq!(FecScheme::None.enc_msg_len(usize::MAX), Ok(usize::MAX));

        // Bit counts exceed usize here, but the encoded byte counts fit.
        let n = usize::MAX / 2;
        let hamming = ((n as u128) * 8).div_ceil(4) * 7;
        assert_eq!(FecScheme::Hamming74.enc_msg_len(n), Ok(hamming.div_ceil(8) as usize));
        assert!(FecScheme::ConvV27P23.enc_msg_len(n).is_ok());
    }

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
        assert!("invalid scheme".parse::<FecScheme>().is_err());

        for &(scheme, name, ..) in SCHEMES {
            assert_eq!(name.parse::<FecScheme>().unwrap(), scheme, "from_str({})", name);
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
    fn test_fec_all_covers_schemes() {
        assert_eq!(FecScheme::ALL.len() + 1, NUM_FEC_SCHEMES);
        assert!(!FecScheme::ALL.contains(&FecScheme::Unknown));

        // no duplicates
        let mut seen = std::collections::HashSet::new();
        for fs in FecScheme::ALL {
            assert!(seen.insert(fs.short_name()), "{:?} listed twice", fs);
            assert_eq!(fs.short_name().parse::<FecScheme>().unwrap(), fs);
            assert!(!fs.long_name().is_empty(), "{:?}", fs);
        }

        let tabled: std::collections::HashSet<_> = SCHEMES.iter().map(|&(_, name, ..)| name).collect();
        assert_eq!(seen, tabled, "SCHEMES and ALL disagree");
    }

    #[test]
    fn test_fec_families_disjoint() {
        for &(scheme, ..) in SCHEMES {
            let families =
                [scheme.is_convolutional(), scheme.is_reedsolomon(), scheme.is_hamming(), scheme.is_repeat()];
            assert!(families.iter().filter(|&&b| b).count() <= 1, "{:?} belongs to more than one family", scheme);
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
            let measured = n as f32 / scheme.enc_msg_len(n).unwrap() as f32;

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
