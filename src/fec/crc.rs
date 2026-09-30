//
// cyclic redundancy check (and family)
//
use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Error-detection scheme for a byte message.
///
/// A scheme computes a key of [`Self::key_len`] bytes from a message. The
/// receiver recomputes the key and compares. Schemes parse from their
/// [short names](Self::short_name) with [`str::parse`], and [`Self::ALL`]
/// lists every usable scheme.
pub enum CrcScheme {
    /// Placeholder for an unrecognized scheme. Validation with it always
    /// fails.
    Unknown,
    /// No error detection. Validation always succeeds.
    None,
    /// 8-bit two's-complement checksum. See [`checksum_generate_key`].
    Checksum,
    /// 8-bit CRC. See [`crc8_generate_key`].
    Crc8,
    /// 16-bit CRC. See [`crc16_generate_key`].
    Crc16,
    /// 24-bit CRC. See [`crc24_generate_key`].
    Crc24,
    /// 32-bit CRC. See [`crc32_generate_key`].
    Crc32,
}

/// Number of CRC-scheme variants, including [`CrcScheme::Unknown`].
pub const NUM_CRC_SCHEMES: usize = CrcScheme::Crc32 as usize + 1;

impl CrcScheme {
    /// Every usable CRC scheme, in declaration order.
    ///
    /// Excludes [`Self::Unknown`], which is not a scheme.
    pub const ALL: [CrcScheme; NUM_CRC_SCHEMES - 1] = [
        CrcScheme::None,
        CrcScheme::Checksum,
        CrcScheme::Crc8,
        CrcScheme::Crc16,
        CrcScheme::Crc24,
        CrcScheme::Crc32,
    ];

    /// Returns the short name that [`str::parse`] accepts, such as `"crc32"`.
    ///
    /// The name of [`Self::Unknown`] is `"unknown"`, which does not parse.
    pub fn short_name(&self) -> &'static str {
        match self {
            CrcScheme::Unknown => "unknown",
            CrcScheme::None => "none",
            CrcScheme::Checksum => "checksum",
            CrcScheme::Crc8 => "crc8",
            CrcScheme::Crc16 => "crc16",
            CrcScheme::Crc24 => "crc24",
            CrcScheme::Crc32 => "crc32",
        }
    }

    fn from_short_name(s: &str) -> Result<Self> {
        Ok(match s {
            "none" => CrcScheme::None,
            "checksum" => CrcScheme::Checksum,
            "crc8" => CrcScheme::Crc8,
            "crc16" => CrcScheme::Crc16,
            "crc24" => CrcScheme::Crc24,
            "crc32" => CrcScheme::Crc32,
            _ => return Err(Error::Config(format!("unknown CRC scheme: {}", s))),
        })
    }

    /// Returns a descriptive name for display, such as `"CRC (32-bit)"`.
    pub fn long_name(&self) -> &'static str {
        match self {
            CrcScheme::Unknown => "unknown",
            CrcScheme::None => "none",
            CrcScheme::Checksum => "checksum (8-bit)",
            CrcScheme::Crc8 => "CRC (8-bit)",
            CrcScheme::Crc16 => "CRC (16-bit)",
            CrcScheme::Crc24 => "CRC (24-bit)",
            CrcScheme::Crc32 => "CRC (32-bit)",
        }
    }

    /// Returns the key length in bytes.
    ///
    /// The length is zero for [`Self::None`] and [`Self::Unknown`].
    pub fn key_len(&self) -> usize {
        match self {
            CrcScheme::Unknown => 0,
            CrcScheme::None => 0,
            CrcScheme::Checksum => 1,
            CrcScheme::Crc8 => 1,
            CrcScheme::Crc16 => 2,
            CrcScheme::Crc24 => 3,
            CrcScheme::Crc32 => 4,
        }
    }
}

impl std::str::FromStr for CrcScheme {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_short_name(s)
    }
}

const CRC8_POLY_REV: u32 = 0xE0; // reverse of CRC8_POLY  0x07
const CRC16_POLY_REV: u32 = 0xA001; // reverse of CRC16_POLY 0x8005
const CRC24_POLY_REV: u32 = 0xD3B6BA; // reverse of CRC24_POLY 0x5D6DCB
const CRC32_POLY_REV: u32 = 0xEDB88320; // reverse of CRC32_POLY 0x04C11DB7

// liquid operates one bit at a time. these tables collapse the inner loop.
// algorithm from: http://www.hackersdelight.org/crc.pdf
const CRC8_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (CRC8_POLY_REV & mask);
            j += 1;
        }
        table[i] = crc as u8;
        i += 1;
    }
    table
};

const CRC16_TABLE: [u16; 256] = {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (CRC16_POLY_REV & mask);
            j += 1;
        }
        table[i] = crc as u16;
        i += 1;
    }
    table
};

const CRC24_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (CRC24_POLY_REV & mask);
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (CRC32_POLY_REV & mask);
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Generates the error-detection key for a message.
///
/// The key occupies the low `8 * scheme.key_len()` bits of the result.
/// [`CrcScheme::None`] and [`CrcScheme::Unknown`] give zero.
///
/// # Panics
///
/// Panics under the same condition as [`checksum_generate_key`] when
/// `scheme` is [`CrcScheme::Checksum`].
pub fn generate_key(scheme: CrcScheme, msg: &[u8]) -> u32 {
    match scheme {
        CrcScheme::Unknown => 0,
        CrcScheme::None => 0,
        CrcScheme::Checksum => checksum_generate_key(msg) as u32,
        CrcScheme::Crc8 => crc8_generate_key(msg) as u32,
        CrcScheme::Crc16 => crc16_generate_key(msg) as u32,
        CrcScheme::Crc24 => crc24_generate_key(msg),
        CrcScheme::Crc32 => crc32_generate_key(msg),
    }
}

/// Generates a key and writes it after the message.
///
/// The first `msg_len` bytes of `msg` are the message. The key is written
/// immediately after them, most significant byte first. Bytes beyond the key
/// are unchanged.
///
/// # Panics
///
/// Panics when `msg_len + scheme.key_len()` exceeds `msg.len()`, or under the
/// same condition as [`checksum_generate_key`] when `scheme` is
/// [`CrcScheme::Checksum`].
pub fn append_key(scheme: CrcScheme, msg: &mut [u8], msg_len: usize) {
    let key_len = scheme.key_len();
    let key = generate_key(scheme, &msg[..msg_len]);

    for i in 0..key_len {
        msg[msg_len + i] = ((key >> ((key_len - i - 1) * 8)) & 0xff) as u8;
    }
}

/// Returns whether `key` matches the key generated for `msg`.
///
/// [`CrcScheme::None`] always returns `true` and [`CrcScheme::Unknown`]
/// always returns `false`. A match does not prove the message is error-free,
/// because different messages can share a key.
///
/// # Panics
///
/// Panics under the same condition as [`checksum_generate_key`] when
/// `scheme` is [`CrcScheme::Checksum`].
pub fn validate_message(scheme: CrcScheme, msg: &[u8], key: u32) -> bool {
    match scheme {
        CrcScheme::Unknown => false,
        CrcScheme::None => true,
        _ => generate_key(scheme, msg) == key,
    }
}

/// Returns whether the key written after a message matches it.
///
/// The first `msg_len` bytes of `msg` are the message and the following
/// `scheme.key_len()` bytes are the key, most significant byte first, as
/// [`append_key`] writes it. See [`validate_message`] for how the key is
/// compared.
///
/// # Panics
///
/// Panics when `msg_len + scheme.key_len()` exceeds `msg.len()`, or under the
/// same condition as [`checksum_generate_key`] when `scheme` is
/// [`CrcScheme::Checksum`].
pub fn check_key(scheme: CrcScheme, msg: &[u8], msg_len: usize) -> bool {
    let key_len = scheme.key_len();

    // extract key from end of message
    let mut key = 0u32;
    for i in 0..key_len {
        key = (key << 8) | (msg[msg_len + i] as u32);
    }

    validate_message(scheme, &msg[..msg_len], key)
}

//
// Checksum
//

/// Generates the 8-bit checksum for `data`.
///
/// The checksum is the two's complement of the byte sum modulo 256, so the
/// bytes and the checksum sum to zero modulo 256.
///
/// # Panics
///
/// Panics when the byte sum exceeds `u32::MAX` and overflow checks are
/// enabled. That takes more than 16 MiB of input. Without overflow checks,
/// the checksum is still correct.
pub fn checksum_generate_key(data: &[u8]) -> u8 {
    let sum: u32 = data.iter().map(|&b| b as u32).sum();
    // mask and convert to 2's complement
    (!(sum as u8)).wrapping_add(1)
}

//
// CRC-8
//

/// Generates the 8-bit CRC for `msg`.
///
/// The CRC uses polynomial `0x07`, processes bits least significant first,
/// starts from all ones, and inverts the result.
pub fn crc8_generate_key(msg: &[u8]) -> u8 {
    let mut crc = 0xffu8;
    for &b in msg {
        crc = CRC8_TABLE[(crc ^ b) as usize];
    }
    !crc
}

//
// CRC-16
//

/// Generates the 16-bit CRC for `msg`.
///
/// The CRC uses polynomial `0x8005`, processes bits least significant first,
/// starts from all ones, and inverts the result. These are the CRC-16/USB
/// parameters.
pub fn crc16_generate_key(msg: &[u8]) -> u16 {
    let mut crc = 0xffffu16;
    for &b in msg {
        crc = CRC16_TABLE[((crc ^ b as u16) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

//
// CRC-24
//

/// Generates the 24-bit CRC for `msg` in the low 24 bits.
///
/// The CRC uses polynomial `0x5D6DCB`, processes bits least significant
/// first, starts from all ones, and inverts the result.
pub fn crc24_generate_key(msg: &[u8]) -> u32 {
    let mut crc = 0xffffffu32;
    for &b in msg {
        crc = CRC24_TABLE[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    (!crc) & 0xffffff
}

//
// CRC-32
//

/// Generates the 32-bit CRC for `msg`.
///
/// The CRC uses polynomial `0x04C11DB7`, processes bits least significant
/// first, starts from all ones, and inverts the result. This is the CRC-32
/// used by zlib.
pub fn crc32_generate_key(msg: &[u8]) -> u32 {
    let mut crc = 0xffffffffu32;
    for &b in msg {
        crc = CRC32_TABLE[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::MaximalLengthSequence;
    use test_macro::autotest_annotate;

    fn crc_test(scheme: CrcScheme, n: usize) {
        // generate pseudo-random data
        let mut ms = MaximalLengthSequence::from_degree(9).unwrap();
        let data: Vec<u8> = (0..n).map(|_| ms.generate_symbol(8) as u8).collect();

        // generate key
        let key = generate_key(scheme, &data);

        // verify data/key are valid
        assert!(validate_message(scheme, &data, key));

        // test flipping each bit and confirm check fails
        let mut data_corrupt = data.clone();
        for i in 0..n {
            for j in 0..8 {
                // copy original
                data_corrupt.copy_from_slice(&data);
                // flip bit j at byte i
                data_corrupt[i] ^= 1 << j;
                // verify check fails
                assert!(!validate_message(scheme, &data_corrupt, key));
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_checksum)]
    fn test_checksum() {
        crc_test(CrcScheme::Checksum, 16);
    }

    #[test]
    #[autotest_annotate(autotest_crc8)]
    fn test_crc8() {
        crc_test(CrcScheme::Crc8, 16);
    }

    #[test]
    #[autotest_annotate(autotest_crc16)]
    fn test_crc16() {
        crc_test(CrcScheme::Crc16, 64);
    }

    #[test]
    #[autotest_annotate(autotest_crc24)]
    fn test_crc24() {
        crc_test(CrcScheme::Crc24, 64);
    }

    #[test]
    #[autotest_annotate(autotest_crc32)]
    fn test_crc32() {
        crc_test(CrcScheme::Crc32, 64);
    }

    #[test]
    #[autotest_annotate(autotest_crc_config)]
    fn test_crc_config() {
        assert!("unknown".parse::<CrcScheme>().is_err());
        assert!("rosebud".parse::<CrcScheme>().is_err());
        assert_eq!("none".parse::<CrcScheme>().unwrap(), CrcScheme::None);
        assert_eq!("checksum".parse::<CrcScheme>().unwrap(), CrcScheme::Checksum);
        assert_eq!("crc8".parse::<CrcScheme>().unwrap(), CrcScheme::Crc8);
        assert_eq!("crc16".parse::<CrcScheme>().unwrap(), CrcScheme::Crc16);
        assert_eq!("crc24".parse::<CrcScheme>().unwrap(), CrcScheme::Crc24);
        assert_eq!("crc32".parse::<CrcScheme>().unwrap(), CrcScheme::Crc32);

        // check length
        assert_eq!(CrcScheme::Unknown.key_len(), 0);
        assert_eq!(CrcScheme::None.key_len(), 0);
        assert_eq!(CrcScheme::Checksum.key_len(), 1);
        assert_eq!(CrcScheme::Crc8.key_len(), 1);
        assert_eq!(CrcScheme::Crc16.key_len(), 2);
        assert_eq!(CrcScheme::Crc24.key_len(), 3);
        assert_eq!(CrcScheme::Crc32.key_len(), 4);
    }

    #[test]
    fn test_crc_all_covers_schemes() {
        assert_eq!(CrcScheme::ALL.len() + 1, NUM_CRC_SCHEMES);
        assert!(!CrcScheme::ALL.contains(&CrcScheme::Unknown));

        let mut seen = std::collections::HashSet::new();
        for cs in CrcScheme::ALL {
            assert!(seen.insert(cs.short_name()), "{:?} listed twice", cs);
            assert_eq!(cs.short_name().parse::<CrcScheme>().unwrap(), cs);
            assert!(!cs.long_name().is_empty(), "{:?}", cs);
        }
    }
}
