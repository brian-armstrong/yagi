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

    /// Returns the scheme whose integer index is `index`, or `None` when
    /// `index` names no scheme.
    ///
    /// This turns a scheme stored as a number, such as in a frame header,
    /// back into the scheme. `index` 0 is [`Self::Unknown`], which this
    /// rejects as not a usable scheme.
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|s| *s as usize == index)
    }

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
/// `packet` holds the message followed by `scheme.key_len()` bytes reserved
/// for the key. The key is written in those last bytes, most significant byte
/// first. Pass a subslice when the backing buffer has extra space.
///
/// # Errors
///
/// Returns an error when `packet` is shorter than the key.
pub fn append_key(scheme: CrcScheme, packet: &mut [u8]) -> Result<()> {
    let key_len = scheme.key_len();
    let msg_len = packet
        .len()
        .checked_sub(key_len)
        .ok_or_else(|| Error::Range("packet is shorter than CRC key".into()))?;
    let (msg, key_bytes) = packet.split_at_mut(msg_len);
    let key = generate_key(scheme, msg);

    for (i, byte) in key_bytes.iter_mut().enumerate() {
        *byte = ((key >> ((key_len - i - 1) * 8)) & 0xff) as u8;
    }
    Ok(())
}

/// Returns whether `key` matches the key generated for `msg`.
///
/// [`CrcScheme::None`] always returns `true` and [`CrcScheme::Unknown`]
/// always returns `false`. A match does not prove the message is error-free,
/// because different messages can share a key.
///
pub fn validate_message(scheme: CrcScheme, msg: &[u8], key: u32) -> bool {
    match scheme {
        CrcScheme::Unknown => false,
        CrcScheme::None => true,
        _ => generate_key(scheme, msg) == key,
    }
}

/// Returns whether the key written after a message matches it.
///
/// `packet` holds the message followed by its `scheme.key_len()`-byte key,
/// most significant byte first, as [`append_key`] writes it. Pass a subslice
/// when the backing buffer has extra space. See [`validate_message`] for how
/// the key is compared.
///
/// # Errors
///
/// Returns an error when `packet` is shorter than the key.
pub fn check_key(scheme: CrcScheme, packet: &[u8]) -> Result<bool> {
    let key_len = scheme.key_len();
    let msg_len = packet
        .len()
        .checked_sub(key_len)
        .ok_or_else(|| Error::Range("packet is shorter than CRC key".into()))?;
    let (msg, key_bytes) = packet.split_at(msg_len);

    // extract key from end of message
    let mut key = 0u32;
    for &byte in key_bytes {
        key = (key << 8) | (byte as u32);
    }

    Ok(validate_message(scheme, msg, key))
}

//
// Checksum
//

/// Generates the 8-bit checksum for `data`.
///
/// The checksum is the two's complement of the byte sum modulo 256, so the
/// bytes and the checksum sum to zero modulo 256.
///
pub fn checksum_generate_key(data: &[u8]) -> u8 {
    let sum = data.iter().fold(0u8, |sum, &byte| sum.wrapping_add(byte));
    // convert to 2's complement
    (!sum).wrapping_add(1)
}

//
// CRC-8
//

/// Generates the 8-bit CRC for `msg`.
///
/// The CRC uses polynomial `0x07`, processes bits least significant first,
/// starts from all ones, and inverts the result. The check value for
/// `"123456789"` is `0x2F`.
///
/// With the `liquid-quirks` feature, the register starts as 32 bits of ones
/// to match liquid-dsp. The extra ones shift into the CRC over the first
/// three bytes, and the check value becomes `0x23`.
pub fn crc8_generate_key(msg: &[u8]) -> u8 {
    #[cfg(feature = "liquid-quirks")]
    let mut crc = 0xffff_ffffu32;
    #[cfg(not(feature = "liquid-quirks"))]
    let mut crc = 0xffu32;
    for &b in msg {
        crc = CRC8_TABLE[((crc ^ b as u32) & 0xff) as usize] as u32 ^ (crc >> 8);
    }
    !crc as u8
}

//
// CRC-16
//

/// Generates the 16-bit CRC for `msg`.
///
/// The CRC uses polynomial `0x8005`, processes bits least significant first,
/// starts from all ones, and inverts the result. These are the CRC-16/USB
/// parameters, with check value `0xB4C8` for `"123456789"`.
///
/// With the `liquid-quirks` feature, the register starts as 32 bits of ones
/// to match liquid-dsp. The extra ones shift into the CRC over the first two
/// bytes, so the result differs from CRC-16/USB for any nonempty message,
/// and the check value becomes `0xB4D3`.
pub fn crc16_generate_key(msg: &[u8]) -> u16 {
    #[cfg(feature = "liquid-quirks")]
    let mut crc = 0xffff_ffffu32;
    #[cfg(not(feature = "liquid-quirks"))]
    let mut crc = 0xffffu32;
    for &b in msg {
        crc = CRC16_TABLE[((crc ^ b as u32) & 0xff) as usize] as u32 ^ (crc >> 8);
    }
    !crc as u16
}

//
// CRC-24
//

/// Generates the 24-bit CRC for `msg` in the low 24 bits.
///
/// The CRC uses polynomial `0x5D6DCB`, processes bits least significant
/// first, starts from all ones, and inverts the result. The check value for
/// `"123456789"` is `0x2F27EE`.
///
/// With the `liquid-quirks` feature, the register starts as 32 bits of ones
/// to match liquid-dsp. The extra ones shift into the CRC on the first byte,
/// and the check value becomes `0x062BB3`.
pub fn crc24_generate_key(msg: &[u8]) -> u32 {
    #[cfg(feature = "liquid-quirks")]
    let mut crc = 0xffff_ffffu32;
    #[cfg(not(feature = "liquid-quirks"))]
    let mut crc = 0xff_ffffu32;
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
    fn crc_appended_key_uses_end_of_packet() {
        let message = b"123456789";
        for scheme in CrcScheme::ALL {
            let packet_len = message.len() + scheme.key_len();
            let mut backing = vec![0xa5; packet_len + 2];
            backing[..message.len()].copy_from_slice(message);
            append_key(scheme, &mut backing[..packet_len]).unwrap();

            let expected_key = generate_key(scheme, message).to_be_bytes();
            assert_eq!(&backing[message.len()..packet_len], &expected_key[4 - scheme.key_len()..]);
            assert_eq!(&backing[packet_len..], &[0xa5, 0xa5]);
            assert!(check_key(scheme, &backing[..packet_len]).unwrap());

            if scheme.key_len() != 0 {
                backing[0] ^= 1;
                assert!(!check_key(scheme, &backing[..packet_len]).unwrap());
            }
        }

        let mut unknown = message.to_vec();
        append_key(CrcScheme::Unknown, &mut unknown).unwrap();
        assert_eq!(unknown, message);
        assert!(!check_key(CrcScheme::Unknown, &unknown).unwrap());
    }

    #[test]
    fn crc_appended_key_rejects_short_packet() {
        let mut packet = [0xa5];
        assert!(matches!(append_key(CrcScheme::Crc32, &mut packet), Err(Error::Range(_))));
        assert_eq!(packet, [0xa5]);
        assert!(matches!(check_key(CrcScheme::Crc32, &packet), Err(Error::Range(_))));
    }

    #[test]
    #[autotest_annotate(autotest_checksum)]
    fn test_checksum() {
        crc_test(CrcScheme::Checksum, 16);
    }

    #[test]
    fn test_checksum_wraps_long_message() {
        let data = vec![255u8; u32::MAX as usize / 255 + 1];
        assert_eq!(checksum_generate_key(&data), data.len() as u8);
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
