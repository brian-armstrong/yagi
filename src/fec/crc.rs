//
// cyclic redundancy check (and family)
//

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrcScheme {
    /// unknown/unavailable CRC scheme
    Unknown,
    /// no error-detection
    None,
    /// 8-bit checksum
    Checksum,
    /// 8-bit CRC
    Crc8,
    /// 16-bit CRC
    Crc16,
    /// 24-bit CRC
    Crc24,
    /// 32-bit CRC
    Crc32,
}

impl CrcScheme {
    /// returns crc_scheme based on input string
    pub fn from_str(s: &str) -> Self {
        match s {
            "none" => CrcScheme::None,
            "checksum" => CrcScheme::Checksum,
            "crc8" => CrcScheme::Crc8,
            "crc16" => CrcScheme::Crc16,
            "crc24" => CrcScheme::Crc24,
            "crc32" => CrcScheme::Crc32,
            _ => CrcScheme::Unknown,
        }
    }

    /// short name
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

    /// long name
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

    /// get length of CRC (bytes)
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

/// generate error-detection key
///
///  scheme     :   error-detection scheme
///  msg        :   input data message
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

/// generate error-detection key and append to end of message
///
///  scheme     :   error-detection scheme (resulting in 'p' bytes)
///  msg        :   input data message, [size: msg_len+p x 1]
///  msg_len    :   input data message size (excluding key at end)
pub fn append_key(scheme: CrcScheme, msg: &mut [u8], msg_len: usize) {
    let key_len = scheme.key_len();
    let key = generate_key(scheme, &msg[..msg_len]);

    for i in 0..key_len {
        msg[msg_len + i] = ((key >> ((key_len - i - 1) * 8)) & 0xff) as u8;
    }
}

/// validate message using error-detection key
///
///  scheme     :   error-detection scheme
///  msg        :   input data message
///  key        :   error-detection key
pub fn validate_message(scheme: CrcScheme, msg: &[u8], key: u32) -> bool {
    match scheme {
        CrcScheme::Unknown => false,
        CrcScheme::None => true,
        _ => generate_key(scheme, msg) == key,
    }
}

/// check message with key appended to end of array
///
///  scheme     :   error-detection scheme (resulting in 'p' bytes)
///  msg        :   input data message, [size: msg_len+p x 1]
///  msg_len    :   input data message size (excluding key at end)
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

/// generate 8-bit checksum key
///
///  msg        :   input data message
pub fn checksum_generate_key(data: &[u8]) -> u8 {
    let sum: u32 = data.iter().map(|&b| b as u32).sum();
    // mask and convert to 2's complement
    (!(sum as u8)).wrapping_add(1)
}

//
// CRC-8
//

/// generate 8-bit cyclic redundancy check key.
///
///  msg        :   input data message
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

/// generate 16-bit cyclic redundancy check key.
///
///  msg        :   input data message
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

/// generate 24-bit cyclic redundancy check key.
///
///  msg        :   input data message
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

/// generate 32-bit cyclic redundancy check key.
///
///  msg        :   input data message
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
    use crate::sequence::MSequence;
    use test_macro::autotest_annotate;

    fn crc_test(scheme: CrcScheme, n: usize) {
        // generate pseudo-random data
        let mut ms = MSequence::create_default(9).unwrap();
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
        assert_eq!(CrcScheme::from_str("unknown"), CrcScheme::Unknown);
        assert_eq!(CrcScheme::from_str("rosebud"), CrcScheme::Unknown);
        assert_eq!(CrcScheme::from_str("none"), CrcScheme::None);
        assert_eq!(CrcScheme::from_str("checksum"), CrcScheme::Checksum);
        assert_eq!(CrcScheme::from_str("crc8"), CrcScheme::Crc8);
        assert_eq!(CrcScheme::from_str("crc16"), CrcScheme::Crc16);
        assert_eq!(CrcScheme::from_str("crc24"), CrcScheme::Crc24);
        assert_eq!(CrcScheme::from_str("crc32"), CrcScheme::Crc32);

        // check length
        assert_eq!(CrcScheme::Unknown.key_len(), 0);
        assert_eq!(CrcScheme::None.key_len(), 0);
        assert_eq!(CrcScheme::Checksum.key_len(), 1);
        assert_eq!(CrcScheme::Crc8.key_len(), 1);
        assert_eq!(CrcScheme::Crc16.key_len(), 2);
        assert_eq!(CrcScheme::Crc24.key_len(), 3);
        assert_eq!(CrcScheme::Crc32.key_len(), 4);
    }
}
