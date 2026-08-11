//
// FEC, repeat code
//

/// encode block of data using rep5 encoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
///  msg_enc        :   encoded message [size: 1 x 5*dec_msg_len]
pub fn rep5_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    let n = msg_dec.len();
    for i in 0..5 {
        msg_enc[i * n..(i + 1) * n].copy_from_slice(msg_dec);
    }
}

/// decode block of data using rep5 decoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message [size: 1 x 5*dec_msg_len]
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
pub fn rep5_decode(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    for i in 0..dec_msg_len {
        let s0 = msg_enc[i];
        let s1 = msg_enc[i + dec_msg_len];
        let s2 = msg_enc[i + 2 * dec_msg_len];
        let s3 = msg_enc[i + 3 * dec_msg_len];
        let s4 = msg_enc[i + 4 * dec_msg_len];

        // compute all triplet combinations
        msg_dec[i] = (s0 & s1 & s2)
            | (s0 & s1 & s3)
            | (s0 & s1 & s4)
            | (s0 & s2 & s3)
            | (s0 & s2 & s4)
            | (s0 & s3 & s4)
            | (s1 & s2 & s3)
            | (s1 & s2 & s4)
            | (s1 & s3 & s4)
            | (s2 & s3 & s4);
    }
}

/// decode block of data using rep5 decoder (soft metrics)
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message [size: 1 x 5*dec_msg_len]
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
pub fn rep5_decode_soft(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    for i in 0..dec_msg_len {
        // clear decoded message
        msg_dec[i] = 0;

        for j in 0..8 {
            let s0 = msg_enc[8 * i + j] as u32;
            let s1 = msg_enc[8 * (i + dec_msg_len) + j] as u32;
            let s2 = msg_enc[8 * (i + 2 * dec_msg_len) + j] as u32;
            let s3 = msg_enc[8 * (i + 3 * dec_msg_len) + j] as u32;
            let s4 = msg_enc[8 * (i + 4 * dec_msg_len) + j] as u32;

            // average three symbols and make decision
            let s_hat = (s0 + s1 + s2 + s3 + s4) / 5;

            if s_hat > 127 {
                msg_dec[i] |= 1 << (7 - j);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rep5_encode() {
        let msg = [0xAB];
        let mut encoded = [0u8; 5];

        rep5_encode(&msg, &mut encoded);

        assert_eq!(encoded, [0xAB, 0xAB, 0xAB, 0xAB, 0xAB]);
    }

    #[test]
    fn test_rep5_decode_no_errors() {
        let encoded = [0xAB, 0xCD, 0xAB, 0xCD, 0xAB, 0xCD, 0xAB, 0xCD, 0xAB, 0xCD];
        let mut decoded = [0u8; 2];

        rep5_decode(2, &encoded, &mut decoded);

        assert_eq!(decoded, [0xAB, 0xCD]);
    }

    #[test]
    fn test_rep5_decode_two_errors() {
        let encoded = [0xAB, 0xFF, 0x00, 0xAB, 0xAB];
        let mut decoded = [0u8; 1];

        rep5_decode(1, &encoded, &mut decoded);

        assert_eq!(decoded, [0xAB]);
    }

    #[test]
    fn test_rep5_roundtrip() {
        let msg = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut encoded = [0u8; 20];
        let mut decoded = [0u8; 4];

        rep5_encode(&msg, &mut encoded);
        rep5_decode(4, &encoded, &mut decoded);

        assert_eq!(decoded, msg);
    }
}
