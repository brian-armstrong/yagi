//
// FEC, repeat code
//

/// encode block of data using rep3 encoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
///  msg_enc        :   encoded message [size: 1 x 3*dec_msg_len]
pub fn rep3_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    let n = msg_dec.len();
    msg_enc[..n].copy_from_slice(msg_dec);
    msg_enc[n..2 * n].copy_from_slice(msg_dec);
    msg_enc[2 * n..3 * n].copy_from_slice(msg_dec);
}

/// decode block of data using rep3 decoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message [size: 1 x 3*dec_msg_len]
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
pub fn rep3_decode(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    for i in 0..dec_msg_len {
        let s0 = msg_enc[i];
        let s1 = msg_enc[i + dec_msg_len];
        let s2 = msg_enc[i + 2 * dec_msg_len];

        //  s0  s1  s2  |   y   e
        //  ------------+---------
        //  0   0   0   |   0   0
        //  0   0   1   |   0   1
        //  0   1   0   |   0   1
        //  0   1   1   |   1   1
        //  1   0   0   |   0   1
        //  1   0   1   |   1   1
        //  1   1   0   |   1   1
        //  1   1   1   |   1   0

        msg_dec[i] = (s0 & s1) | (s0 & s2) | (s1 & s2);
    }
}

/// decode block of data using rep3 decoder (soft metrics)
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message [size: 1 x 3*dec_msg_len]
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
pub fn rep3_decode_soft(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    for i in 0..dec_msg_len {
        // clear decoded message
        msg_dec[i] = 0;

        for j in 0..8 {
            let s0 = msg_enc[8 * i + j] as u32;
            let s1 = msg_enc[8 * (i + dec_msg_len) + j] as u32;
            let s2 = msg_enc[8 * (i + 2 * dec_msg_len) + j] as u32;

            // average three symbols and make decision
            let s_hat = (s0 + s1 + s2) / 3;

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
    fn test_rep3_encode() {
        let msg = [0x12, 0x34];
        let mut encoded = [0u8; 6];

        rep3_encode(&msg, &mut encoded);

        assert_eq!(encoded, [0x12, 0x34, 0x12, 0x34, 0x12, 0x34]);
    }

    #[test]
    fn test_rep3_decode_no_errors() {
        let encoded = [0x12, 0x34, 0x12, 0x34, 0x12, 0x34];
        let mut decoded = [0u8; 2];

        rep3_decode(2, &encoded, &mut decoded);

        assert_eq!(decoded, [0x12, 0x34]);
    }

    #[test]
    fn test_rep3_decode_one_error() {
        let encoded = [0x12, 0x34, 0xFF, 0xFF, 0x12, 0x34];
        let mut decoded = [0u8; 2];

        rep3_decode(2, &encoded, &mut decoded);

        assert_eq!(decoded, [0x12, 0x34]);
    }

    #[test]
    fn test_rep3_roundtrip() {
        let msg = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut encoded = [0u8; 12];
        let mut decoded = [0u8; 4];

        rep3_encode(&msg, &mut encoded);
        rep3_decode(4, &encoded, &mut decoded);

        assert_eq!(decoded, msg);
    }
}
