//
// FEC, none/pass
//

pub fn pass_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    msg_enc[..msg_dec.len()].copy_from_slice(msg_dec);
}

pub fn pass_decode(msg_enc: &[u8], msg_dec: &mut [u8]) {
    msg_dec[..msg_enc.len()].copy_from_slice(msg_enc);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_encode_decode() {
        let msg = [0x12, 0x34, 0x56, 0x78];
        let mut encoded = [0u8; 4];
        let mut decoded = [0u8; 4];

        pass_encode(&msg, &mut encoded);
        assert_eq!(encoded, msg);

        pass_decode(&encoded, &mut decoded);
        assert_eq!(decoded, msg);
    }

}
