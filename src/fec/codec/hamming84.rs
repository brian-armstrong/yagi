//
// 1/2-rate (8,4) Hamming code
//

// encoder look-up table
const ENC_GENTAB: [u8; 16] = [
    0x00, 0xd2, 0x55, 0x87, 0x99, 0x4b, 0xcc, 0x1e,
    0xe1, 0x33, 0xb4, 0x66, 0x78, 0xaa, 0x2d, 0xff,
];

// decoder look-up table
const DEC_GENTAB: [u8; 256] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x03,
    0x00, 0x00, 0x05, 0x05, 0x0e, 0x0e, 0x07, 0x07,
    0x00, 0x00, 0x09, 0x09, 0x02, 0x02, 0x07, 0x07,
    0x04, 0x04, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x00, 0x00, 0x09, 0x09, 0x0e, 0x0e, 0x0b, 0x0b,
    0x0e, 0x0e, 0x0d, 0x0d, 0x0e, 0x0e, 0x0e, 0x0e,
    0x09, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x09, 0x09,
    0x0c, 0x0c, 0x09, 0x09, 0x0e, 0x0e, 0x07, 0x07,
    0x00, 0x00, 0x05, 0x05, 0x02, 0x02, 0x0b, 0x0b,
    0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x05, 0x05,
    0x02, 0x02, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02,
    0x0c, 0x0c, 0x05, 0x05, 0x02, 0x02, 0x07, 0x07,
    0x08, 0x08, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b,
    0x0c, 0x0c, 0x05, 0x05, 0x0e, 0x0e, 0x0b, 0x0b,
    0x0c, 0x0c, 0x09, 0x09, 0x02, 0x02, 0x0b, 0x0b,
    0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0f, 0x0f,
    0x00, 0x00, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x04, 0x04, 0x0d, 0x0d, 0x06, 0x06, 0x03, 0x03,
    0x04, 0x04, 0x01, 0x01, 0x0a, 0x0a, 0x03, 0x03,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x07, 0x07,
    0x08, 0x08, 0x0d, 0x0d, 0x0a, 0x0a, 0x03, 0x03,
    0x0d, 0x0d, 0x0d, 0x0d, 0x0e, 0x0e, 0x0d, 0x0d,
    0x0a, 0x0a, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a,
    0x04, 0x04, 0x0d, 0x0d, 0x0a, 0x0a, 0x0f, 0x0f,
    0x08, 0x08, 0x01, 0x01, 0x06, 0x06, 0x03, 0x03,
    0x06, 0x06, 0x05, 0x05, 0x06, 0x06, 0x06, 0x06,
    0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x01, 0x01,
    0x04, 0x04, 0x01, 0x01, 0x06, 0x06, 0x0f, 0x0f,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0b, 0x0b,
    0x08, 0x08, 0x0d, 0x0d, 0x06, 0x06, 0x0f, 0x0f,
    0x08, 0x08, 0x01, 0x01, 0x0a, 0x0a, 0x0f, 0x0f,
    0x0c, 0x0c, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
];

/// encode block of data using Hamming(8,4) encoder
///
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
///  msg_enc        :   encoded message [size: 1 x 2*dec_msg_len]
pub fn hamming84_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    let dec_msg_len = msg_dec.len();
    let mut j = 0usize;

    for i in 0..dec_msg_len {
        let s0 = (msg_dec[i] >> 4) & 0x0f;
        let s1 = msg_dec[i] & 0x0f;

        msg_enc[j] = ENC_GENTAB[s0 as usize];
        msg_enc[j + 1] = ENC_GENTAB[s1 as usize];

        j += 2;
    }
}

/// decode block of data using Hamming(8,4) decoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message [size: 1 x 2*dec_msg_len]
///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
pub fn hamming84_decode(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    for i in 0..dec_msg_len {
        let r0 = msg_enc[2 * i];
        let r1 = msg_enc[2 * i + 1];

        let s0 = DEC_GENTAB[r0 as usize];
        let s1 = DEC_GENTAB[r1 as usize];

        msg_dec[i] = (s0 << 4) | s1;
    }
}

// soft decoding of one symbol
fn soft_decode_symbol(soft_bits: &[u8]) -> u8 {
    // find symbol with minimum distance from all 2^4 possible
    let mut dmin = u32::MAX;
    let mut s_hat = 0u8;

    for s in 0u8..16 {
        // encode symbol
        let c = ENC_GENTAB[s as usize];

        // compute distance metric
        let mut d = 0u32;
        d += if c & 0x80 != 0 { 255 - soft_bits[0] as u32 } else { soft_bits[0] as u32 };
        d += if c & 0x40 != 0 { 255 - soft_bits[1] as u32 } else { soft_bits[1] as u32 };
        d += if c & 0x20 != 0 { 255 - soft_bits[2] as u32 } else { soft_bits[2] as u32 };
        d += if c & 0x10 != 0 { 255 - soft_bits[3] as u32 } else { soft_bits[3] as u32 };
        d += if c & 0x08 != 0 { 255 - soft_bits[4] as u32 } else { soft_bits[4] as u32 };
        d += if c & 0x04 != 0 { 255 - soft_bits[5] as u32 } else { soft_bits[5] as u32 };
        d += if c & 0x02 != 0 { 255 - soft_bits[6] as u32 } else { soft_bits[6] as u32 };
        d += if c & 0x01 != 0 { 255 - soft_bits[7] as u32 } else { soft_bits[7] as u32 };

        if d < dmin {
            s_hat = s;
            dmin = d;
        }
    }

    s_hat
}

/// decode block of data using Hamming(8,4) soft decoder
///
///  dec_msg_len    :   decoded message length (number of bytes)
///  msg_enc        :   encoded message [size: 8*enc_msg_len x 1]
///  msg_dec        :   decoded message [size: dec_msg_len x 1]
pub fn hamming84_decode_soft(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    let mut k = 0usize; // array bit index

    for i in 0..dec_msg_len {
        let s0 = soft_decode_symbol(&msg_enc[k..]);
        let s1 = soft_decode_symbol(&msg_enc[k + 8..]);
        k += 16;

        // pack two 4-bit symbols into one 8-bit byte
        msg_dec[i] = (s0 << 4) | s1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::{Fec, FecScheme};
    use test_macro::autotest_annotate;

    #[test]
    #[autotest_annotate(autotest_hamming84_codec)]
    fn test_hamming84_codec() {
        let n = 4;
        let msg: [u8; 4] = [0x25, 0x62, 0x3F, 0x52];
        let fs = FecScheme::Hamming84;

        // create arrays
        let n_enc = fs.enc_msg_len(n);
        let mut msg_enc = vec![0u8; n_enc];
        let mut msg_dec = [0u8; 4];

        // create object
        let mut q = Fec::new(fs).unwrap();

        // encode message
        q.encode(&msg, &mut msg_enc).unwrap();

        // corrupt encoded message
        msg_enc[0] ^= 0x04; // position 5
        msg_enc[1] ^= 0x04;
        msg_enc[2] ^= 0x02;
        msg_enc[3] ^= 0x01;
        msg_enc[4] ^= 0x80;
        msg_enc[5] ^= 0x40;
        msg_enc[6] ^= 0x20;
        msg_enc[7] ^= 0x10;

        // decode message
        q.decode(n, &msg_enc, &mut msg_dec).unwrap();

        // validate data are the same
        assert_eq!(msg, msg_dec);
    }

    #[test]
    #[autotest_annotate(autotest_hamming84_codec_soft)]
    fn test_hamming84_codec_soft() {
        // generate each of the 2^4=16 symbols, encode, and decode
        // using soft decoding algorithm
        for s in 0u8..16 {
            // encode using look-up table
            let c = ENC_GENTAB[s as usize];

            // expand soft bits
            let mut c_soft = [0u8; 8];
            for (i, bit) in c_soft.iter_mut().enumerate() {
                *bit = if c & (0x80 >> i) != 0 { 255 } else { 0 };
            }

            // decode using internal soft decoding method
            let s_hat = soft_decode_symbol(&c_soft);

            assert_eq!(s, s_hat);
        }
    }
}
