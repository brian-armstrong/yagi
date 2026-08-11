//
// SEC-DED forward error-correction block codes
//
// References:
//  [Lin:2004] Lin, Shu and Costello, Daniel L. Jr., "Error Control
//      Coding," Prentice Hall, New Jersey, 2nd edition, 2004.
//

/// zero/one/multiple errors detected, respectively
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecdedResult {
    NoErrors,
    SingleErrorCorrected,
    MultipleErrors,
}

struct SecdedCode {
    data_bytes: usize,
    parity_bits: usize,
    /// P matrix, stored row-major: `parity_bits` rows of `data_bytes` bytes
    p: &'static [u8],
    /// syndrome vectors for errors of weight 1
    syndrome_w1: &'static [u8],
}

impl SecdedCode {
    // compute parity on input
    fn compute_parity(&self, m: &[u8]) -> u8 {
        // compute encoded/transmitted message: v = m*G
        let mut parity = 0u8;

        for i in 0..self.parity_bits {
            parity <<= 1;

            let row = &self.p[i * self.data_bytes..(i + 1) * self.data_bytes];
            let mut p = 0u32;
            for (k, &row_byte) in row.iter().enumerate() {
                p += (row_byte & m[k]).count_ones();
            }

            parity |= (p & 0x01) as u8;
        }

        parity
    }

    // compute syndrome on input
    fn compute_syndrome(&self, v: &[u8]) -> u8 {
        let mut syndrome = 0u8;

        for i in 0..self.parity_bits {
            syndrome <<= 1;

            let parity_bit = if v[0] & (1 << (self.parity_bits - i - 1)) != 0 {
                1u32
            } else {
                0
            };

            let row = &self.p[i * self.data_bytes..(i + 1) * self.data_bytes];
            let mut p = parity_bit;
            for (k, &row_byte) in row.iter().enumerate() {
                p += (row_byte & v[k + 1]).count_ones();
            }

            syndrome |= (p & 0x01) as u8;
        }

        syndrome
    }

    fn estimate_ehat(&self, sym_enc: &[u8], e_hat: &mut [u8]) -> SecdedResult {
        let n = self.data_bytes + 1;

        // clear output array
        e_hat[..n].fill(0);

        // compute syndrome vector, s = r*H^T = ( H*r^T )^T
        let s = self.compute_syndrome(sym_enc);

        if s == 0 {
            // no errors detected
            return SecdedResult::NoErrors;
        }

        // estimate error location; search for syndrome with error
        // vector of weight one
        if let Some(pos) = self.syndrome_w1.iter().position(|&w| w == s) {
            // single error detected at location 'pos'
            let byte = pos / 8;
            let bit = pos % 8;
            e_hat[n - byte - 1] = 1 << bit;
            return SecdedResult::SingleErrorCorrected;
        }

        // no syndrome match; multiple errors detected
        SecdedResult::MultipleErrors
    }

    /// encode symbol
    ///
    ///  sym_dec    :   decoded symbol
    ///  sym_enc    :   encoded symbol, sym_enc[0] holds the parity bits
    fn encode_symbol(&self, sym_dec: &[u8], sym_enc: &mut [u8]) {
        // first bits are parity block
        sym_enc[0] = self.compute_parity(sym_dec);

        // copy remaining values
        sym_enc[1..=self.data_bytes].copy_from_slice(&sym_dec[..self.data_bytes]);
    }

    /// decode symbol, returning zero/one/multiple errors detected
    ///
    ///  sym_enc    :   encoded symbol, sym_enc[0] holds the parity bits
    ///  sym_dec    :   decoded symbol
    fn decode_symbol(&self, sym_enc: &[u8], sym_dec: &mut [u8]) -> SecdedResult {
        // estimate error vector
        let mut e_hat = [0u8; 9];
        let result = self.estimate_ehat(sym_enc, &mut e_hat);

        // compute estimated transmit vector
        // NOTE: indices take into account first element in sym_enc and e_hat
        //       arrays holds the parity bits
        for k in 0..self.data_bytes {
            sym_dec[k] = sym_enc[k + 1] ^ e_hat[k + 1];
        }

        result
    }

    /// encode block of data using SEC-DEC encoder
    ///
    ///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
    ///  msg_enc        :   encoded message
    fn encode(&self, msg_dec: &[u8], msg_enc: &mut [u8]) {
        let dec_msg_len = msg_dec.len();

        // determine remainder of input length
        let r = dec_msg_len % self.data_bytes;

        let mut i = 0usize; // decoded byte counter
        let mut j = 0usize; // encoded byte counter

        while i < dec_msg_len - r {
            // compute parity on input bytes
            msg_enc[j] = self.compute_parity(&msg_dec[i..]);

            // copy remaining input bytes
            msg_enc[j + 1..=j + self.data_bytes]
                .copy_from_slice(&msg_dec[i..i + self.data_bytes]);

            // increment output counter
            i += self.data_bytes;
            j += self.data_bytes + 1;
        }

        // if input length isn't divisible, encode last few bytes
        if r != 0 {
            let mut m = [0u8; 8];
            m[..r].copy_from_slice(&msg_dec[i..i + r]);

            // there is no need to actually send all the bytes;
            // the last bytes are zero and can be artificially
            // inserted at the decoder
            msg_enc[j] = self.compute_parity(&m);
            msg_enc[j + 1..=j + r].copy_from_slice(&msg_dec[i..i + r]);
        }
    }

    /// decode block of data using SEC-DEC decoder
    ///
    ///  dec_msg_len    :   decoded message length (number of bytes)
    ///  msg_enc        :   encoded message
    ///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
    fn decode(&self, dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
        // determine remainder of input length
        let r = dec_msg_len % self.data_bytes;

        let mut i = 0usize; // decoded byte counter
        let mut j = 0usize; // encoded byte counter

        while i < dec_msg_len - r {
            self.decode_symbol(&msg_enc[j..], &mut msg_dec[i..]);

            i += self.data_bytes;
            j += self.data_bytes + 1;
        }

        if r != 0 {
            // decode last symbol, artificially inserting the zeros
            // the encoder omitted
            let mut v = [0u8; 9];
            v[0] = msg_enc[j];
            v[1..=r].copy_from_slice(&msg_enc[j + 1..=j + r]);

            let mut m_hat = [0u8; 8];
            self.decode_symbol(&v, &mut m_hat);

            msg_dec[i..i + r].copy_from_slice(&m_hat[..r]);
        }
    }
}

// P matrix [6 x 16 bits], [6 x 2 bytes]
//  1001 1001 0011 1100 :
//  0011 1110 1000 1010 :
//  1110 1110 0110 0000 :
//  1110 0001 1101 0001 :
//  0001 0011 1100 0111 :
//  0100 0100 0011 1111 :
const SECDED2216: SecdedCode = SecdedCode {
    data_bytes: 2,
    parity_bits: 6,
    p: &[
        0x99, 0x3c,
        0x3e, 0x8a,
        0xee, 0x60,
        0xe1, 0xd1,
        0x13, 0xc7,
        0x44, 0x3f,
    ],
    syndrome_w1: &[
        0x07, 0x13, 0x23, 0x31, 0x25, 0x29, 0x0e, 0x16, 0x26, 0x1a, 0x19, 0x38, 0x32, 0x1c, 0x0d,
        0x2c, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
    ],
};

// P matrix [7 x 32 bits], [7 x 4 bytes]
//  1000 1010 1000 0010 0000 1111 0001 1011
//  0001 0000 0001 1111 0111 0001 0110 0001
//  0001 0110 1111 0000 1001 0010 1010 0110
//  1111 1111 0000 0001 1010 0100 0100 0100
//  0110 1100 1111 1111 0000 1000 0000 1000
//  0010 0001 0010 0100 1111 1111 1001 0000
//  1100 0001 0100 1000 0100 0000 1111 1111
const SECDED3932: SecdedCode = SecdedCode {
    data_bytes: 4,
    parity_bits: 7,
    p: &[
        0x8a, 0x82, 0x0f, 0x1b,
        0x10, 0x1f, 0x71, 0x61,
        0x16, 0xf0, 0x92, 0xa6,
        0xff, 0x01, 0xa4, 0x44,
        0x6c, 0xff, 0x08, 0x08,
        0x21, 0x24, 0xff, 0x90,
        0xc1, 0x48, 0x40, 0xff,
    ],
    syndrome_w1: &[
        0x61, 0x51, 0x19, 0x45, 0x43, 0x31, 0x29, 0x13, 0x62, 0x52, 0x4a, 0x46, 0x32, 0x2a, 0x23,
        0x1a, 0x2c, 0x64, 0x26, 0x25, 0x34, 0x16, 0x15, 0x54, 0x0b, 0x58, 0x1c, 0x4c, 0x38, 0x0e,
        0x0d, 0x49, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    ],
};

// P matrix [8 x 64]
//  11111111 00001111 00001111 00001100 01101000 10001000 10001000 10000000 :
//  11110000 11111111 00000000 11110011 01100100 01000100 01000100 01000000 :
//  00110000 11110000 11111111 00001111 00000010 00100010 00100010 00100110 :
//  11001111 00000000 11110000 11111111 00000001 00010001 00010001 00010110 :
//  01101000 10001000 10001000 10000000 11111111 00001111 00000000 11110011 :
//  01100100 01000100 01000100 01000000 11110000 11111111 00001111 00001100 :
//  00000010 00100010 00100010 00100110 11001111 00000000 11111111 00001111 :
//  00000001 00010001 00010001 00010110 00110000 11110000 11110000 11111111 :
const SECDED7264: SecdedCode = SecdedCode {
    data_bytes: 8,
    parity_bits: 8,
    p: &[
        0xFF, 0x0F, 0x0F, 0x0C, 0x68, 0x88, 0x88, 0x80, //
        0xF0, 0xFF, 0x00, 0xF3, 0x64, 0x44, 0x44, 0x40, //
        0x30, 0xF0, 0xFF, 0x0F, 0x02, 0x22, 0x22, 0x26, //
        0xCF, 0x00, 0xF0, 0xFF, 0x01, 0x11, 0x11, 0x16, //
        0x68, 0x88, 0x88, 0x80, 0xFF, 0x0F, 0x00, 0xF3, //
        0x64, 0x44, 0x44, 0x40, 0xF0, 0xFF, 0x0F, 0x0C, //
        0x02, 0x22, 0x22, 0x26, 0xCF, 0x00, 0xFF, 0x0F, //
        0x01, 0x11, 0x11, 0x16, 0x30, 0xF0, 0xF0, 0xFF,
    ],
    syndrome_w1: &[
        0x0b, 0x3b, 0x37, 0x07, 0x19, 0x29, 0x49, 0x89, 0x16, 0x26, 0x46, 0x86, 0x13, 0x23, 0x43,
        0x83, 0x1c, 0x2c, 0x4c, 0x8c, 0x15, 0x25, 0x45, 0x85, 0x1a, 0x2a, 0x4a, 0x8a, 0x0d, 0xcd,
        0xce, 0x0e, 0x70, 0x73, 0xb3, 0xb0, 0x51, 0x52, 0x54, 0x58, 0xa1, 0xa2, 0xa4, 0xa8, 0x31,
        0x32, 0x34, 0x38, 0xc1, 0xc2, 0xc4, 0xc8, 0x61, 0x62, 0x64, 0x68, 0x91, 0x92, 0x94, 0x98,
        0xe0, 0xec, 0xdc, 0xd0, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    ],
};

pub fn secded2216_encode_symbol(sym_dec: &[u8], sym_enc: &mut [u8]) {
    SECDED2216.encode_symbol(sym_dec, sym_enc)
}

pub fn secded2216_decode_symbol(sym_enc: &[u8], sym_dec: &mut [u8]) -> SecdedResult {
    SECDED2216.decode_symbol(sym_enc, sym_dec)
}

/// encode block of data using SEC-DEC (22,16) encoder
pub fn secded2216_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    SECDED2216.encode(msg_dec, msg_enc)
}

/// decode block of data using SEC-DEC (22,16) decoder
pub fn secded2216_decode(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    SECDED2216.decode(dec_msg_len, msg_enc, msg_dec)
}

// SEC-DED (39,32)

pub fn secded3932_encode_symbol(sym_dec: &[u8], sym_enc: &mut [u8]) {
    SECDED3932.encode_symbol(sym_dec, sym_enc)
}

pub fn secded3932_decode_symbol(sym_enc: &[u8], sym_dec: &mut [u8]) -> SecdedResult {
    SECDED3932.decode_symbol(sym_enc, sym_dec)
}

/// encode block of data using SEC-DEC (39,32) encoder
pub fn secded3932_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    SECDED3932.encode(msg_dec, msg_enc)
}

/// decode block of data using SEC-DEC (39,32) decoder
pub fn secded3932_decode(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    SECDED3932.decode(dec_msg_len, msg_enc, msg_dec)
}

// SEC-DED (72,64)

pub fn secded7264_encode_symbol(sym_dec: &[u8], sym_enc: &mut [u8]) {
    SECDED7264.encode_symbol(sym_dec, sym_enc)
}

pub fn secded7264_decode_symbol(sym_enc: &[u8], sym_dec: &mut [u8]) -> SecdedResult {
    SECDED7264.decode_symbol(sym_enc, sym_dec)
}

/// encode block of data using SEC-DEC (72,64) encoder
pub fn secded7264_encode(msg_dec: &[u8], msg_enc: &mut [u8]) {
    SECDED7264.encode(msg_dec, msg_enc)
}

/// decode block of data using SEC-DEC (72,64) decoder
pub fn secded7264_decode(dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) {
    SECDED7264.decode(dec_msg_len, msg_enc, msg_dec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    fn error_vector(n: usize, positions: &[usize]) -> [u8; 9] {
        let mut e = [0u8; 9];
        for &k in positions {
            e[n - k / 8 - 1] |= 1 << (k % 8);
        }
        e
    }

    fn codec_e0(code: &SecdedCode) {
        let n = code.data_bytes;
        let mut rng = rand::thread_rng();

        // generate symbol
        let sym_org: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

        // encoded symbol
        let mut sym_enc = [0u8; 9];
        code.encode_symbol(&sym_org, &mut sym_enc);

        // decoded symbol
        let mut sym_dec = [0u8; 8];
        let result = code.decode_symbol(&sym_enc, &mut sym_dec);

        // validate data are the same
        assert_eq!(result, SecdedResult::NoErrors);
        assert_eq!(&sym_dec[..n], &sym_org[..]);
    }

    fn codec_e1(code: &SecdedCode) {
        let n = code.data_bytes;
        let bits = code.data_bytes * 8 + code.parity_bits;
        let mut rng = rand::thread_rng();

        for k in 0..bits {
            // generate symbol
            let sym_org: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

            // encoded symbol
            let mut sym_enc = [0u8; 9];
            code.encode_symbol(&sym_org, &mut sym_enc);

            // generate error vector (single error)
            let e = error_vector(n + 1, &[k]);

            // received symbol
            let sym_rec: Vec<u8> = (0..=n).map(|i| sym_enc[i] ^ e[i]).collect();

            // decoded symbol
            let mut sym_dec = [0u8; 8];
            code.decode_symbol(&sym_rec, &mut sym_dec);

            // validate data are the same
            assert_eq!(
                &sym_dec[..n],
                &sym_org[..],
                "failed to correct single error at bit {}",
                k
            );
        }
    }

    fn codec_e2(code: &SecdedCode) {
        let n = code.data_bytes;
        let bits = code.data_bytes * 8 + code.parity_bits;
        let mut rng = rand::thread_rng();

        for j in 0..bits - 1 {
            for k in (j + 1)..bits {
                // generate symbol
                let sym_org: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

                // encoded symbol
                let mut sym_enc = [0u8; 9];
                code.encode_symbol(&sym_org, &mut sym_enc);

                // generate error vector (double error)
                let e = error_vector(n + 1, &[j, k]);

                // received symbol
                let sym_rec: Vec<u8> = (0..=n).map(|i| sym_enc[i] ^ e[i]).collect();

                // decoded symbol
                let mut sym_dec = [0u8; 8];
                let result = code.decode_symbol(&sym_rec, &mut sym_dec);

                // validate that the error was detected
                assert_eq!(
                    result,
                    SecdedResult::MultipleErrors,
                    "double error at bits {},{} was not detected",
                    j,
                    k
                );
            }
        }
    }

    fn block_roundtrip(
        encode: fn(&[u8], &mut [u8]),
        decode: fn(usize, &[u8], &mut [u8]),
        data_bytes: usize,
    ) {
        for n in 1..=(3 * data_bytes + 1) {
            let msg: Vec<u8> = (0..n)
                .map(|i| (i as u8).wrapping_mul(53).wrapping_add(7))
                .collect();

            let r = n % data_bytes;
            let enc_len = (n - r) / data_bytes * (data_bytes + 1) + if r != 0 { r + 1 } else { 0 };

            let mut encoded = vec![0u8; enc_len];
            let mut decoded = vec![0u8; n];

            encode(&msg, &mut encoded);
            decode(n, &encoded, &mut decoded);

            assert_eq!(msg, decoded, "block round trip failed for length {}", n);
        }
    }

    // SEC-DED (22,16)

    #[test]
    #[autotest_annotate(autotest_secded2216_codec_e0)]
    fn test_secded2216_codec_e0() {
        codec_e0(&SECDED2216);
    }

    #[test]
    #[autotest_annotate(autotest_secded2216_codec_e1)]
    fn test_secded2216_codec_e1() {
        codec_e1(&SECDED2216);
    }

    #[test]
    #[autotest_annotate(autotest_secded2216_codec_e2)]
    fn test_secded2216_codec_e2() {
        // total combinations of double errors: nchoosek(22,2) = 231
        codec_e2(&SECDED2216);
    }

    #[test]
    fn test_secded2216_block_roundtrip() {
        block_roundtrip(secded2216_encode, secded2216_decode, 2);
    }

    // SEC-DED (39,32)

    #[test]
    #[autotest_annotate(autotest_secded3932_codec_e0)]
    fn test_secded3932_codec_e0() {
        codec_e0(&SECDED3932);
    }

    #[test]
    #[autotest_annotate(autotest_secded3932_codec_e1)]
    fn test_secded3932_codec_e1() {
        codec_e1(&SECDED3932);
    }

    #[test]
    #[autotest_annotate(autotest_secded3932_codec_e2)]
    fn test_secded3932_codec_e2() {
        // total combinations of double errors: nchoosek(39,2) = 741
        codec_e2(&SECDED3932);
    }

    #[test]
    fn test_secded3932_block_roundtrip() {
        block_roundtrip(secded3932_encode, secded3932_decode, 4);
    }

    // SEC-DED (72,64)

    #[test]
    #[autotest_annotate(autotest_secded7264_codec_e0)]
    fn test_secded7264_codec_e0() {
        codec_e0(&SECDED7264);
    }

    #[test]
    #[autotest_annotate(autotest_secded7264_codec_e1)]
    fn test_secded7264_codec_e1() {
        codec_e1(&SECDED7264);
    }

    #[test]
    #[autotest_annotate(autotest_secded7264_codec_e2)]
    fn test_secded7264_codec_e2() {
        // total combinations of double errors: nchoosek(72,2) = 2556
        codec_e2(&SECDED7264);
    }

    #[test]
    fn test_secded7264_block_roundtrip() {
        block_roundtrip(secded7264_encode, secded7264_decode, 8);
    }

    #[test]
    fn test_secded_syndrome_tables_consistent() {
        for code in [&SECDED2216, &SECDED3932, &SECDED7264] {
            let n = code.data_bytes;
            let bits = n * 8 + code.parity_bits;

            let zero = [0u8; 8];
            let mut sym_enc = [0u8; 9];
            code.encode_symbol(&zero[..n], &mut sym_enc);

            for k in 0..bits {
                let e = error_vector(n + 1, &[k]);
                let sym_rec: Vec<u8> = (0..=n).map(|i| sym_enc[i] ^ e[i]).collect();

                assert_eq!(
                    code.compute_syndrome(&sym_rec),
                    code.syndrome_w1[k],
                    "syndrome table mismatch at bit {} for a {}-byte code",
                    k,
                    n
                );
            }
        }
    }
}
