//
// forward error-correction encoder/decoder
//

use crate::error::{Error, Result};

use super::codec::{
    golay2412_decode, golay2412_encode, hamming128_decode, hamming128_decode_soft,
    hamming128_encode, hamming74_decode,
    hamming74_decode_soft, hamming74_encode, hamming84_decode, hamming84_decode_soft,
    hamming84_encode, pass_decode, pass_encode, rep3_decode, rep3_decode_soft,
    rep3_encode, rep5_decode, rep5_decode_soft, rep5_encode, secded2216_decode, secded2216_encode,
    secded3932_decode, secded3932_encode, secded7264_decode, secded7264_encode,
    conv_scheme_params, Convolutional, ReedSolomon,
};
use super::scheme::FecScheme;

#[derive(Debug, Clone)]
enum FecData {
    Convolutional(Convolutional),
    ReedSolomon(ReedSolomon),
}

/// forward error-correction encoder/decoder
#[derive(Debug, Clone)]
pub struct Fec {
    scheme: FecScheme,
    rate: f32,
    data: Option<FecData>,
}

impl Fec {
    /// create fec object
    pub fn new(scheme: FecScheme) -> Result<Self> {
        let data = match scheme {
            FecScheme::Unknown => {
                return Err(Error::Config("cannot create FEC with unknown scheme".into()))
            }
            FecScheme::None
            | FecScheme::Rep3
            | FecScheme::Rep5
            | FecScheme::Hamming74
            | FecScheme::Hamming84
            | FecScheme::Hamming128
            | FecScheme::Golay2412
            | FecScheme::Secded2216
            | FecScheme::Secded3932
            | FecScheme::Secded7264 => None,
            FecScheme::RsM8 => Some(FecData::ReedSolomon(ReedSolomon::new_m8())),
            _ => {
                // convolutional, punctured or otherwise
                let (params, matrix) = conv_scheme_params(scheme)
                    .ok_or_else(|| Error::Config(format!("unhandled scheme {:?}", scheme)))?;
                Some(FecData::Convolutional(Convolutional::new(params, matrix)))
            }
        };

        Ok(Self {
            scheme,
            rate: scheme.rate(),
            data,
        })
    }

    /// get scheme
    pub fn scheme(&self) -> FecScheme {
        self.scheme
    }

    /// get rate
    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// get encoded message length
    pub fn enc_msg_len(&self, dec_msg_len: usize) -> usize {
        self.scheme.enc_msg_len(dec_msg_len)
    }

    /// encode a block of data using a fec scheme
    ///
    ///  msg_dec        :   decoded message
    ///  msg_enc        :   encoded message, at least enc_msg_len() bytes
    pub fn encode(&mut self, msg_dec: &[u8], msg_enc: &mut [u8]) -> Result<()> {
        let dec_len = msg_dec.len();
        let enc_len = self.enc_msg_len(dec_len);

        if msg_enc.len() < enc_len {
            return Err(Error::Config(format!(
                "encoded buffer too small: {} < {}",
                msg_enc.len(),
                enc_len
            )));
        }

        match self.scheme {
            FecScheme::Unknown => {
                return Err(Error::Config("cannot encode with unknown scheme".into()))
            }
            FecScheme::None => pass_encode(msg_dec, msg_enc),
            FecScheme::Rep3 => rep3_encode(msg_dec, msg_enc),
            FecScheme::Rep5 => rep5_encode(msg_dec, msg_enc),
            FecScheme::Hamming74 => hamming74_encode(msg_dec, msg_enc),
            FecScheme::Hamming84 => hamming84_encode(msg_dec, msg_enc),
            FecScheme::Hamming128 => hamming128_encode(msg_dec, msg_enc),
            FecScheme::Golay2412 => golay2412_encode(msg_dec, msg_enc),
            FecScheme::Secded2216 => secded2216_encode(msg_dec, msg_enc),
            FecScheme::Secded3932 => secded3932_encode(msg_dec, msg_enc),
            FecScheme::Secded7264 => secded7264_encode(msg_dec, msg_enc),
            FecScheme::RsM8 => match &mut self.data {
                Some(FecData::ReedSolomon(rs)) => rs.encode(msg_dec, msg_enc)?,
                _ => return Err(Error::Config("Reed-Solomon state missing".into())),
            },
            _ => match &mut self.data {
                Some(FecData::Convolutional(c)) => c.encode(msg_dec, msg_enc)?,
                _ => return Err(Error::Config("convolutional state missing".into())),
            },
        }

        Ok(())
    }

    /// decode a block of data using a fec scheme
    ///
    ///  dec_msg_len    :   decoded message length (number of bytes)
    ///  msg_enc        :   encoded message
    ///  msg_dec        :   decoded message, at least dec_msg_len bytes
    pub fn decode(&mut self, dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) -> Result<()> {
        let enc_len = self.enc_msg_len(dec_msg_len);

        if msg_enc.len() < enc_len {
            return Err(Error::Config(format!(
                "encoded buffer too small: {} < {}",
                msg_enc.len(),
                enc_len
            )));
        }

        if msg_dec.len() < dec_msg_len {
            return Err(Error::Config(format!(
                "decoded buffer too small: {} < {}",
                msg_dec.len(),
                dec_msg_len
            )));
        }

        match self.scheme {
            FecScheme::Unknown => {
                return Err(Error::Config("cannot decode with unknown scheme".into()))
            }
            FecScheme::None => pass_decode(&msg_enc[..dec_msg_len], msg_dec),
            FecScheme::Rep3 => rep3_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Rep5 => rep5_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Hamming74 => hamming74_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Hamming84 => hamming84_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Hamming128 => hamming128_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Golay2412 => golay2412_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Secded2216 => secded2216_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Secded3932 => secded3932_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Secded7264 => secded7264_decode(dec_msg_len, msg_enc, msg_dec),
            FecScheme::RsM8 => match &mut self.data {
                Some(FecData::ReedSolomon(rs)) => {
                    rs.decode(dec_msg_len, msg_enc, msg_dec)?;
                }
                _ => return Err(Error::Config("Reed-Solomon state missing".into())),
            },
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
            | FecScheme::ConvV29P78 => match &mut self.data {
                Some(FecData::Convolutional(c)) => c.decode(dec_msg_len, msg_enc, msg_dec)?,
                _ => return Err(Error::Config("convolutional state missing".into())),
            },
        }

        Ok(())
    }

    /// decode a block of data using a fec scheme (soft decision)
    ///
    ///  dec_msg_len    :   decoded message length (number of bytes)
    ///  msg_enc        :   encoded soft bits, 8 bytes per hard bit
    ///  msg_dec        :   decoded message, at least dec_msg_len bytes
    pub fn decode_soft(
        &mut self,
        dec_msg_len: usize,
        msg_enc: &[u8],
        msg_dec: &mut [u8],
    ) -> Result<()> {
        let enc_len = self.enc_msg_len(dec_msg_len);
        let soft_len = enc_len * 8;

        if msg_enc.len() < soft_len {
            return Err(Error::Config(format!(
                "soft encoded buffer too small: {} < {}",
                msg_enc.len(),
                soft_len
            )));
        }

        if msg_dec.len() < dec_msg_len {
            return Err(Error::Config(format!(
                "decoded buffer too small: {} < {}",
                msg_dec.len(),
                dec_msg_len
            )));
        }

        match self.scheme {
            FecScheme::Unknown => {
                return Err(Error::Config("cannot decode with unknown scheme".into()))
            }
            FecScheme::Rep3 => rep3_decode_soft(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Rep5 => rep5_decode_soft(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Hamming74 => hamming74_decode_soft(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Hamming84 => hamming84_decode_soft(dec_msg_len, msg_enc, msg_dec),
            FecScheme::Hamming128 => hamming128_decode_soft(dec_msg_len, msg_enc, msg_dec),
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
            | FecScheme::ConvV29P78 => match &mut self.data {
                Some(FecData::Convolutional(c)) => c.decode_soft(dec_msg_len, msg_enc, msg_dec)?,
                _ => return Err(Error::Config("convolutional state missing".into())),
            },
            // soft-to-hard decode fallback
            FecScheme::None
            | FecScheme::Golay2412
            | FecScheme::Secded2216
            | FecScheme::Secded3932
            | FecScheme::Secded7264
            | FecScheme::RsM8 => {
                let mut msg_enc_hard = vec![0u8; enc_len];
                for (i, byte) in msg_enc_hard.iter_mut().enumerate() {
                    for j in 0..8 {
                        if msg_enc[8 * i + j] & 0x80 != 0 {
                            *byte |= 0x80 >> j;
                        }
                    }
                }
                return self.decode(dec_msg_len, &msg_enc_hard, msg_dec);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    #[test]
    fn test_fec_pass() {
        // pass-through doesn't correct errors, so test without corruption
        let mut fec = Fec::new(FecScheme::None).unwrap();
        let msg = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut msg_enc = [0u8; 4];
        let mut msg_dec = [0u8; 4];

        fec.encode(&msg, &mut msg_enc).unwrap();
        fec.decode(4, &msg_enc, &mut msg_dec).unwrap();

        assert_eq!(msg, msg_dec);
    }

    // the pass-through has no soft decoder either, so this is the generic
    // fallback thresholding bits back into bytes
    #[test]
    fn test_fec_pass_soft() {
        let mut fec = Fec::new(FecScheme::None).unwrap();

        // 8 soft bytes per hard bit, MSB first
        let soft = [
            // 0xA5 = 1010 0101
            0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0xFF,
            // 0x5A = 0101 1010
            0x00, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00,
        ];
        let mut msg_dec = [0u8; 2];

        fec.decode_soft(2, &soft, &mut msg_dec).unwrap();

        assert_eq!(msg_dec, [0xA5, 0x5A]);
    }

    fn fec_test_codec(scheme: FecScheme, n: usize) {
        let mut fec = Fec::new(scheme).unwrap();

        // initialize message
        let mut rng = rand::thread_rng();
        let msg_org: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

        // allocate buffers
        let enc_len = fec.enc_msg_len(n);
        let mut msg_enc = vec![0u8; enc_len];
        let mut msg_dec = vec![0u8; n];

        // encode
        fec.encode(&msg_org, &mut msg_enc).unwrap();

        // channel: add single error
        msg_enc[0] ^= 0x01;

        // decode
        fec.decode(n, &msg_enc, &mut msg_dec).unwrap();

        // validate data are the same
        assert_eq!(&msg_org[..], &msg_dec[..]);
    }

    #[test]
    #[autotest_annotate(autotest_fec_r3)]
    fn test_fec_r3() {
        fec_test_codec(FecScheme::Rep3, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_r5)]
    fn test_fec_r5() {
        fec_test_codec(FecScheme::Rep5, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_h74)]
    fn test_fec_h74() {
        fec_test_codec(FecScheme::Hamming74, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_h84)]
    fn test_fec_h84() {
        fec_test_codec(FecScheme::Hamming84, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_h128)]
    fn test_fec_h128() {
        fec_test_codec(FecScheme::Hamming128, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_g2412)]
    fn test_fec_g2412() {
        fec_test_codec(FecScheme::Golay2412, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_secded2216)]
    fn test_fec_secded2216() {
        fec_test_codec(FecScheme::Secded2216, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_secded3932)]
    fn test_fec_secded3932() {
        fec_test_codec(FecScheme::Secded3932, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_secded7264)]
    fn test_fec_secded7264() {
        fec_test_codec(FecScheme::Secded7264, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v27)]
    fn test_fec_v27() {
        fec_test_codec(FecScheme::ConvV27, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v29)]
    fn test_fec_v29() {
        fec_test_codec(FecScheme::ConvV29, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v39)]
    fn test_fec_v39() {
        fec_test_codec(FecScheme::ConvV39, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v615)]
    fn test_fec_v615() {
        fec_test_codec(FecScheme::ConvV615, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v27p23)]
    fn test_fec_v27p23() {
        fec_test_codec(FecScheme::ConvV27P23, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v27p34)]
    fn test_fec_v27p34() {
        fec_test_codec(FecScheme::ConvV27P34, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v27p45)]
    fn test_fec_v27p45() {
        fec_test_codec(FecScheme::ConvV27P45, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v27p56)]
    fn test_fec_v27p56() {
        fec_test_codec(FecScheme::ConvV27P56, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v27p67)]
    fn test_fec_v27p67() {
        fec_test_codec(FecScheme::ConvV27P67, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v27p78)]
    fn test_fec_v27p78() {
        fec_test_codec(FecScheme::ConvV27P78, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v29p23)]
    fn test_fec_v29p23() {
        fec_test_codec(FecScheme::ConvV29P23, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v29p34)]
    fn test_fec_v29p34() {
        fec_test_codec(FecScheme::ConvV29P34, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v29p45)]
    fn test_fec_v29p45() {
        fec_test_codec(FecScheme::ConvV29P45, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v29p56)]
    fn test_fec_v29p56() {
        fec_test_codec(FecScheme::ConvV29P56, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v29p67)]
    fn test_fec_v29p67() {
        fec_test_codec(FecScheme::ConvV29P67, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_v29p78)]
    fn test_fec_v29p78() {
        fec_test_codec(FecScheme::ConvV29P78, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_rs8)]
    fn test_fec_rs8() {
        fec_test_codec(FecScheme::RsM8, 64);
    }

    // test soft-decoding of a particular coding scheme
    fn fec_test_soft_codec(scheme: FecScheme, n: usize) {
        let mut fec = Fec::new(scheme).unwrap();

        // initialize message
        let mut rng = rand::thread_rng();
        let msg_org: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

        // allocate buffers
        let n_enc = fec.enc_msg_len(n);
        let mut msg_enc = vec![0u8; n_enc];
        let mut msg_soft = vec![0u8; 8 * n_enc];
        let mut msg_dec = vec![0u8; n];

        // encode
        fec.encode(&msg_org, &mut msg_enc).unwrap();

        // convert to soft bits
        for i in 0..n_enc {
            for j in 0..8 {
                msg_soft[8 * i + j] = if msg_enc[i] & (0x80 >> j) != 0 { 255 } else { 0 };
            }
        }

        // channel: add single error
        msg_soft[0] = 255 - msg_soft[0];

        // decode
        fec.decode_soft(n, &msg_soft, &mut msg_dec).unwrap();

        // validate data are the same
        assert_eq!(&msg_org[..], &msg_dec[..]);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_r3)]
    fn test_fecsoft_r3() {
        fec_test_soft_codec(FecScheme::Rep3, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_r5)]
    fn test_fecsoft_r5() {
        fec_test_soft_codec(FecScheme::Rep5, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_h74)]
    fn test_fecsoft_h74() {
        fec_test_soft_codec(FecScheme::Hamming74, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_h84)]
    fn test_fecsoft_h84() {
        fec_test_soft_codec(FecScheme::Hamming84, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_h128)]
    fn test_fecsoft_h128() {
        fec_test_soft_codec(FecScheme::Hamming128, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v27)]
    fn test_fecsoft_v27() {
        fec_test_soft_codec(FecScheme::ConvV27, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v29)]
    fn test_fecsoft_v29() {
        fec_test_soft_codec(FecScheme::ConvV29, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v39)]
    fn test_fecsoft_v39() {
        fec_test_soft_codec(FecScheme::ConvV39, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v615)]
    fn test_fecsoft_v615() {
        fec_test_soft_codec(FecScheme::ConvV615, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v27p23)]
    fn test_fecsoft_v27p23() {
        fec_test_soft_codec(FecScheme::ConvV27P23, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v27p34)]
    fn test_fecsoft_v27p34() {
        fec_test_soft_codec(FecScheme::ConvV27P34, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v27p45)]
    fn test_fecsoft_v27p45() {
        fec_test_soft_codec(FecScheme::ConvV27P45, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v27p56)]
    fn test_fecsoft_v27p56() {
        fec_test_soft_codec(FecScheme::ConvV27P56, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v27p67)]
    fn test_fecsoft_v27p67() {
        fec_test_soft_codec(FecScheme::ConvV27P67, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v27p78)]
    fn test_fecsoft_v27p78() {
        fec_test_soft_codec(FecScheme::ConvV27P78, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v29p23)]
    fn test_fecsoft_v29p23() {
        fec_test_soft_codec(FecScheme::ConvV29P23, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v29p34)]
    fn test_fecsoft_v29p34() {
        fec_test_soft_codec(FecScheme::ConvV29P34, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v29p45)]
    fn test_fecsoft_v29p45() {
        fec_test_soft_codec(FecScheme::ConvV29P45, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v29p56)]
    fn test_fecsoft_v29p56() {
        fec_test_soft_codec(FecScheme::ConvV29P56, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v29p67)]
    fn test_fecsoft_v29p67() {
        fec_test_soft_codec(FecScheme::ConvV29P67, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fecsoft_v29p78)]
    fn test_fecsoft_v29p78() {
        fec_test_soft_codec(FecScheme::ConvV29P78, 64);
    }
    // Reed-Solomon has no soft decoder, so this runs the hard-decision fallback
    #[test]
    #[autotest_annotate(autotest_fecsoft_rs8)]
    fn test_fecsoft_rs8() {
        fec_test_soft_codec(FecScheme::RsM8, 64);
    }

    #[test]
    fn test_fec_golay_enc_msg_len() {
        let fec = Fec::new(FecScheme::Golay2412).unwrap();

        for n in 1..=32usize {
            let r = n % 3;
            let expected = (n - r) / 3 * 6 + r * 3;
            assert_eq!(
                fec.enc_msg_len(n),
                expected,
                "enc_msg_len disagrees with encoder layout for length {}",
                n
            );
        }
    }

    #[test]
    fn test_fec_rs8_byte_errors() {
        let n = 64;
        let mut fec = Fec::new(FecScheme::RsM8).unwrap();

        let mut rng = rand::thread_rng();
        let msg_org: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

        let enc_len = fec.enc_msg_len(n);
        let mut msg_enc = vec![0u8; enc_len];
        let mut msg_dec = vec![0u8; n];

        fec.encode(&msg_org, &mut msg_enc).unwrap();

        // corrupt 16 bytes, the correction limit
        for k in 0..16 {
            msg_enc[k * 5] ^= 0xff;
        }

        fec.decode(n, &msg_enc, &mut msg_dec).unwrap();

        assert_eq!(&msg_org[..], &msg_dec[..]);
    }

    #[test]
    #[autotest_annotate(autotest_reedsolomon_223_255)]
    fn test_reedsolomon_223_255() {
        let dec_msg_len = 223;

        let mut fec = Fec::new(FecScheme::RsM8).unwrap();

        // compute and test encoded message length
        let enc_msg_len = fec.enc_msg_len(dec_msg_len);
        assert_eq!(enc_msg_len, 255);

        let msg_org: Vec<u8> = (0..dec_msg_len).map(|i| (i & 0xff) as u8).collect();

        let mut msg_enc = vec![0u8; enc_msg_len];
        let mut msg_dec = vec![0u8; dec_msg_len];

        fec.encode(&msg_org, &mut msg_enc).unwrap();

        // corrupt encoded message; can withstand up to 16 symbol errors
        let mut msg_rec = msg_enc.clone();
        for i in 0..16 {
            msg_rec[i] ^= 0x75;
        }

        fec.decode(dec_msg_len, &msg_rec, &mut msg_dec).unwrap();

        assert_eq!(msg_org, msg_dec);
    }

    #[test]
    fn test_fec_soft_falls_back_to_hard() {
        for scheme in [
            FecScheme::Golay2412,
            FecScheme::Secded2216,
            FecScheme::Secded3932,
            FecScheme::Secded7264,
            FecScheme::RsM8,
        ] {
            let n = 64;
            let mut fec = Fec::new(scheme).unwrap();

            let mut rng = rand::thread_rng();
            let msg_org: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

            let n_enc = fec.enc_msg_len(n);
            let mut msg_enc = vec![0u8; n_enc];
            fec.encode(&msg_org, &mut msg_enc).unwrap();

            // single error, then expand to soft rails
            msg_enc[0] ^= 0x01;
            let mut msg_soft = vec![0u8; 8 * n_enc];
            for i in 0..n_enc {
                for j in 0..8 {
                    msg_soft[8 * i + j] = if msg_enc[i] & (0x80 >> j) != 0 { 255 } else { 0 };
                }
            }

            let mut dec_soft = vec![0u8; n];
            let mut dec_hard = vec![0u8; n];
            fec.decode_soft(n, &msg_soft, &mut dec_soft).unwrap();
            fec.decode(n, &msg_enc, &mut dec_hard).unwrap();

            assert_eq!(
                dec_soft, dec_hard,
                "{:?}: soft fallback disagreed with hard decoding",
                scheme
            );
            assert_eq!(dec_soft, msg_org, "{:?}: soft fallback lost data", scheme);
        }
    }

    fn fec_test_copy(scheme: FecScheme, n: usize) {
        let mut q0 = Fec::new(scheme).unwrap();

        let n_enc = q0.enc_msg_len(n);
        let mut rng = rand::thread_rng();
        let msg_org: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

        let mut msg_enc_0 = vec![0u8; n_enc];
        let mut msg_enc_1 = vec![0u8; n_enc];

        // encode with the original
        q0.encode(&msg_org, &mut msg_enc_0).unwrap();

        // clone the object and repeat
        let mut q1 = q0.clone();
        q1.encode(&msg_org, &mut msg_enc_1).unwrap();

        assert_eq!(msg_enc_0, msg_enc_1, "{:?}: clone encoded differently", scheme);

        // now decode random bits through both and compare
        for i in 0..n_enc {
            msg_enc_0[i] = rng.gen::<u8>();
        }
        msg_enc_1.copy_from_slice(&msg_enc_0);

        let mut msg_dec_0 = vec![0u8; n];
        let mut msg_dec_1 = vec![0u8; n];

        let r0 = q0.decode(n, &msg_enc_0, &mut msg_dec_0);
        let r1 = q1.decode(n, &msg_enc_1, &mut msg_dec_1);

        assert_eq!(
            r0.is_ok(),
            r1.is_ok(),
            "{:?}: clone disagreed on decode success",
            scheme
        );

        // on the failure path Reed-Solomon returns before writing `msg_dec`,
        // so comparing the outputs there would just compare two untouched
        // zero buffers. only compare them when decoding actually ran.
        if r0.is_ok() {
            assert_eq!(msg_dec_0, msg_dec_1, "{:?}: clone decoded differently", scheme);
        }

        // repeat with correctable corruption so every scheme has its
        // clone's decode path exercised on real output.
        q0.encode(&msg_org, &mut msg_enc_0).unwrap();
        msg_enc_0[0] ^= 0x01;
        msg_enc_1.copy_from_slice(&msg_enc_0);

        q0.decode(n, &msg_enc_0, &mut msg_dec_0).unwrap();
        q1.decode(n, &msg_enc_1, &mut msg_dec_1).unwrap();

        assert_eq!(
            msg_dec_0, msg_dec_1,
            "{:?}: clone decoded corrupted message differently",
            scheme
        );
        assert_eq!(
            msg_dec_0, msg_org,
            "{:?}: failed to correct a single bit error",
            scheme
        );
    }

    // repeat codes
    #[test]
    #[autotest_annotate(autotest_fec_copy_r3)]
    fn test_fec_copy_r3() {
        fec_test_copy(FecScheme::Rep3, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_r5)]
    fn test_fec_copy_r5() {
        fec_test_copy(FecScheme::Rep5, 64);
    }

    // Hamming block codes
    #[test]
    #[autotest_annotate(autotest_fec_copy_h74)]
    fn test_fec_copy_h74() {
        fec_test_copy(FecScheme::Hamming74, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_h84)]
    fn test_fec_copy_h84() {
        fec_test_copy(FecScheme::Hamming84, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_h128)]
    fn test_fec_copy_h128() {
        fec_test_copy(FecScheme::Hamming128, 64);
    }

    // Golay block code
    #[test]
    #[autotest_annotate(autotest_fec_copy_g2412)]
    fn test_fec_copy_g2412() {
        fec_test_copy(FecScheme::Golay2412, 64);
    }

    // SEC-DED block codes
    #[test]
    #[autotest_annotate(autotest_fec_copy_secded2216)]
    fn test_fec_copy_secded2216() {
        fec_test_copy(FecScheme::Secded2216, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_secded3932)]
    fn test_fec_copy_secded3932() {
        fec_test_copy(FecScheme::Secded3932, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_secded7264)]
    fn test_fec_copy_secded7264() {
        fec_test_copy(FecScheme::Secded7264, 64);
    }

    // convolutional codes
    #[test]
    #[autotest_annotate(autotest_fec_copy_v27)]
    fn test_fec_copy_v27() {
        fec_test_copy(FecScheme::ConvV27, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v29)]
    fn test_fec_copy_v29() {
        fec_test_copy(FecScheme::ConvV29, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v39)]
    fn test_fec_copy_v39() {
        fec_test_copy(FecScheme::ConvV39, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v615)]
    fn test_fec_copy_v615() {
        fec_test_copy(FecScheme::ConvV615, 64);
    }

    // convolutional codes (punctured)
    #[test]
    #[autotest_annotate(autotest_fec_copy_v27p23)]
    fn test_fec_copy_v27p23() {
        fec_test_copy(FecScheme::ConvV27P23, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v27p34)]
    fn test_fec_copy_v27p34() {
        fec_test_copy(FecScheme::ConvV27P34, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v27p45)]
    fn test_fec_copy_v27p45() {
        fec_test_copy(FecScheme::ConvV27P45, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v27p56)]
    fn test_fec_copy_v27p56() {
        fec_test_copy(FecScheme::ConvV27P56, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v27p67)]
    fn test_fec_copy_v27p67() {
        fec_test_copy(FecScheme::ConvV27P67, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v27p78)]
    fn test_fec_copy_v27p78() {
        fec_test_copy(FecScheme::ConvV27P78, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v29p23)]
    fn test_fec_copy_v29p23() {
        fec_test_copy(FecScheme::ConvV29P23, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v29p34)]
    fn test_fec_copy_v29p34() {
        fec_test_copy(FecScheme::ConvV29P34, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v29p45)]
    fn test_fec_copy_v29p45() {
        fec_test_copy(FecScheme::ConvV29P45, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v29p56)]
    fn test_fec_copy_v29p56() {
        fec_test_copy(FecScheme::ConvV29P56, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v29p67)]
    fn test_fec_copy_v29p67() {
        fec_test_copy(FecScheme::ConvV29P67, 64);
    }

    #[test]
    #[autotest_annotate(autotest_fec_copy_v29p78)]
    fn test_fec_copy_v29p78() {
        fec_test_copy(FecScheme::ConvV29P78, 64);
    }

    // Reed-Solomon block code
    #[test]
    #[autotest_annotate(autotest_fec_copy_rs8)]
    fn test_fec_copy_rs8() {
        fec_test_copy(FecScheme::RsM8, 64);
    }

    // enc_msg_len must match what each encoder actually writes.
    #[test]
    fn test_fec_secded_enc_msg_len() {
        for (scheme, data_bytes) in [
            (FecScheme::Secded2216, 2usize),
            (FecScheme::Secded3932, 4),
            (FecScheme::Secded7264, 8),
        ] {
            let fec = Fec::new(scheme).unwrap();

            for n in 1..=40usize {
                let r = n % data_bytes;
                let expected =
                    (n - r) / data_bytes * (data_bytes + 1) + if r != 0 { r + 1 } else { 0 };
                assert_eq!(
                    fec.enc_msg_len(n),
                    expected,
                    "{:?} enc_msg_len disagrees with encoder layout for length {}",
                    scheme,
                    n
                );
            }
        }
    }

    #[test]
    #[autotest_annotate(autotest_rep3_codec)]
    fn test_rep3_codec() {
        let n = 4;
        let msg: [u8; 4] = [0x25, 0x62, 0x3F, 0x52];

        let mut fec = Fec::new(FecScheme::Rep3).unwrap();
        let n_enc = fec.enc_msg_len(n);
        let mut msg_enc = vec![0u8; n_enc];
        let mut msg_dec = [0u8; 4];

        // encode
        fec.encode(&msg, &mut msg_enc).unwrap();

        // Corrupt encoded message (one full copy)
        msg_enc[0] = !msg_enc[0];
        msg_enc[1] = !msg_enc[1];
        msg_enc[2] = !msg_enc[2];
        msg_enc[3] = !msg_enc[3];

        // decode
        fec.decode(n, &msg_enc, &mut msg_dec).unwrap();

        // Validate
        assert_eq!(msg, msg_dec);
    }

    #[test]
    #[autotest_annotate(autotest_rep5_codec)]
    fn test_rep5_codec() {
        let n = 4;
        let msg: [u8; 4] = [0x25, 0x62, 0x3F, 0x52];

        let mut fec = Fec::new(FecScheme::Rep5).unwrap();
        let n_enc = fec.enc_msg_len(n);
        let mut msg_enc = vec![0u8; n_enc];
        let mut msg_dec = [0u8; 4];

        // encode
        fec.encode(&msg, &mut msg_enc).unwrap();

        // Corrupt encoded message (2 of 5 copies for each byte)
        msg_enc[0] = !msg_enc[0];
        msg_enc[4] = !msg_enc[4];

        msg_enc[1] = !msg_enc[1];
        msg_enc[9] = !msg_enc[9];

        msg_enc[10] = !msg_enc[10];
        msg_enc[14] = !msg_enc[14];

        msg_enc[3] = !msg_enc[3];
        msg_enc[19] = !msg_enc[19];

        // decode
        fec.decode(n, &msg_enc, &mut msg_dec).unwrap();

        // Validate
        assert_eq!(msg, msg_dec);
    }

    #[test]
    #[autotest_annotate(autotest_fec_config)]
    fn test_fec_config() {
        use crate::fec::codec::{
            golay2412_decode_symbol, golay2412_encode_symbol, hamming128_decode_symbol,
            hamming1511_decode_symbol, hamming1511_encode_symbol, hamming3126_decode_symbol,
            hamming3126_encode_symbol,
        };

        fn panics(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
            // the panic message is expected, so keep it out of the test log
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(f);
            std::panic::set_hook(prev);
            result.is_err()
        }

        assert!(panics(|| { golay2412_encode_symbol(1 << 12); }));
        assert!(panics(|| { golay2412_decode_symbol(1 << 24); }));

        assert!(panics(|| { hamming3126_encode_symbol(1 << 26); }));
        assert!(panics(|| { hamming3126_decode_symbol(1u32 << 31); }));

        assert!(panics(|| { hamming1511_encode_symbol(1 << 11); }));
        assert!(panics(|| { hamming1511_decode_symbol(1 << 15); }));

        // liquid also checks hamming128_encode_symbol(1<<8), but ours takes a
        // u8 so the type already bounds it
        assert!(panics(|| { hamming128_decode_symbol(1 << 12); }));
    }
}
