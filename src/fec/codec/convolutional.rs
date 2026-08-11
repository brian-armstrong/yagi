//
// convolutional code
//

use fec::convolutional::{Decoder, Encoder, Puncturer};
#[cfg(feature = "simd")]
use fec::convolutional::SimdDecoder;

use crate::fec::FecScheme;
use crate::error::{Error, Result};

use super::conv_params::{self, ConvParams};

// The `fec` crate flushes order+1 tail bits where the convention (Proakis;
// Lin & Costello; CCSDS; libfec, and so liquid) is K-1, two input bits fewer.
// The output is a parity function of the shift register, which is already
// all-zero after K-1 zeros, so those extra bits are always zero: we trim them
// after encoding and restore them before decoding to match liquid's lengths.
//
// TODO: remove this once the `fec` crate flushes K-1
const TAIL_EXCESS_INPUT_BITS: usize = 2;

// extra tail output bits, for a given rate
const fn tail_excess_bits(rate: u32) -> usize {
    TAIL_EXCESS_INPUT_BITS * rate as usize
}

// bits are packed MSB-first within each byte
#[inline]
fn get_bit(buf: &[u8], i: usize) -> u8 {
    (buf[i / 8] >> (7 - (i % 8))) & 1
}

fn clear_bits(buf: &mut [u8], from: usize, to: usize) {
    for i in from..to {
        buf[i / 8] &= !(1 << (7 - (i % 8)));
    }
}

fn set_bits(buf: &mut [u8], from: usize, to: usize) {
    for i in from..to {
        buf[i / 8] |= 1 << (7 - (i % 8));
    }
}

// SimdDecoder takes rate and order as const generics, so it needs one variant
// per shape. the scalar Decoder takes them at runtime and needs only one
#[derive(Clone)]
enum ConvDecoder {
    Scalar(Decoder),
    #[cfg(feature = "simd")]
    Simd27(Box<SimdDecoder<2, 7>>),
    #[cfg(feature = "simd")]
    Simd29(Box<SimdDecoder<2, 9>>),
    #[cfg(feature = "simd")]
    Simd39(Box<SimdDecoder<3, 9>>),
    #[cfg(feature = "simd")]
    Simd615(Box<SimdDecoder<6, 15>>),
}

impl ConvDecoder {
    #[cfg(test)]
    fn kind(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar",
            #[cfg(feature = "simd")]
            Self::Simd27(_) => "simd<2,7>",
            #[cfg(feature = "simd")]
            Self::Simd29(_) => "simd<2,9>",
            #[cfg(feature = "simd")]
            Self::Simd39(_) => "simd<3,9>",
            #[cfg(feature = "simd")]
            Self::Simd615(_) => "simd<6,15>",
        }
    }

    fn new(params: &ConvParams) -> Self {
        #[cfg(feature = "simd")]
        {
            match (params.rate, params.order) {
                (2, 7) => return Self::Simd27(Box::new(SimdDecoder::new(params.polys))),
                (2, 9) => return Self::Simd29(Box::new(SimdDecoder::new(params.polys))),
                (3, 9) => return Self::Simd39(Box::new(SimdDecoder::new(params.polys))),
                (6, 15) => return Self::Simd615(Box::new(SimdDecoder::new(params.polys))),
                // no SIMD instantiation for this shape; fall through to scalar
                _ => {}
            }
        }

        Self::Scalar(Decoder::new(params.rate, params.order, params.polys))
    }

    fn decode_hard(
        &mut self,
        encoded: &[u8],
        num_encoded_bits: usize,
        msg: &mut [u8],
    ) -> std::result::Result<usize, fec::convolutional::DecodeError> {
        match self {
            Self::Scalar(d) => d.decode_hard(encoded, num_encoded_bits, msg),
            #[cfg(feature = "simd")]
            Self::Simd27(d) => d.decode_hard(encoded, num_encoded_bits, msg),
            #[cfg(feature = "simd")]
            Self::Simd29(d) => d.decode_hard(encoded, num_encoded_bits, msg),
            #[cfg(feature = "simd")]
            Self::Simd39(d) => d.decode_hard(encoded, num_encoded_bits, msg),
            #[cfg(feature = "simd")]
            Self::Simd615(d) => d.decode_hard(encoded, num_encoded_bits, msg),
        }
    }

    fn decode_soft(
        &mut self,
        encoded: &[u8],
        msg: &mut [u8],
    ) -> std::result::Result<usize, fec::convolutional::DecodeError> {
        match self {
            Self::Scalar(d) => d.decode_soft(encoded, msg),
            #[cfg(feature = "simd")]
            Self::Simd27(d) => d.decode_soft(encoded, msg),
            #[cfg(feature = "simd")]
            Self::Simd29(d) => d.decode_soft(encoded, msg),
            #[cfg(feature = "simd")]
            Self::Simd39(d) => d.decode_soft(encoded, msg),
            #[cfg(feature = "simd")]
            Self::Simd615(d) => d.decode_soft(encoded, msg),
        }
    }

    fn decode_hard_with_erasure(
        &mut self,
        encoded: &[u8],
        num_encoded_bits: usize,
        erasure: &[u8],
        msg: &mut [u8],
    ) -> std::result::Result<usize, fec::convolutional::DecodeError> {
        match self {
            Self::Scalar(d) => d.decode_hard_with_erasure(encoded, num_encoded_bits, erasure, msg),
            #[cfg(feature = "simd")]
            Self::Simd27(d) => d.decode_hard_with_erasure(encoded, num_encoded_bits, erasure, msg),
            #[cfg(feature = "simd")]
            Self::Simd29(d) => d.decode_hard_with_erasure(encoded, num_encoded_bits, erasure, msg),
            #[cfg(feature = "simd")]
            Self::Simd39(d) => d.decode_hard_with_erasure(encoded, num_encoded_bits, erasure, msg),
            #[cfg(feature = "simd")]
            Self::Simd615(d) => d.decode_hard_with_erasure(encoded, num_encoded_bits, erasure, msg),
        }
    }

    fn decode_soft_with_erasure(
        &mut self,
        encoded: &[u8],
        erasure: &[u8],
        msg: &mut [u8],
    ) -> std::result::Result<usize, fec::convolutional::DecodeError> {
        match self {
            Self::Scalar(d) => d.decode_soft_with_erasure(encoded, erasure, msg),
            #[cfg(feature = "simd")]
            Self::Simd27(d) => d.decode_soft_with_erasure(encoded, erasure, msg),
            #[cfg(feature = "simd")]
            Self::Simd29(d) => d.decode_soft_with_erasure(encoded, erasure, msg),
            #[cfg(feature = "simd")]
            Self::Simd39(d) => d.decode_soft_with_erasure(encoded, erasure, msg),
            #[cfg(feature = "simd")]
            Self::Simd615(d) => d.decode_soft_with_erasure(encoded, erasure, msg),
        }
    }
}

/// parameters and puncturing matrix for a scheme (None if not convolutional)
pub fn conv_scheme_params(
    scheme: crate::fec::FecScheme,
) -> Option<(ConvParams, Option<Vec<Vec<bool>>>)> {

    let to_matrix = |m: &[&[bool]]| -> Option<Vec<Vec<bool>>> {
        Some(m.iter().map(|r| r.to_vec()).collect())
    };

    Some(match scheme {
        FecScheme::ConvV27 => (conv_params::CONV_V27, None),
        FecScheme::ConvV29 => (conv_params::CONV_V29, None),
        FecScheme::ConvV39 => (conv_params::CONV_V39, None),
        FecScheme::ConvV615 => (conv_params::CONV_V615, None),
        FecScheme::ConvV27P23 => (conv_params::CONV_V27, to_matrix(&conv_params::PMATRIX_V27P23)),
        FecScheme::ConvV27P34 => (conv_params::CONV_V27, to_matrix(&conv_params::PMATRIX_V27P34)),
        FecScheme::ConvV27P45 => (conv_params::CONV_V27, to_matrix(&conv_params::PMATRIX_V27P45)),
        FecScheme::ConvV27P56 => (conv_params::CONV_V27, to_matrix(&conv_params::PMATRIX_V27P56)),
        FecScheme::ConvV27P67 => (conv_params::CONV_V27, to_matrix(&conv_params::PMATRIX_V27P67)),
        FecScheme::ConvV27P78 => (conv_params::CONV_V27, to_matrix(&conv_params::PMATRIX_V27P78)),
        FecScheme::ConvV29P23 => (conv_params::CONV_V29, to_matrix(&conv_params::PMATRIX_V29P23)),
        FecScheme::ConvV29P34 => (conv_params::CONV_V29, to_matrix(&conv_params::PMATRIX_V29P34)),
        FecScheme::ConvV29P45 => (conv_params::CONV_V29, to_matrix(&conv_params::PMATRIX_V29P45)),
        FecScheme::ConvV29P56 => (conv_params::CONV_V29, to_matrix(&conv_params::PMATRIX_V29P56)),
        FecScheme::ConvV29P67 => (conv_params::CONV_V29, to_matrix(&conv_params::PMATRIX_V29P67)),
        FecScheme::ConvV29P78 => (conv_params::CONV_V29, to_matrix(&conv_params::PMATRIX_V29P78)),
        _ => return None,
    })
}

/// convolutional codec, optionally punctured
#[derive(Clone)]
pub struct Convolutional {
    params: ConvParams,
    encoder: Encoder,
    decoder: ConvDecoder,
    /// None for the base codes
    puncturer: Option<Puncturer>,
    /// scratch for the unpunctured stream
    full: Vec<u8>,
    /// scratch for the erasure mask
    erasure: Vec<u8>,
}

impl std::fmt::Debug for Convolutional {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Convolutional")
            .field("rate", &self.params.rate)
            .field("order", &self.params.order)
            .field("punctured", &self.puncturer.is_some())
            .finish_non_exhaustive()
    }
}


impl Convolutional {
    /// create codec for params, optionally punctured by matrix
    ///
    ///  matrix     :   rate rows of period flags; true means transmitted
    pub fn new(params: ConvParams, matrix: Option<Vec<Vec<bool>>>) -> Self {
        let puncturer = matrix.as_ref().map(|m| {
            let rows: Vec<&[bool]> = m.iter().map(|r| r.as_slice()).collect();
            Puncturer::from_matrix(&rows).expect("puncturing matrix is well-formed")
        });

        Self {
            params,
            encoder: Encoder::new(params.rate, params.order, params.polys),
            decoder: ConvDecoder::new(&params),
            puncturer,
            full: Vec::new(),
            erasure: Vec::new(),
        }
    }

    fn scratch(buf: &mut Vec<u8>, n: usize) {
        if buf.len() < n {
            buf.resize(n, 0);
        }
        // as an extra precaution, clear the final byte. if the final byte gets only
        //   a partial write, then we'll have initialized it.
        if n > 0 {
            buf[n - 1] = 0;
        }
    }

    pub fn min_dec_msg_len(&self) -> usize {
        (2 * self.params.order as usize - 1).div_ceil(8)
    }

    fn check_len(&self, dec_msg_len: usize) -> Result<()> {
        let min = self.min_dec_msg_len();
        if dec_msg_len < min {
            return Err(Error::Config(format!(
                "convolutional r=1/{} K={} needs at least {} bytes, got {}",
                self.params.rate, self.params.order, min, dec_msg_len
            )));
        }
        Ok(())
    }

    /// encode block of data using convolutional encoder
    ///
    ///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
    ///  msg_enc        :   encoded message
    pub fn encode(&mut self, msg_dec: &[u8], msg_enc: &mut [u8]) -> Result<()> {
        self.check_len(msg_dec.len())?;

        let full_bits = self.encoder.encode_len(msg_dec.len());
        let excess = tail_excess_bits(self.params.rate);
        let kept_bits = full_bits - excess;

        // encode into scratch, since we may need to puncture afterwards
        Self::scratch(&mut self.full, full_bits.div_ceil(8));

        let written = self
            .encoder
            .encode(msg_dec, &mut self.full)
            .map_err(|e| Error::Config(format!("convolutional encode failed: {:?}", e)))?;
        debug_assert_eq!(written, full_bits);

        // emitted from an all-zero shift register, so dropping them is lossless
        debug_assert!(
            (kept_bits..full_bits).all(|i| get_bit(&self.full, i) == 0),
            "tail flush bits were not zero"
        );

        match &self.puncturer {
            Some(p) => {
                p.puncture(&self.full, kept_bits, msg_enc)
                    .map_err(|e| Error::Config(format!("puncturing failed: {:?}", e)))?;
            }
            None => {
                let nbytes = kept_bits.div_ceil(8);
                if msg_enc.len() < nbytes {
                    return Err(Error::Config(format!(
                        "encoded buffer too small: {} < {}",
                        msg_enc.len(),
                        nbytes
                    )));
                }
                msg_enc[..nbytes].copy_from_slice(&self.full[..nbytes]);
                // zero any bits past the trim point in the final byte
                clear_bits(msg_enc, kept_bits, nbytes * 8);
            }
        }

        Ok(())
    }

    /// decode block of data using convolutional decoder
    ///
    ///  dec_msg_len    :   decoded message length (number of bytes)
    ///  msg_enc        :   encoded message
    ///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
    pub fn decode(&mut self, dec_msg_len: usize, msg_enc: &[u8], msg_dec: &mut [u8]) -> Result<()> {
        self.check_len(dec_msg_len)?;

        let full_bits = self.encoder.encode_len(dec_msg_len);
        let excess = tail_excess_bits(self.params.rate);
        let kept_bits = full_bits - excess;

        // sized for the full encoding, so the trimmed tail restores as zeros
        Self::scratch(&mut self.full, full_bits.div_ceil(8));

        match &self.puncturer {
            Some(p) => {
                // depuncturing reinserts the punctured positions and reports
                // them as erasures
                Self::scratch(&mut self.erasure, full_bits.div_ceil(8));
                p.depuncture_hard(msg_enc, kept_bits, &mut self.full, &mut self.erasure)
                    .map_err(|e| Error::Config(format!("depuncturing failed: {:?}", e)))?;

                // depuncturing only fills the first kept_bits; the tail beyond
                // that was never transmitted, so mark it erased rather than
                // leaving a confidently-received zero
                clear_bits(&mut self.full, kept_bits, full_bits);
                set_bits(&mut self.erasure, kept_bits, full_bits);

                self.decoder
                    .decode_hard_with_erasure(&self.full, full_bits, &self.erasure, msg_dec)
            }
            None => {
                let nbytes = kept_bits.div_ceil(8);
                self.full[..nbytes].copy_from_slice(&msg_enc[..nbytes]);
                clear_bits(&mut self.full, kept_bits, full_bits);
                self.decoder.decode_hard(&self.full, full_bits, msg_dec)
            }
        }
        .map_err(|e| Error::Config(format!("convolutional decode failed: {:?}", e)))?;

        Ok(())
    }

    /// decode block of data using convolutional soft decoder
    ///
    ///  dec_msg_len    :   decoded message length (number of bytes)
    ///  msg_enc        :   encoded message [size: 8*enc_msg_len x 1]
    ///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
    pub fn decode_soft(
        &mut self,
        dec_msg_len: usize,
        msg_enc: &[u8],
        msg_dec: &mut [u8],
    ) -> Result<()> {
        self.check_len(dec_msg_len)?;

        let full_bits = self.encoder.encode_len(dec_msg_len);
        let excess = tail_excess_bits(self.params.rate);
        let kept_bits = full_bits - excess;

        // one byte per encoded bit. the restored tail is a hard zero, so it
        // gets confidence 0
        Self::scratch(&mut self.full, full_bits);

        match &self.puncturer {
            Some(p) => {
                Self::scratch(&mut self.erasure, full_bits.div_ceil(8));

                // depuncture_soft infers its input count from dst.len(),
                // unlike depuncture_hard which takes the bit count. only
                // kept_bits were transmitted, so pass a kept_bits-long view
                p.depuncture_soft(msg_enc, &mut self.full[..kept_bits], &mut self.erasure)
                    .map_err(|e| Error::Config(format!("depuncturing failed: {:?}", e)))?;

                // as in decode, the tail was never transmitted, so mark it
                // erased rather than leaving a sample read as a hard 0
                for s in self.full[kept_bits..full_bits].iter_mut() {
                    *s = 0;
                }
                set_bits(&mut self.erasure, kept_bits, full_bits);

                self.decoder
                    .decode_soft_with_erasure(&self.full[..full_bits], &self.erasure, msg_dec)
            }
            None => {
                self.full[..kept_bits].copy_from_slice(&msg_enc[..kept_bits]);
                self.decoder.decode_soft(&self.full[..full_bits], msg_dec)
            }
        }
        .map_err(|e| Error::Config(format!("convolutional decode failed: {:?}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::codec::conv_params::{CONV_V27, CONV_V29, CONV_V39, CONV_V615};
    use rand::Rng;

    use crate::fec::FecScheme;

    // carries the scheme so tests size buffers through FecScheme::enc_msg_len
    // exactly as callers do
    fn base_codes() -> [(&'static str, ConvParams, FecScheme); 4] {
        [
            ("v27", CONV_V27, FecScheme::ConvV27),
            ("v29", CONV_V29, FecScheme::ConvV29),
            ("v39", CONV_V39, FecScheme::ConvV39),
            ("v615", CONV_V615, FecScheme::ConvV615),
        ]
    }

    #[test]
    fn test_conv_roundtrip_clean() {
        let mut rng = rand::thread_rng();

        for (name, params, scheme) in base_codes() {
            let mut c = Convolutional::new(params, None);
            let min = c.min_dec_msg_len();

            for n in [min, min + 1, 8, 16, 64] {
                if n < min {
                    continue;
                }
                let msg: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

                let mut encoded = vec![0u8; scheme.enc_msg_len(n)];
                let mut decoded = vec![0u8; n];

                c.encode(&msg, &mut encoded).unwrap();
                c.decode(n, &encoded, &mut decoded).unwrap();

                assert_eq!(msg, decoded, "{}: clean round trip failed at n={}", name, n);
            }
        }
    }

    #[test]
    fn test_conv_corrects_bit_errors() {
        let mut rng = rand::thread_rng();

        for (name, params, scheme) in base_codes() {
            let mut c = Convolutional::new(params, None);
            let n = 64;

            let msg: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();
            let mut encoded = vec![0u8; scheme.enc_msg_len(n)];
            let mut decoded = vec![0u8; n];

            c.encode(&msg, &mut encoded).unwrap();

            // scatter a few bit errors, spaced well apart
            for k in 0..4 {
                let bit = 40 + k * 97;
                encoded[bit / 8] ^= 1 << (7 - (bit % 8));
            }

            c.decode(n, &encoded, &mut decoded).unwrap();

            assert_eq!(msg, decoded, "{}: failed to correct scattered bit errors", name);
        }
    }

    #[test]
    fn test_conv_soft_roundtrip() {
        let mut rng = rand::thread_rng();

        for (name, params, scheme) in base_codes() {
            let mut c = Convolutional::new(params, None);
            let n = 32;

            let msg: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();
            let mut encoded = vec![0u8; scheme.enc_msg_len(n)];
            c.encode(&msg, &mut encoded).unwrap();

            // expand to soft bits: 255 for a 1, 0 for a 0. The unpunctured
            // encoded length in bits is rate*(n*8 + K - 1), matching the
            // tail-trimmed stream the encoder wrote.
            let nbits = params.rate as usize * (n * 8 + params.order as usize - 1);
            let soft: Vec<u8> = (0..nbits)
                .map(|i| if get_bit(&encoded, i) != 0 { 255 } else { 0 })
                .collect();

            let mut decoded = vec![0u8; n];
            c.decode_soft(n, &soft, &mut decoded).unwrap();

            assert_eq!(msg, decoded, "{}: soft round trip failed", name);
        }
    }

    #[test]
    fn test_conv_rejects_short_payload() {
        for (name, params, _scheme) in base_codes() {
            let mut c = Convolutional::new(params, None);
            let min = c.min_dec_msg_len();

            let msg = vec![0u8; min - 1];
            let mut encoded = vec![0u8; 64];

            assert!(
                c.encode(&msg, &mut encoded).is_err(),
                "{}: should reject payload below minimum",
                name
            );
        }
    }

    #[test]
    fn test_conv_decoder_selection() {
        let expected: [&str; 4] = if cfg!(feature = "simd") {
            ["simd<2,7>", "simd<2,9>", "simd<3,9>", "simd<6,15>"]
        } else {
            ["scalar"; 4]
        };

        for ((name, params, _), want) in base_codes().into_iter().zip(expected) {
            let c = Convolutional::new(params, None);
            assert_eq!(
                c.decoder.kind(),
                want,
                "{}: unexpected decoder implementation",
                name
            );
        }
    }

    #[test]
    fn test_conv_punctured_clean_roundtrip() {
        use crate::fec::codec::conv_params::*;

        let cases: [(&str, ConvParams, &[&[bool]], FecScheme); 12] = [
            ("v27p23", CONV_V27, &PMATRIX_V27P23, FecScheme::ConvV27P23),
            ("v27p34", CONV_V27, &PMATRIX_V27P34, FecScheme::ConvV27P34),
            ("v27p45", CONV_V27, &PMATRIX_V27P45, FecScheme::ConvV27P45),
            ("v27p56", CONV_V27, &PMATRIX_V27P56, FecScheme::ConvV27P56),
            ("v27p67", CONV_V27, &PMATRIX_V27P67, FecScheme::ConvV27P67),
            ("v27p78", CONV_V27, &PMATRIX_V27P78, FecScheme::ConvV27P78),
            ("v29p23", CONV_V29, &PMATRIX_V29P23, FecScheme::ConvV29P23),
            ("v29p34", CONV_V29, &PMATRIX_V29P34, FecScheme::ConvV29P34),
            ("v29p45", CONV_V29, &PMATRIX_V29P45, FecScheme::ConvV29P45),
            ("v29p56", CONV_V29, &PMATRIX_V29P56, FecScheme::ConvV29P56),
            ("v29p67", CONV_V29, &PMATRIX_V29P67, FecScheme::ConvV29P67),
            ("v29p78", CONV_V29, &PMATRIX_V29P78, FecScheme::ConvV29P78),
        ];

        let mut rng = rand::thread_rng();
        let mut failures = Vec::new();

        for (name, params, matrix, scheme) in cases {
            let m: Vec<Vec<bool>> = matrix.iter().map(|r| r.to_vec()).collect();
            let mut c = Convolutional::new(params, Some(m));

            let n = 64;
            let msg: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();
            let mut encoded = vec![0u8; scheme.enc_msg_len(n)];
            let mut decoded = vec![0u8; n];

            c.encode(&msg, &mut encoded).unwrap();
            c.decode(n, &encoded, &mut decoded).unwrap();

            if msg != decoded {
                let bad = msg.iter().zip(&decoded).filter(|(a, b)| a != b).count();
                failures.push(format!("{} ({} bytes differ)", name, bad));
            }
        }

        assert!(
            failures.is_empty(),
            "clean round trip failed for: {}",
            failures.join(", ")
        );
    }

    #[test]
    fn test_conv_punctured_enc_len_matches_liquid() {
        use crate::fec::codec::conv_params::*;
        use crate::fec::scheme::conv_punctured_enc_msg_len;
        use fec::convolutional::Puncturer;

        let cases: [(&str, ConvParams, &[&[bool]], usize, usize); 12] = [
            ("v27p23", CONV_V27, &PMATRIX_V27P23, 7, 2),
            ("v27p34", CONV_V27, &PMATRIX_V27P34, 7, 3),
            ("v27p45", CONV_V27, &PMATRIX_V27P45, 7, 4),
            ("v27p56", CONV_V27, &PMATRIX_V27P56, 7, 5),
            ("v27p67", CONV_V27, &PMATRIX_V27P67, 7, 6),
            ("v27p78", CONV_V27, &PMATRIX_V27P78, 7, 7),
            ("v29p23", CONV_V29, &PMATRIX_V29P23, 9, 2),
            ("v29p34", CONV_V29, &PMATRIX_V29P34, 9, 3),
            ("v29p45", CONV_V29, &PMATRIX_V29P45, 9, 4),
            ("v29p56", CONV_V29, &PMATRIX_V29P56, 9, 5),
            ("v29p67", CONV_V29, &PMATRIX_V29P67, 9, 6),
            ("v29p78", CONV_V29, &PMATRIX_V29P78, 9, 7),
        ];

        for (name, params, matrix, k, p) in cases {
            let rows: Vec<&[bool]> = matrix.to_vec();
            let puncturer = Puncturer::from_matrix(&rows).unwrap();

            for n in 4..=80usize {
                let trimmed_bits =
                    params.rate as usize * (n * 8 + params.order as usize - 1);
                let ours = puncturer.punctured_len(trimmed_bits).div_ceil(8);

                assert_eq!(
                    ours,
                    conv_punctured_enc_msg_len(n, k, p),
                    "{}: punctured length disagrees with liquid at n={}",
                    name,
                    n
                );
            }
        }
    }

    #[test]
    fn test_conv_enc_len_matches_liquid() {
        // liquid's fec_get_enc_msg_length, unpunctured cases
        fn liquid_len(rate: u32, order: u32, n: usize) -> usize {
            match (rate, order) {
                (2, 7) => 2 * n + 2,
                (2, 9) => 2 * n + 2,
                (3, 9) => 3 * n + 3,
                (6, 15) => 6 * n + 11,
                _ => unreachable!(),
            }
        }

        for (name, params, scheme) in base_codes() {
            let mut c = Convolutional::new(params, None);

            for n in [8usize, 16, 64, 100, 255] {
                let want = liquid_len(params.rate, params.order, n);

                // the scheme table must report liquid's length
                assert_eq!(
                    scheme.enc_msg_len(n),
                    want,
                    "{}: scheme length disagrees with liquid at n={}",
                    name,
                    n
                );

                let mut encoded = vec![0u8; want];
                let msg = vec![0xa5u8; n];
                c.encode(&msg, &mut encoded).expect("encode must fit exactly");

                // one byte less must not be enough
                let mut too_small = vec![0u8; want - 1];
                assert!(
                    c.encode(&msg, &mut too_small).is_err(),
                    "{}: encoder fit in {} bytes, so liquid's {} is too many at n={}",
                    name,
                    want - 1,
                    want,
                    n
                );
            }
        }
    }

    #[test]
    fn test_conv_matches_liquid_codewords() {
        use super::super::test_data_conv::*;
        use crate::fec::{Fec, FecScheme as S};

        let cases: &[(S, usize, &[u8])] = &[
            (S::ConvV27, 8, &CONV_V27_ENC_8),
            (S::ConvV27, 64, &CONV_V27_ENC_64),
            (S::ConvV29, 8, &CONV_V29_ENC_8),
            (S::ConvV29, 64, &CONV_V29_ENC_64),
            (S::ConvV39, 8, &CONV_V39_ENC_8),
            (S::ConvV39, 64, &CONV_V39_ENC_64),
            (S::ConvV615, 8, &CONV_V615_ENC_8),
            (S::ConvV615, 64, &CONV_V615_ENC_64),
            (S::ConvV27P23, 8, &CONV_V27P23_ENC_8),
            (S::ConvV27P23, 64, &CONV_V27P23_ENC_64),
            (S::ConvV27P78, 8, &CONV_V27P78_ENC_8),
            (S::ConvV27P78, 64, &CONV_V27P78_ENC_64),
            (S::ConvV29P23, 8, &CONV_V29P23_ENC_8),
            (S::ConvV29P23, 64, &CONV_V29P23_ENC_64),
            (S::ConvV29P78, 8, &CONV_V29P78_ENC_8),
            (S::ConvV29P78, 64, &CONV_V29P78_ENC_64),
        ];

        for &(scheme, n, want) in cases {
            let mut q = Fec::new(scheme).unwrap();
            let msg: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();

            let mut got = vec![0u8; q.enc_msg_len(n)];
            q.encode(&msg, &mut got).unwrap();

            assert_eq!(got.len(), want.len(), "{:?} n={}: encoded length", scheme, n);
            assert_eq!(
                &got[..], want,
                "{:?} n={}: codeword differs from liquid",
                scheme, n
            );

            // and the same bytes must decode back
            let mut dec = vec![0u8; n];
            q.decode(n, want, &mut dec).unwrap();
            assert_eq!(dec, msg, "{:?} n={}: liquid codeword failed to decode", scheme, n);
        }
    }

    #[test]
    fn test_conv_scratch_reuse_shrinking() {
        use crate::fec::FecScheme as S;

        for scheme in [S::ConvV27, S::ConvV39, S::ConvV27P23, S::ConvV29P78] {
            let (params, matrix) = conv_scheme_params(scheme).unwrap();
            let mut c = Convolutional::new(params, matrix);

            // descending sizes, so every call after the first reuses a buffer
            // larger than it needs
            for n in [256usize, 200, 64, 33, 8] {
                let msg: Vec<u8> = (0..n).map(|i| ((i * 91 + 7) & 0xff) as u8).collect();

                let mut enc = vec![0u8; scheme.enc_msg_len(n)];
                c.encode(&msg, &mut enc).unwrap();

                let mut dec = vec![0u8; n];
                c.decode(n, &enc, &mut dec).unwrap();
                assert_eq!(dec, msg, "{:?} n={}: hard decode after shrink", scheme, n);

                // and the soft path, which sizes scratch in samples not bytes
                let mut soft = vec![0u8; enc.len() * 8];
                for (i, byte) in enc.iter().enumerate() {
                    for j in 0..8 {
                        soft[8 * i + j] = if byte & (0x80 >> j) != 0 { 255 } else { 0 };
                    }
                }
                let mut dec_soft = vec![0u8; n];
                c.decode_soft(n, &soft, &mut dec_soft).unwrap();
                assert_eq!(dec_soft, msg, "{:?} n={}: soft decode after shrink", scheme, n);
            }
        }
    }

}
