//
// Reed-Solomon (macros)
//
// the codec itself comes from the `fec` crate; this adds liquid's
// block-splitting layer on top

use fec::reed_solomon::{DecodeError, Decoder, Encoder};

use crate::error::{Error, Result};
#[cfg(test)]
use crate::fec::scheme::{RS_M8_KK, RS_M8_NROOTS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RsLayout {
    pub num_blocks: usize,
    pub dec_block_len: usize,
    pub enc_block_len: usize,
    /// residual padding symbols in the last block
    pub res_block_len: usize,
}

impl RsLayout {
    // Divides the input message into several blocks and allows the decoder to
    // pad each block appropriately.
    //
    // For example : if we are using the 8-bit code,
    //      nroots  = 32
    //      nn      = 255
    //      kk      = 223
    // Let dec_msg_len = 1024, then
    //      num_blocks = ceil(1024/223)
    //                 = ceil(4.5919)
    //                 = 5
    //      dec_block_len = ceil(1024/num_blocks)
    //                    = ceil(204.8)
    //                    = 205
    //      enc_block_len = dec_block_len + nroots
    //                    = 237
    //      res_block_len = mod(num_blocks*dec_block_len,dec_msg_len)
    //                    = mod(5*205,1024)
    //                    = mod(1025,1024)
    //                    = 1 (cannot evenly divide input sequence)
    //      pad = kk - dec_block_len
    //          = 223 - 205
    //          = 18
    //
    // Thus, the 1024-byte input message is broken into 5 blocks, the first
    // four have a length 205, and the last block has a length 204 (which is
    // externally padded to 205, e.g. res_block_len = 1). This code adds 32
    // parity symbols, so each block is extended to 237 bytes. The codec auto-
    // matically extends the internal data to 255 bytes by padding with 18
    // symbols.  Therefore, the final output length is 237 * 5 = 1185 symbols.
    pub(crate) fn new(dec_msg_len: usize, nroots: usize, kk: usize) -> Self {
        // compute the total number of blocks necessary: ceil(dec_msg_len / kk)
        let num_blocks = (dec_msg_len + kk - 1) / kk;

        // compute the decoded block length: ceil(dec_msg_len / num_blocks)
        let dec_block_len = (dec_msg_len + num_blocks - 1) / num_blocks;

        // compute the encoded block length: dec_block_len + nroots
        let enc_block_len = dec_block_len + nroots;

        // compute the residual padding symbols in the last block:
        // mod(num_blocks*dec_block_len, dec_msg_len)
        let res_block_len = (num_blocks * dec_block_len) % dec_msg_len;

        Self {
            num_blocks,
            dec_block_len,
            enc_block_len,
            res_block_len,
        }
    }

    pub(crate) fn enc_msg_len(&self) -> usize {
        self.enc_block_len * self.num_blocks
    }
}

/// Reed-Solomon codec, m=8 (n=255, k=223)
///
/// the encoder and decoder carry scratch buffers, so running them needs
/// `&mut self`
#[derive(Clone)]
pub struct ReedSolomon {
    encoder: Encoder,
    decoder: Decoder,
    nroots: usize,
    kk: usize,
    /// scratch for rebuilding a block
    tblock: [u8; RS_M8_NN],
    /// scratch for one decoded block
    decoded: [u8; RS_M8_NN],
}

impl std::fmt::Debug for ReedSolomon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReedSolomon")
            .field("nroots", &self.nroots)
            .field("kk", &self.kk)
            .finish_non_exhaustive()
    }
}

/// block length of the m=8 code: nn = (1 << symsize) - 1
const RS_M8_NN: usize = 255;

const P8_GENPOLY: u16 = 0x11d;
const P8_FCS: u8 = 1;
const P8_PRIM: u8 = 1;
const P8_NROOTS: usize = 32;

impl ReedSolomon {
    /// create m=8 Reed-Solomon codec, matching liquid's LIQUID_FEC_RS_M8
    pub fn new_m8() -> Self {
        let encoder = Encoder::new(P8_GENPOLY, P8_FCS, P8_PRIM, P8_NROOTS);
        let decoder = Decoder::new(P8_GENPOLY, P8_FCS, P8_PRIM, P8_NROOTS);

        // initialize basic parameters
        let nroots = encoder.min_distance();
        let kk = encoder.message_length();
        debug_assert_eq!(nroots, decoder.min_distance());
        debug_assert_eq!(kk, decoder.message_length());

        // allocate memory for arrays
        Self {
            encoder,
            decoder,
            nroots,
            kk,
            tblock: [0u8; RS_M8_NN],
            decoded: [0u8; RS_M8_NN],
        }
    }

    pub(crate) fn layout(&self, dec_msg_len: usize) -> RsLayout {
        RsLayout::new(dec_msg_len, self.nroots, self.kk)
    }

    /// encode block of data using Reed-Solomon encoder
    ///
    ///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
    ///  msg_enc        :   encoded message
    pub fn encode(&mut self, msg_dec: &[u8], msg_enc: &mut [u8]) -> Result<()> {
        // validate input
        let dec_msg_len = msg_dec.len();
        if dec_msg_len == 0 {
            return Err(Error::Config("input length must be > 0".into()));
        }

        let layout = self.layout(dec_msg_len);

        let mut n0 = 0usize; // input index
        let mut n1 = 0usize; // output index

        for i in 0..layout.num_blocks {
            // the last block is smaller by the residual block length
            let block_size = if i == layout.num_blocks - 1 {
                layout.dec_block_len - layout.res_block_len
            } else {
                layout.dec_block_len
            };

            // encode data, appending parity bits to end of sequence
            self.encoder
                .encode(&msg_dec[n0..n0 + block_size], &mut msg_enc[n1..])
                .map_err(|e| Error::Config(format!("Reed-Solomon encode failed: {:?}", e)))?;

            // the encoder writes parity directly after the data, so a short
            // final block leaves it at the wrong offset. move it out to the
            // fixed stride every block is read back at
            if block_size != layout.dec_block_len {
                msg_enc.copy_within(
                    n1 + block_size..n1 + block_size + self.nroots,
                    n1 + layout.dec_block_len,
                );
                msg_enc[n1 + block_size..n1 + layout.dec_block_len].fill(0);
            }

            // increment counters
            n0 += block_size;
            n1 += layout.enc_block_len;
        }

        // sanity check
        debug_assert_eq!(n0, dec_msg_len);
        debug_assert_eq!(n1, layout.enc_msg_len());
        Ok(())
    }

    /// decode block of data using Reed-Solomon decoder, returning the total
    /// number of byte errors corrected across all blocks
    ///
    /// `Err` is only returned when the request is structurally malformed 
    /// (wrong arguments or bad block). If the decoder detects that there are
    /// more errors than it can correct, this will actually return Ok(0) and
    /// leave the block uncorrected. This enables the Packetizer or other
    /// callers to return the block with errors when desired.
    ///
    ///  dec_msg_len    :   decoded message length (number of bytes)
    ///  msg_enc        :   encoded message
    ///  msg_dec        :   decoded message [size: 1 x dec_msg_len]
    pub fn decode(
        &mut self,
        dec_msg_len: usize,
        msg_enc: &[u8],
        msg_dec: &mut [u8],
    ) -> Result<usize> {
        // validate input
        if dec_msg_len == 0 {
            return Err(Error::Config("output length must be > 0".into()));
        }

        let layout = self.layout(dec_msg_len);

        let mut n0 = 0usize; // output index
        let mut n1 = 0usize; // input index
        let mut total_corrected = 0usize;

        for i in 0..layout.num_blocks {
            // the last block is smaller by the residual block length
            let block_size = if i == layout.num_blocks - 1 {
                layout.dec_block_len - layout.res_block_len
            } else {
                layout.dec_block_len
            };

            // copy sequence. the block's length is what tells the decoder how
            // much shortening to assume
            let enc = &msg_enc[n1..n1 + layout.enc_block_len];
            self.tblock[..block_size].copy_from_slice(&enc[..block_size]);
            self.tblock[block_size..block_size + self.nroots]
                .copy_from_slice(&enc[layout.dec_block_len..layout.dec_block_len + self.nroots]);

            match self.decoder.decode(
                &self.tblock[..block_size + self.nroots],
                &mut self.decoded[..block_size],
            ) {
                Ok(corrected) => {
                    total_corrected += corrected;
                    msg_dec[n0..n0 + block_size].copy_from_slice(&self.decoded[..block_size]);
                }
                Err(DecodeError::TooManyErrors) => {
                    // allow this case to propagate through uncorrected
                    msg_dec[n0..n0 + block_size].copy_from_slice(&self.tblock[..block_size]);
                }
                Err(e) => {
                    return Err(Error::Internal(format!(
                        "Reed-Solomon decode failed: {e} \
                         (block {i} of {}, block_size {block_size}, nroots {})",
                        layout.num_blocks, self.nroots
                    )))
                }
            }

            // increment counters
            n0 += block_size;
            n1 += layout.enc_block_len;
        }

        // sanity check
        debug_assert_eq!(n0, dec_msg_len);
        debug_assert_eq!(n1, layout.enc_msg_len());
        Ok(total_corrected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::scheme::rs_enc_msg_len;
    use rand::Rng;

    #[test]
    fn test_rs_layout_matches_enc_msg_len() {
        for n in 1..=600usize {
            let layout = RsLayout::new(n, RS_M8_NROOTS, RS_M8_KK);
            assert_eq!(
                layout.enc_msg_len(),
                rs_enc_msg_len(n, RS_M8_NROOTS, RS_M8_KK),
                "layout disagrees with enc_msg_len for length {}",
                n
            );
        }
    }

    #[test]
    fn test_rs_layout_liquid_example() {
        let layout = RsLayout::new(1024, RS_M8_NROOTS, RS_M8_KK);
        assert_eq!(layout.num_blocks, 5);
        assert_eq!(layout.dec_block_len, 205);
        assert_eq!(layout.enc_block_len, 237);
        assert_eq!(layout.enc_msg_len(), 1185);
    }

    #[test]
    fn test_rs_roundtrip_clean() {
        let mut rs = ReedSolomon::new_m8();
        let mut rng = rand::thread_rng();

        for n in [1usize, 8, 64, 223, 224, 300, 1024] {
            let msg: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();
            let enc_len = rs.layout(n).enc_msg_len();

            let mut encoded = vec![0u8; enc_len];
            let mut decoded = vec![0u8; n];

            rs.encode(&msg, &mut encoded).unwrap();
            let corrected = rs.decode(n, &encoded, &mut decoded).unwrap();

            assert_eq!(corrected, 0, "clean block reported corrections at n={}", n);
            assert_eq!(msg, decoded, "clean round trip failed at n={}", n);
        }
    }

    #[test]
    fn test_rs_corrects_byte_errors() {
        let mut rs = ReedSolomon::new_m8();
        let mut rng = rand::thread_rng();

        for n in [64usize, 223, 300] {
            let msg: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();
            let layout = rs.layout(n);

            let mut encoded = vec![0u8; layout.enc_msg_len()];
            let mut decoded = vec![0u8; n];

            rs.encode(&msg, &mut encoded).unwrap();

            // corrupt encoded message. can withstand up to 16 symbol errors
            for k in 0..16 {
                encoded[k * 3] ^= 0xff;
            }

            let corrected = rs.decode(n, &encoded, &mut decoded).unwrap();

            assert_eq!(msg, decoded, "failed to correct 16 byte errors at n={}", n);
            assert_eq!(corrected, 16, "unexpected correction count at n={}", n);
        }
    }

    #[test]
    fn test_rs_uncorrectable_returns_data_not_error() {
        let mut rs = ReedSolomon::new_m8();

        for n in [64usize, 223, 300] {
            let msg: Vec<u8> = (0..n).map(|i| ((i * 11 + 5) % 256) as u8).collect();
            let layout = rs.layout(n);

            let mut encoded = vec![0u8; layout.enc_msg_len()];
            rs.encode(&msg, &mut encoded).unwrap();

            // 17+ symbol errors per block is past the 16-symbol budget
            for k in 0..40 {
                encoded[k * 2] ^= 0xff;
            }

            let mut decoded = vec![0xAAu8; n];
            let corrected = rs
                .decode(n, &encoded, &mut decoded)
                .unwrap_or_else(|e| panic!("n={n}: uncorrectable block returned Err({e})"));

            // the buffer was written, not left at its fill value
            assert!(
                decoded.iter().any(|&b| b != 0xAA),
                "n={n}: decode left the output untouched"
            );
            // and the failed block contributed nothing to the count
            assert!(
                corrected == 0,
                "n={n}: reported {corrected} corrections on an uncorrectable block"
            );
        }
    }

    #[test]
    fn test_rs_zero_length_rejected() {
        let mut rs = ReedSolomon::new_m8();
        let mut out = [0u8; 32];
        assert!(rs.encode(&[], &mut out).is_err());

        let mut dec = [0u8; 1];
        assert!(rs.decode(0, &out, &mut dec).is_err());
    }

    #[test]
    fn test_rs_constants_match_crate() {
        let rs = ReedSolomon::new_m8();
        assert_eq!(RS_M8_NROOTS, rs.nroots, "RS_M8_NROOTS is stale");
        assert_eq!(RS_M8_KK, rs.kk, "RS_M8_KK is stale");
        assert_eq!(RS_M8_NROOTS + RS_M8_KK, rs.encoder.block_length());
    }

    // liquid's own encoder output. see test_data_rs.rs
    #[test]
    fn test_rs_matches_liquid_codewords() {
        use super::super::test_data_rs::*;

        let cases: &[(usize, &[u8])] = &[
            (1, &RS_M8_ENC_1),
            (8, &RS_M8_ENC_8),
            (32, &RS_M8_ENC_32),
            (64, &RS_M8_ENC_64),
            (223, &RS_M8_ENC_223),
            (224, &RS_M8_ENC_224),
            (300, &RS_M8_ENC_300),
        ];

        let mut rs = ReedSolomon::new_m8();
        for &(n, want) in cases {
            let msg: Vec<u8> = (0..n).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
            let mut got = vec![0u8; rs.layout(n).enc_msg_len()];
            rs.encode(&msg, &mut got).unwrap();

            assert_eq!(got.len(), want.len(), "n={}: encoded length", n);
            assert_eq!(&got[..], want, "n={}: codeword differs from liquid", n);

            // and the same bytes must decode back
            let mut dec = vec![0u8; n];
            rs.decode(n, want, &mut dec).unwrap();
            assert_eq!(dec, msg, "n={}: liquid codeword failed to decode", n);
        }
    }
}
