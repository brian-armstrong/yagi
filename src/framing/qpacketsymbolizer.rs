//
// qpacketsymbolizer
//
// Packet encoder/decoder producing symbol indices rather than modulated samples.
// Wraps a packetizer, which adds a CRC and two layers of forward error
// correction, and repacks its encoded bytes into bps-bit symbols.

use crate::error::{Error, Result};
use crate::fec::{CrcScheme, FecScheme, Packetizer};
use crate::utility::bits::repack_bytes;

/// packet encoder/decoder producing symbol indices
#[derive(Debug, Clone)]
pub struct QPacketSymbolizer {
    /// packet encoder/decoder
    packetizer: Packetizer,
    /// bits per symbol
    bits_per_symbol: usize,
    /// number of decoded payload bytes
    payload_dec_len: usize,
    /// number of encoded payload bytes
    byte_len: usize,
    /// number of symbols the encoded payload occupies
    symbol_len: usize,
    /// encoded payload; holds bytes, or soft values when soft-decoding
    payload_enc: Vec<u8>,
}

impl QPacketSymbolizer {
    /// create packet encoder/decoder with a particular configuration
    ///
    ///  payload_len     :   length of payload message [bytes]
    ///  check           :   data integrity check, e.g CrcScheme::Crc32
    ///  fec0            :   forward error-correction scheme (inner)
    ///  fec1            :   forward error-correction scheme (outer)
    ///  bits_per_symbol :   bits per output symbol
    pub fn new(
        payload_len: usize,
        check: CrcScheme,
        fec0: FecScheme,
        fec1: FecScheme,
        bits_per_symbol: usize,
    ) -> Result<Self> {
        if bits_per_symbol == 0 || bits_per_symbol > 8 {
            return Err(Error::Config(format!(
                "qpacketsymbolizer, bits per symbol must be in 1..=8, got {}",
                bits_per_symbol
            )));
        }

        // create packetizer object and compute encoded payload length
        let packetizer = Packetizer::new(payload_len, check, fec0, fec1)?;
        let byte_len = packetizer.enc_msg_len();

        // number of symbols in the encoded payload, from the number of bits in it
        let symbol_len = (8 * byte_len).div_ceil(bits_per_symbol);

        Ok(Self {
            packetizer,
            bits_per_symbol,
            payload_dec_len: payload_len,
            byte_len,
            symbol_len,
            // leave room for soft-decision decoding
            payload_enc: vec![0u8; bits_per_symbol * symbol_len],
        })
    }

    /// reconfigure object with particular parameters
    ///
    ///  payload_len     :   length of payload message [bytes]
    ///  check           :   data integrity check, e.g CrcScheme::Crc32
    ///  fec0            :   forward error-correction scheme (inner)
    ///  fec1            :   forward error-correction scheme (outer)
    ///  bits_per_symbol :   bits per output symbol
    pub fn reconfigure(
        &mut self,
        payload_len: usize,
        check: CrcScheme,
        fec0: FecScheme,
        fec1: FecScheme,
        bits_per_symbol: usize,
    ) -> Result<()> {
        *self = Self::new(payload_len, check, fec0, fec1, bits_per_symbol)?;
        Ok(())
    }

    /// bits per symbol
    pub fn bits_per_symbol(&self) -> usize {
        self.bits_per_symbol
    }

    /// number of symbols the encoded payload occupies
    pub fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// unencoded/decoded payload length (bytes)
    pub fn payload_len(&self) -> usize {
        self.payload_dec_len
    }

    /// encoded payload length (bytes)
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// number of soft values the encoded payload occupies, one per bit
    pub fn soft_len(&self) -> usize {
        8 * self.byte_len
    }

    /// total soft values a demodulator writes for the whole frame, `bps` per
    /// symbol; exceeds [`Self::soft_len`] when the last symbol is partial
    pub fn soft_bit_len(&self) -> usize {
        self.symbol_len * self.bits_per_symbol
    }

    /// data integrity check
    pub fn crc(&self) -> CrcScheme {
        self.packetizer.crc()
    }

    /// inner forward error-correction scheme
    pub fn fec0(&self) -> FecScheme {
        self.packetizer.fec0()
    }

    /// outer forward error-correction scheme
    pub fn fec1(&self) -> FecScheme {
        self.packetizer.fec1()
    }

    /// bytes produced by unpacking every symbol
    ///
    /// rounding the encoded payload up to a whole number of symbols leaves a
    /// partial trailing symbol, so this exceeds [`Self::byte_len`] by one for bps
    /// in {3,5,6,7}. the padding is under one byte, and the payload is a whole
    /// number of bytes, so the extra byte holds no payload bits
    fn packed_len(&self) -> usize {
        self.soft_bit_len().div_ceil(8)
    }

    /// encode packet into symbol indices
    ///
    /// liquid: qpacketmodem_encode_syms
    ///
    ///  payload :   unencoded payload bytes
    ///  symbols :   encoded payload symbol indices [size: symbol_len]
    pub fn encode(&mut self, payload: &[u8], symbols: &mut [u8]) -> Result<()> {
        self.packetizer
            .encode(payload, &mut self.payload_enc[..self.byte_len])?;

        self.to_symbols(symbols)
    }

    /// encode an all-zero packet into symbol indices
    ///
    /// liquid: qpacketmodem_encode_syms with a NULL payload
    ///
    ///  symbols :   encoded payload symbol indices [size: symbol_len]
    pub fn encode_zero(&mut self, symbols: &mut [u8]) -> Result<()> {
        self.packetizer.encode_zero(&mut self.payload_enc[..self.byte_len])?;

        self.to_symbols(symbols)
    }

    /// repack the encoded payload into 'bps'-bit symbols
    fn to_symbols(&self, symbols: &mut [u8]) -> Result<()> {
        let num_written = repack_bytes(
            &self.payload_enc[..self.byte_len],
            8,
            &mut symbols[..self.symbol_len],
            self.bits_per_symbol,
        )?;

        if num_written != self.symbol_len {
            return Err(Error::Config(format!(
                "qpacketsymbolizer, unexpected number of symbols: {} != {}",
                num_written, self.symbol_len
            )));
        }
        Ok(())
    }

    /// decode packet from symbol indices (hard-decision decoding), returning
    /// whether the data integrity check passed
    ///
    /// liquid: qpacketmodem_decode_syms
    ///
    ///  symbols :   received hard-decision symbol indices [size: symbol_len]
    ///  payload :   recovered decoded payload bytes
    pub fn decode(&mut self, symbols: &[u8], payload: &mut [u8]) -> Result<bool> {
        // unpack symbols into the encoded payload
        let packed_len = self.packed_len();
        let num_written = repack_bytes(
            &symbols[..self.symbol_len],
            self.bits_per_symbol,
            &mut self.payload_enc[..packed_len],
            8,
        )?;

        if num_written != packed_len {
            return Err(Error::Config(format!(
                "qpacketsymbolizer, unexpected number of bytes: {} != {}",
                num_written, packed_len
            )));
        }

        self.packetizer
            .decode(&self.payload_enc[..self.byte_len], payload)
    }

    /// decode packet from soft-decision bits
    ///
    /// liquid: qpacketmodem_decode_bits
    ///
    ///  bits    :   received soft-decision bits [size: soft_len]
    ///  payload :   recovered decoded payload bytes
    pub fn decode_soft(&mut self, bits: &[u8], payload: &mut [u8]) -> Result<bool> {
        self.packetizer.decode_soft(bits, payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    fn symbolizer(payload_len: usize, bps: usize) -> QPacketSymbolizer {
        QPacketSymbolizer::new(
            payload_len,
            CrcScheme::Crc32,
            FecScheme::Hamming128,
            FecScheme::None,
            bps,
        )
        .unwrap()
    }

    #[test]
    fn test_symbolizer_lengths() {
        let s = symbolizer(64, 8);
        assert_eq!(s.symbol_len(), s.byte_len());
        assert_eq!(s.packed_len(), s.byte_len());
        assert_eq!(s.soft_len(), 8 * s.byte_len());

        let s = symbolizer(64, 3);
        assert_eq!(s.symbol_len(), (8 * s.byte_len()).div_ceil(3));
        assert_eq!(s.packed_len(), s.soft_bit_len().div_ceil(8));

        let s = symbolizer(63, 3);
        assert_ne!(8 * s.byte_len() % 3, 0);
        assert_eq!(s.packed_len(), s.byte_len() + 1);
    }

    #[test]
    fn test_symbolizer_invalid_bps() {
        for bps in [0usize, 9] {
            assert!(QPacketSymbolizer::new(
                16,
                CrcScheme::Crc32,
                FecScheme::None,
                FecScheme::None,
                bps
            )
            .is_err());
        }
    }

    #[test]
    fn test_symbolizer_buffer_invariants() {
        for payload_len in [1usize, 7, 16, 64, 256] {
            for bps in 1..=8 {
                let s = symbolizer(payload_len, bps);
                let buf = s.payload_enc.len();

                assert!(buf >= s.byte_len(), "{} {}", payload_len, bps);
                assert!(buf >= s.soft_len(), "{} {}", payload_len, bps);
                assert!(buf >= s.symbol_len(), "{} {}", payload_len, bps);
                assert!(buf >= s.packed_len(), "{} {}", payload_len, bps);

                // unpacking recovers at least the payload, and at most one byte
                // more, so the padding can never swallow a payload byte
                assert!(s.packed_len() >= s.byte_len(), "{} {}", payload_len, bps);
                assert!(
                    s.packed_len() <= s.byte_len() + 1,
                    "{} {}",
                    payload_len,
                    bps
                );
            }
        }
    }

    // payload -> symbols -> payload, through the full crc/fec chain
    #[test]
    fn test_symbolizer_round_trip() {
        for payload_len in [1usize, 7, 16, 64, 256] {
            for bps in 1..=8 {
                let mut s = symbolizer(payload_len, bps);

                let payload_tx: Vec<u8> =
                    (0..payload_len).map(|i| (i * 37 + 11) as u8).collect();
                let mut symbols = vec![0u8; s.symbol_len()];
                s.encode(&payload_tx, &mut symbols).unwrap();

                // every symbol must fit in bps bits
                let max = (1u16 << bps) - 1;
                for (i, &sym) in symbols.iter().enumerate() {
                    assert!(sym as u16 <= max, "bps {} symbol {} = {}", bps, i, sym);
                }

                let mut payload_rx = vec![0u8; payload_len];
                let crc_pass = s.decode(&symbols, &mut payload_rx).unwrap();

                assert!(crc_pass, "payload_len {} bps {}", payload_len, bps);
                assert_eq!(payload_tx, payload_rx, "payload_len {} bps {}", payload_len, bps);
            }
        }
    }

    #[test]
    fn test_symbolizer_encode_zero() {
        let mut s = symbolizer(32, 4);
        let mut symbols = vec![0u8; s.symbol_len()];
        s.encode_zero(&mut symbols).unwrap();

        let mut payload_rx = vec![0xffu8; 32];
        let crc_pass = s.decode(&symbols, &mut payload_rx).unwrap();

        assert!(crc_pass);
        assert_eq!(payload_rx, vec![0u8; 32]);
    }

    #[test]
    fn test_symbolizer_soft_bits_path() {
        for bps in 1..=8 {
            let mut s = symbolizer(32, bps);
            let payload_tx: Vec<u8> = (0..32).map(|i| (i * 37 + 11) as u8).collect();
            let mut symbols = vec![0u8; s.symbol_len()];
            s.encode(&payload_tx, &mut symbols).unwrap();

            // expand each symbol into bps saturated soft values, msb first,
            // matching what the modem's soft demodulator writes
            let mut soft = vec![0u8; s.soft_bit_len()];
            for (i, &sym) in symbols.iter().enumerate() {
                for j in 0..bps {
                    soft[i * bps + j] = if sym & (1 << (bps - 1 - j)) != 0 { 255 } else { 0 };
                }
            }

            let mut payload_rx = vec![0u8; 32];
            let crc_pass = s
                .decode_soft(&soft[..s.soft_len()], &mut payload_rx)
                .unwrap();

            assert!(crc_pass, "bps {}", bps);
            assert_eq!(payload_tx, payload_rx, "bps {}", bps);
        }
    }

    fn qpacketmodem_unmodulated(
        payload_len: usize,
        check: CrcScheme,
        fec0: FecScheme,
        fec1: FecScheme,
        bits_per_symbol: usize,
    ) {
        // create and configure packet encoder/decoder object
        let mut q =
            QPacketSymbolizer::new(payload_len, check, fec0, fec1, bits_per_symbol).unwrap();

        // initialize payload
        let mut rng = rand::thread_rng();
        let payload_tx: Vec<u8> = (0..payload_len).map(|_| rng.gen::<u8>()).collect();
        let mut payload_rx: Vec<u8> = (0..payload_len).map(|_| rng.gen::<u8>()).collect();

        // get frame length (symbols) and allocate memory for frame symbols
        let frame_len = q.symbol_len();
        let mut frame_syms = vec![0u8; frame_len];

        // encode frame symbols
        q.encode(&payload_tx, &mut frame_syms).unwrap();

        // decode frame symbols
        let crc_pass = q.decode(&frame_syms, &mut payload_rx).unwrap();

        // check to see that frame was recovered
        assert!(crc_pass);
        assert_eq!(payload_tx, payload_rx);
    }

    // note that this tested a bps of 2 in liquid, but it should be 1. we test 1.
    #[test]
    #[autotest_annotate(autotest_qpacketmodem_unmod_bpsk)]
    fn test_qpacketmodem_unmod_bpsk() {
        qpacketmodem_unmodulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, 1);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_unmod_qpsk)]
    fn test_qpacketmodem_unmod_qpsk() {
        qpacketmodem_unmodulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, 2);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_unmod_psk8)]
    fn test_qpacketmodem_unmod_psk8() {
        qpacketmodem_unmodulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, 3);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_unmod_qam16)]
    fn test_qpacketmodem_unmod_qam16() {
        qpacketmodem_unmodulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, 4);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_unmod_sqam32)]
    fn test_qpacketmodem_unmod_sqam32() {
        qpacketmodem_unmodulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, 5);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_unmod_qam64)]
    fn test_qpacketmodem_unmod_qam64() {
        qpacketmodem_unmodulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, 6);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_unmod_sqam128)]
    fn test_qpacketmodem_unmod_sqam128() {
        qpacketmodem_unmodulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, 7);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_unmod_qam256)]
    fn test_qpacketmodem_unmod_qam256() {
        qpacketmodem_unmodulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, 8);
    }
}
