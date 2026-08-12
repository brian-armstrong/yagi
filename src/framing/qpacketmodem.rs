//
// qpacketmodem : packet encoder/decoder with modulation
//

use num_complex::Complex32;

use crate::error::{Error, Result};
use crate::fec::{CrcScheme, FecScheme};
use crate::modem::modem::{Modem, ModulationScheme};

use super::qpacketsymbolizer::QPacketSymbolizer;

/// packet encoder/decoder with modulation
#[derive(Debug, Clone)]
pub struct QPacketModem {
    /// payload modulator/demodulator
    mod_payload: Modem,
    /// packet encoder/decoder producing symbol indices
    sym: QPacketSymbolizer,
    /// payload symbols (modulator input, demod output)
    payload_mod: Vec<u8>,
    /// soft values from the demodulator, bps per symbol
    payload_soft: Vec<u8>,
    /// soft values received so far, for the incremental decoder
    n: usize,
    /// estimated error vector magnitude
    evm: f32,
}

impl QPacketModem {
    /// create packet encoder with a particular configuration
    ///
    ///  payload_len :   length of payload message [bytes]
    ///  check       :   data integrity check, e.g CrcScheme::Crc32
    ///  fec0        :   forward error-correction scheme (inner)
    ///  fec1        :   forward error-correction scheme (outer)
    ///  ms          :   modulation scheme, e.g. ModulationScheme::Psk4
    pub fn new(
        payload_len: usize,
        check: CrcScheme,
        fec0: FecScheme,
        fec1: FecScheme,
        ms: ModulationScheme,
    ) -> Result<Self> {
        // create payload modem and get bits per symbol
        let mod_payload = Modem::new(ms)?;
        let bits_per_symbol = mod_payload.get_bps();

        // create the symbolizer, which owns the packetizer
        let sym = QPacketSymbolizer::new(payload_len, check, fec0, fec1, bits_per_symbol)?;

        Ok(Self {
            mod_payload,
            payload_mod: vec![0u8; sym.symbol_len()],
            payload_soft: vec![0u8; sym.soft_bit_len()],
            sym,
            n: 0,
            evm: 0.0,
        })
    }

    /// reconfigure object with particular parameters
    ///
    ///  payload_len :   length of payload message [bytes]
    ///  check       :   data integrity check, e.g CrcScheme::Crc32
    ///  fec0        :   forward error-correction scheme (inner)
    ///  fec1        :   forward error-correction scheme (outer)
    ///  ms          :   modulation scheme, e.g. ModulationScheme::Psk4
    pub fn reconfigure(
        &mut self,
        payload_len: usize,
        check: CrcScheme,
        fec0: FecScheme,
        fec1: FecScheme,
        ms: ModulationScheme,
    ) -> Result<()> {
        *self = Self::new(payload_len, check, fec0, fec1, ms)?;
        Ok(())
    }

    /// reset internal state of modem object
    pub fn reset(&mut self) {
        self.n = 0;
        self.evm = 0.0;
    }

    /// get length of encoded frame in symbols
    pub fn frame_len(&self) -> usize {
        self.sym.symbol_len()
    }

    /// get unencoded/decoded payload length (bytes)
    pub fn payload_len(&self) -> usize {
        self.sym.payload_len()
    }

    /// get data integrity check
    pub fn crc(&self) -> CrcScheme {
        self.sym.crc()
    }

    /// get inner forward error-correction scheme
    pub fn fec0(&self) -> FecScheme {
        self.sym.fec0()
    }

    /// get outer forward error-correction scheme
    pub fn fec1(&self) -> FecScheme {
        self.sym.fec1()
    }

    /// get modulation scheme
    pub fn modscheme(&self) -> ModulationScheme {
        self.mod_payload.get_scheme()
    }

    /// get demodulator phase error (instantaneous) [radians]
    pub fn demodulator_phase_error(&self) -> f32 {
        self.mod_payload.get_demodulator_phase_error()
    }

    /// get demodulator error-vector magnitude after frame was received
    pub fn demodulator_evm(&self) -> f32 {
        self.evm
    }

    /// encode and modulate packet into modulated frame samples
    ///
    /// liquid: qpacketmodem_encode
    ///
    ///  payload :   unencoded payload bytes
    ///  frame   :   encoded/modulated payload symbols
    pub fn encode(&mut self, payload: &[u8], frame: &mut [Complex32]) -> Result<()> {
        self.encode_inner(Some(payload), frame)
    }

    /// encode and modulate an all-zero packet into modulated frame samples
    ///
    ///  frame   :   encoded/modulated payload symbols
    pub fn encode_zero(&mut self, frame: &mut [Complex32]) -> Result<()> {
        self.encode_inner(None, frame)
    }

    /// shared by encode and encode_zero; None encodes an all-zero payload
    fn encode_inner(&mut self, payload: Option<&[u8]>, frame: &mut [Complex32]) -> Result<()> {
        self.check_frame_len(frame.len())?;

        // encode payload into symbol indices
        match payload {
            Some(p) => self.sym.encode(p, &mut self.payload_mod)?,
            None => self.sym.encode_zero(&mut self.payload_mod)?,
        };

        // modulate symbols
        for i in 0..self.frame_len() {
            frame[i] = self.mod_payload.modulate(self.payload_mod[i] as u32)?;
        }
        Ok(())
    }

    /// decode packet from modulated frame samples using hard-decision decoding,
    /// returning whether the data integrity check passed
    ///
    /// liquid: qpacketmodem_decode
    ///
    ///  frame   :   encoded/modulated payload symbols
    ///  payload :   recovered decoded payload bytes
    pub fn decode(&mut self, frame: &[Complex32], payload: &mut [u8]) -> Result<bool> {
        self.check_frame_len(frame.len())?;

        // demodulate to symbol indices, then hand them over to be packed
        self.evm = 0.0;
        for i in 0..self.frame_len() {
            self.payload_mod[i] = self.mod_payload.demodulate(frame[i])? as u8;
            self.accumulate_evm();
        }
        self.finish_evm();

        self.sym.decode(&mut self.payload_mod, payload)
    }

    /// decode packet from modulated frame samples using soft-decision decoding,
    /// returning whether the data integrity check passed
    ///
    /// liquid: qpacketmodem_decode_soft
    ///
    ///  frame   :   encoded/modulated payload symbols
    ///  payload :   recovered decoded payload bytes
    pub fn decode_soft(&mut self, frame: &[Complex32], payload: &mut [u8]) -> Result<bool> {
        self.check_frame_len(frame.len())?;

        // demodulate soft values for the whole frame, then decode them
        let bps = self.sym.bits_per_symbol();
        self.evm = 0.0;
        for i in 0..self.frame_len() {
            self.mod_payload
                .demodulate_soft(frame[i], &mut self.payload_soft[i * bps..(i + 1) * bps])?;
            self.accumulate_evm();
        }
        self.finish_evm();

        self.decode_soft_buffered(payload)
    }

    /// accumulate one received symbol, returning whether the frame is complete
    ///
    /// pairs with [`Self::finish_soft_decode`]; liquid: qpacketmodem_decode_soft_sym
    ///
    ///  symbol  :   input received symbol before demodulation
    pub fn push_soft_symbol(&mut self, symbol: Complex32) -> Result<bool> {
        let bps = self.sym.bits_per_symbol();
        self.mod_payload
            .demodulate_soft(symbol, &mut self.payload_soft[self.n..self.n + bps])?;
        self.n += bps;
        Ok(self.n == self.sym.soft_bit_len())
    }

    /// decode the accumulated frame, after [`Self::push_soft_symbol`] reports it
    /// is complete
    ///
    /// liquid: qpacketmodem_decode_soft_payload
    ///
    ///  payload :   output payload [bytes]
    pub fn finish_soft_decode(&mut self, payload: &mut [u8]) -> Result<bool> {
        if self.n != self.sym.soft_bit_len() {
            return Err(Error::Config(
                "qpacketmodem finish_soft_decode(), insufficient number of symbols received"
                    .into(),
            ));
        }
        self.n = 0;
        self.decode_soft_buffered(payload)
    }

    /// decode the accumulated soft values
    fn decode_soft_buffered(&mut self, payload: &mut [u8]) -> Result<bool> {
        let soft_len = self.sym.soft_len();
        self.sym.decode_soft(&self.payload_soft[..soft_len], payload)
    }

    /// the frame buffer must hold every symbol
    fn check_frame_len(&self, len: usize) -> Result<()> {
        if len < self.frame_len() {
            return Err(Error::Config(format!(
                "frame buffer too small: {} < {}",
                len,
                self.frame_len()
            )));
        }
        Ok(())
    }

    /// accumulate the squared error vector magnitude for one symbol
    fn accumulate_evm(&mut self) {
        let e = self.mod_payload.get_demodulator_evm();
        self.evm += e * e;
    }

    /// convert the accumulated squared error into the dB estimate
    fn finish_evm(&mut self) {
        self.evm = 10.0 * (self.evm / self.frame_len() as f32).log10();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::randnf;
    use rand::Rng;
    use test_macro::autotest_annotate;

    fn qpacketmodem_modulated(
        payload_len: usize,
        check: CrcScheme,
        fec0: FecScheme,
        fec1: FecScheme,
        ms: ModulationScheme,
    ) {
        // create and configure packet encoder/decoder object
        let mut q = QPacketModem::new(payload_len, check, fec0, fec1, ms).unwrap();

        // initialize payload
        let mut rng = rand::thread_rng();
        let payload_tx: Vec<u8> = (0..payload_len).map(|_| rng.gen::<u8>()).collect();
        let mut payload_rx: Vec<u8> = (0..payload_len).map(|_| rng.gen::<u8>()).collect();

        // get frame length and allocate memory for frame samples
        let frame_len = q.frame_len();
        let mut frame = vec![Complex32::new(0.0, 0.0); frame_len];

        // encode frame
        q.encode(&payload_tx, &mut frame).unwrap();

        // decode frame
        let crc_pass = q.decode_soft(&frame, &mut payload_rx).unwrap();

        // check to see that frame was recovered
        assert!(crc_pass);
        assert_eq!(payload_tx, payload_rx);
    }

    // note: liquid uses Psk4 here, but it's meant to be Bpsk. We test Bpsk.
    #[test]
    #[autotest_annotate(autotest_qpacketmodem_bpsk)]
    fn test_qpacketmodem_bpsk() {
        qpacketmodem_modulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, ModulationScheme::Bpsk);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_qpsk)]
    fn test_qpacketmodem_qpsk() {
        qpacketmodem_modulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, ModulationScheme::Psk4);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_psk8)]
    fn test_qpacketmodem_psk8() {
        qpacketmodem_modulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, ModulationScheme::Psk8);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_qam16)]
    fn test_qpacketmodem_qam16() {
        qpacketmodem_modulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, ModulationScheme::Qam16);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_sqam32)]
    fn test_qpacketmodem_sqam32() {
        qpacketmodem_modulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, ModulationScheme::Sqam32);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_qam64)]
    fn test_qpacketmodem_qam64() {
        qpacketmodem_modulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, ModulationScheme::Qam64);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_sqam128)]
    fn test_qpacketmodem_sqam128() {
        qpacketmodem_modulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, ModulationScheme::Sqam128);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_qam256)]
    fn test_qpacketmodem_qam256() {
        qpacketmodem_modulated(400, CrcScheme::Crc32, FecScheme::None, FecScheme::None, ModulationScheme::Qam256);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_evm)]
    fn test_qpacketmodem_evm() {
        let payload_len = 800;
        let check = CrcScheme::Crc32;
        let fec0 = FecScheme::None;
        let fec1 = FecScheme::None;
        let ms = ModulationScheme::Psk4;
        let snr_db = 25.0f32;

        // create and configure packet encoder/decoder object
        let mut q = QPacketModem::new(payload_len, check, fec0, fec1, ms).unwrap();

        // get frame length and allocate memory for frame samples
        let frame_len = q.frame_len();
        let mut frame = vec![Complex32::new(0.0, 0.0); frame_len];

        // encode frame (zero payload)
        q.encode_zero(&mut frame).unwrap();

        // add noise
        let nstd = 10f32.powf(-snr_db / 20.0);
        for f in frame.iter_mut() {
            *f += nstd * Complex32::new(randnf(), randnf()) * std::f32::consts::FRAC_1_SQRT_2;
        }

        // decode frame and get EVM estimate
        let mut payload_rx = vec![0u8; payload_len];
        q.decode_soft(&frame, &mut payload_rx).unwrap();
        let evm = q.demodulator_evm();

        // check EVM estimate; don't bother to check that the frame was recovered
        assert!(
            (-evm - snr_db).abs() < 0.5,
            "EVM: {:.3} dB, SNR: {:.3} dB",
            evm,
            snr_db
        );
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_copy)]
    fn test_qpacketmodem_copy() {
        let payload_len = 400;
        let check = CrcScheme::Crc24;
        let fec0 = FecScheme::Secded7264;
        let fec1 = FecScheme::Hamming128;
        let ms = ModulationScheme::Pi4Dqpsk;

        // create and configure packet encoder/decoder object
        let mut q0 = QPacketModem::new(payload_len, check, fec0, fec1, ms).unwrap();

        // initialize buffers
        let frame_len = q0.frame_len();
        let mut rng = rand::thread_rng();
        let payload_tx: Vec<u8> = (0..payload_len).map(|_| rng.gen::<u8>()).collect();

        // encode frame symbols
        let mut frame_syms_0 = vec![Complex32::new(0.0, 0.0); frame_len];
        q0.encode(&payload_tx, &mut frame_syms_0).unwrap();

        // copy object, then have both encode and compare outputs
        //
        // the copy carries the original's state, including a differential
        // scheme's accumulated phase, so both must encode from the copy point
        // forward for their outputs to agree
        let mut q1 = q0.clone();

        q0.encode(&payload_tx, &mut frame_syms_0).unwrap();
        let mut frame_syms_1 = vec![Complex32::new(0.0, 0.0); frame_len];
        q1.encode(&payload_tx, &mut frame_syms_1).unwrap();
        assert_eq!(frame_syms_0, frame_syms_1);

        // initialize received vector (can be random; just testing for equality
        // with objects)
        for i in 0..frame_len {
            frame_syms_0[i] = Complex32::new(randnf(), randnf());
            frame_syms_1[i] = frame_syms_0[i];
        }

        // decode frame symbols and compare outputs
        let mut payload_rx_0 = vec![0u8; payload_len];
        let mut payload_rx_1 = vec![0u8; payload_len];
        let crc_pass_0 = q0.decode(&frame_syms_0, &mut payload_rx_0).unwrap();
        let crc_pass_1 = q1.decode(&frame_syms_1, &mut payload_rx_1).unwrap();

        // check to see that frame was recovered
        assert_eq!(payload_rx_0, payload_rx_1);
        assert_eq!(crc_pass_0, crc_pass_1);
    }
}
