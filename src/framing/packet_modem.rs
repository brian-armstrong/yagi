//
// qpacketmodem : packet encoder/decoder with modulation
//

use num_complex::Complex32;

use crate::error::{Error, Result};
use crate::fec::{CrcScheme, FecScheme};
use crate::modem::{Modem, ModulationScheme};

use super::packet_symbolizer::PacketSymbolizer;

/// Packet encoder and decoder that maps coded packets onto a constellation.
///
/// Encoding adds a CRC and up to two FEC stages to a byte payload, then maps
/// the coded bits onto complex constellation symbols. Decoding reverses this
/// with either hard or soft symbol decisions. For symbols that arrive one at
/// a time, use [`Self::push_soft_symbol`] and [`Self::finish_soft_decode`].
/// The object does no synchronization, so its input must already be corrected
/// for timing, carrier, and gain.
///
/// A decode returns `true` when the CRC passes. On `false`, the payload is
/// still written and is likely corrupt. With [`CrcScheme::None`] a decode
/// always returns `true`, which says nothing about the payload.
///
/// Both peers must use the same packet format. A [`FecScheme::RsM8`] stage
/// interleaves whole bytes by default. Enable the `liquid-quirks` feature to
/// use liquid-dsp's depth-4 interleaving and exchange RS8 packets with a
/// liquid-dsp peer. See [`Packetizer`](crate::fec::Packetizer).
#[derive(Debug, Clone)]
#[doc(alias = "QPacketModem")]
pub struct PacketModem {
    /// payload modulator/demodulator
    mod_payload: Modem,
    /// packet encoder/decoder producing symbol indices
    sym: PacketSymbolizer,
    /// payload symbols (modulator input, demod output)
    payload_mod: Vec<u8>,
    /// soft values from the demodulator, bps per symbol
    payload_soft: Vec<u8>,
    /// soft values received so far, for the incremental decoder
    n: usize,
    /// estimated error vector magnitude
    evm: f32,
}

impl PacketModem {
    /// Creates a packet modem for a payload of `payload_len` bytes.
    ///
    /// `outer_fec` is applied first and `inner_fec` second, next to the
    /// channel.
    ///
    /// # Errors
    ///
    /// Returns an error when `modulation_scheme` has no built-in constellation
    /// or [`Packetizer::new`](crate::fec::Packetizer::new) rejects the schemes
    /// for this payload length.
    pub fn new(
        payload_len: usize,
        crc_scheme: CrcScheme,
        outer_fec: FecScheme,
        inner_fec: FecScheme,
        modulation_scheme: ModulationScheme,
    ) -> Result<Self> {
        // create payload modem and get bits per symbol
        let mod_payload = Modem::new(modulation_scheme)?;
        let bits_per_symbol = mod_payload.bps();

        // create the symbolizer, which owns the packetizer
        let sym = PacketSymbolizer::new(payload_len, crc_scheme, outer_fec, inner_fec, bits_per_symbol)?;

        Ok(Self {
            mod_payload,
            payload_mod: vec![0u8; sym.symbol_len()],
            payload_soft: vec![0u8; sym.soft_bit_len()],
            sym,
            n: 0,
            evm: 0.0,
        })
    }

    /// Replaces the configuration, as if newly created with [`Self::new`].
    ///
    /// This discards any partly pushed frame and the EVM estimate.
    ///
    /// # Errors
    ///
    /// Returns an error under the same conditions as [`Self::new`]. The modem
    /// is unchanged on error.
    pub fn reconfigure(
        &mut self,
        payload_len: usize,
        crc_scheme: CrcScheme,
        outer_fec: FecScheme,
        inner_fec: FecScheme,
        modulation_scheme: ModulationScheme,
    ) -> Result<()> {
        *self = Self::new(payload_len, crc_scheme, outer_fec, inner_fec, modulation_scheme)?;
        Ok(())
    }

    /// Discards any partly pushed frame and sets the EVM estimate to zero.
    pub fn reset(&mut self) {
        self.n = 0;
        self.evm = 0.0;
    }

    /// Returns the encoded frame length in constellation symbols.
    pub fn frame_len(&self) -> usize {
        self.sym.symbol_len()
    }

    /// Returns the payload length in bytes.
    pub fn payload_len(&self) -> usize {
        self.sym.payload_len()
    }

    /// Returns the CRC scheme.
    pub fn crc_scheme(&self) -> CrcScheme {
        self.sym.crc_scheme()
    }

    /// Returns the FEC scheme applied first during encoding.
    pub fn outer_fec(&self) -> FecScheme {
        self.sym.outer_fec()
    }

    /// Returns the channel-facing FEC scheme, applied second during encoding.
    pub fn inner_fec(&self) -> FecScheme {
        self.sym.inner_fec()
    }

    /// Returns the modulation scheme.
    pub fn modulation_scheme(&self) -> ModulationScheme {
        self.mod_payload.scheme()
    }

    /// Returns the demodulator phase error for the last symbol demodulated, in
    /// radians.
    pub fn demodulator_phase_error(&self) -> f32 {
        self.mod_payload.demodulator_phase_error()
    }

    /// Returns the error vector magnitude of the last decoded frame, in dB.
    ///
    /// This averages the squared demodulator error over the frame.
    /// [`Self::decode`] and [`Self::decode_soft`] update it, but incremental
    /// decoding does not. It is zero before the first frame and after
    /// [`Self::reset`].
    pub fn demodulator_evm(&self) -> f32 {
        self.evm
    }

    /// Encodes a payload into a frame of constellation symbols.
    ///
    /// `payload` must hold exactly [`Self::payload_len`] bytes. The first
    /// [`Self::frame_len`] symbols of `frame` are written, and the rest are
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `payload` has the wrong length or `frame` is
    /// shorter than [`Self::frame_len`]. `frame` is unchanged on error.
    pub fn encode(&mut self, payload: &[u8], frame: &mut [Complex32]) -> Result<()> {
        self.encode_inner(Some(payload), frame)
    }

    /// Encodes an all-zero payload into a frame of constellation symbols.
    ///
    /// The first [`Self::frame_len`] symbols of `frame` are written, and the
    /// rest are unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `frame` is shorter than [`Self::frame_len`].
    /// `frame` is unchanged on error.
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

    /// Decodes a frame using hard symbol decisions.
    ///
    /// Only the first [`Self::frame_len`] symbols of `frame` are read. The
    /// first [`Self::payload_len`] bytes of `payload` are written. Returns
    /// `true` when the CRC passes. On `false`, `payload` still receives the
    /// recovered bytes, which are likely corrupt.
    ///
    /// # Errors
    ///
    /// Returns an error when `frame` is shorter than [`Self::frame_len`] or
    /// `payload` is shorter than [`Self::payload_len`]. `payload` is unchanged
    /// on error. A short `payload` is found after demodulation, so
    /// [`Self::demodulator_evm`] then describes the rejected frame.
    pub fn decode(&mut self, frame: &[Complex32], payload: &mut [u8]) -> Result<bool> {
        self.check_frame_len(frame.len())?;

        // demodulate to symbol indices, then hand them over to be packed
        self.evm = 0.0;
        for i in 0..self.frame_len() {
            self.payload_mod[i] = self.mod_payload.demodulate(frame[i])? as u8;
            self.accumulate_evm();
        }
        self.finish_evm();

        self.sym.decode(&self.payload_mod, payload)
    }

    /// Decodes a frame using soft symbol decisions.
    ///
    /// This behaves like [`Self::decode`], but the inner FEC stage decodes
    /// from soft bit values. It uses the same storage as
    /// [`Self::push_soft_symbol`], so it corrupts a partly pushed frame. Call
    /// [`Self::reset`] before resuming incremental decoding.
    ///
    /// # Errors
    ///
    /// Returns an error when `frame` is shorter than [`Self::frame_len`] or
    /// `payload` is shorter than [`Self::payload_len`]. `payload` is unchanged
    /// on error. A short `payload` is found after demodulation, so
    /// [`Self::demodulator_evm`] then describes the rejected frame.
    pub fn decode_soft(&mut self, frame: &[Complex32], payload: &mut [u8]) -> Result<bool> {
        self.check_frame_len(frame.len())?;

        // demodulate soft values for the whole frame, then decode them
        let bps = self.sym.bits_per_symbol();
        self.evm = 0.0;
        for i in 0..self.frame_len() {
            self.mod_payload.demodulate_soft(frame[i], &mut self.payload_soft[i * bps..(i + 1) * bps])?;
            self.accumulate_evm();
        }
        self.finish_evm();

        self.decode_soft_buffered(payload)
    }

    /// Demodulates one symbol and stores its soft bit values for incremental
    /// decoding.
    ///
    /// Returns `true` once [`Self::frame_len`] symbols have been pushed. Then
    /// call [`Self::finish_soft_decode`] before pushing another symbol. This
    /// path does not update [`Self::demodulator_evm`]. This never returns an
    /// error.
    ///
    /// # Panics
    ///
    /// Panics when a complete frame is already held, because
    /// [`Self::finish_soft_decode`] or [`Self::reset`] was not called after
    /// the previous frame. A configuration whose frame has zero symbols
    /// panics on the first push.
    pub fn push_soft_symbol(&mut self, symbol: Complex32) -> Result<bool> {
        let bps = self.sym.bits_per_symbol();
        self.mod_payload.demodulate_soft(symbol, &mut self.payload_soft[self.n..self.n + bps])?;
        self.n += bps;
        Ok(self.n == self.sym.soft_bit_len())
    }

    /// Decodes the frame collected by [`Self::push_soft_symbol`].
    ///
    /// The first [`Self::payload_len`] bytes of `payload` are written. Returns
    /// `true` when the CRC passes. On `false`, `payload` still receives the
    /// recovered bytes, which are likely corrupt. The collected frame is
    /// discarded, so the next push starts a new frame.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than [`Self::frame_len`] symbols have been
    /// pushed, which keeps them for further pushes. Also returns an error when
    /// `payload` is shorter than [`Self::payload_len`]. In that case the
    /// frame is already discarded and `payload` is unchanged.
    pub fn finish_soft_decode(&mut self, payload: &mut [u8]) -> Result<bool> {
        if self.n != self.sym.soft_bit_len() {
            return Err(Error::Config(
                "qpacketmodem finish_soft_decode(), insufficient number of symbols received".into(),
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
            return Err(Error::Config(format!("frame buffer too small: {} < {}", len, self.frame_len())));
        }
        Ok(())
    }

    /// accumulate the squared error vector magnitude for one symbol
    fn accumulate_evm(&mut self) {
        let e = self.mod_payload.demodulator_evm();
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
        crc_scheme: CrcScheme,
        outer_fec: FecScheme,
        inner_fec: FecScheme,
        modulation_scheme: ModulationScheme,
    ) {
        // create and configure packet encoder/decoder object
        let mut q = PacketModem::new(payload_len, crc_scheme, outer_fec, inner_fec, modulation_scheme).unwrap();

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
        let crc_scheme = CrcScheme::Crc32;
        let outer_fec = FecScheme::None;
        let inner_fec = FecScheme::None;
        let modulation_scheme = ModulationScheme::Psk4;
        let snr_db = 25.0f32;

        // create and configure packet encoder/decoder object
        let mut q = PacketModem::new(payload_len, crc_scheme, outer_fec, inner_fec, modulation_scheme).unwrap();

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
        assert!((-evm - snr_db).abs() < 0.5, "EVM: {:.3} dB, SNR: {:.3} dB", evm, snr_db);
    }

    #[test]
    #[autotest_annotate(autotest_qpacketmodem_copy)]
    fn test_qpacketmodem_copy() {
        let payload_len = 400;
        let crc_scheme = CrcScheme::Crc24;
        let outer_fec = FecScheme::Secded7264;
        let inner_fec = FecScheme::Hamming128;
        let modulation_scheme = ModulationScheme::Pi4Dqpsk;

        // create and configure packet encoder/decoder object
        let mut q0 = PacketModem::new(payload_len, crc_scheme, outer_fec, inner_fec, modulation_scheme).unwrap();

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
