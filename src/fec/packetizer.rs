//
// packetizer
//
// Chains a CRC, two forward error-correction stages, and an interleaver per
// stage. Buffers ping-pong between the two stages: the encoder runs
// buffer_0 -> buffer_1 and the interleaver runs buffer_1 -> buffer_0.

use crate::error::{Error, Result};
use crate::random::scramble::{scramble_data, unscramble_data};

use super::crc::{self, CrcScheme};
use super::fec::Fec;
use super::interleaver::Interleaver;
use super::scheme::FecScheme;

/// number of fec/interleaver stages
const PLAN_LEN: usize = 2;

/// one fec/interleaver stage
#[derive(Debug, Clone)]
struct PacketizerPlan {
    fec_scheme: FecScheme,
    dec_msg_len: usize,
    enc_msg_len: usize,
    fec: Fec,
    interleaver: Interleaver,
}

/// computes the number of encoded bytes after packetizing
///
///  n      :   number of uncoded input bytes
///  crc    :   error-detecting scheme
///  fec0   :   inner forward error-correction code
///  fec1   :   outer forward error-correction code
pub fn packetizer_compute_enc_msg_len(
    n: usize,
    crc: CrcScheme,
    fec0: FecScheme,
    fec1: FecScheme,
) -> usize {
    let k = n + crc.key_len();
    let n0 = fec0.enc_msg_len(k);
    fec1.enc_msg_len(n0)
}

/// computes the number of decoded bytes before packetizing
///
/// Errors if `k` is not a packet length these schemes can produce. Block codes
/// round up to whole symbols, so most lengths are unreachable
///
///  k      :   number of encoded bytes
///  crc    :   error-detecting scheme
///  fec0   :   inner forward error-correction code
///  fec1   :   outer forward error-correction code
pub fn packetizer_compute_dec_msg_len(
    k: usize,
    crc: CrcScheme,
    fec0: FecScheme,
    fec1: FecScheme,
) -> Result<usize> {
    let mut n_hat = 0usize;
    let mut k_hat = 0usize;

    // check for zero-length packet
    // TODO : implement faster method
    while k_hat < k {
        // compute encoded packet length
        k_hat = packetizer_compute_enc_msg_len(n_hat, crc, fec0, fec1);

        if k_hat == k {
            return Ok(n_hat);
        } else if k_hat > k {
            let lo = packetizer_compute_enc_msg_len(n_hat.saturating_sub(1), crc, fec0, fec1);
            return Err(Error::Config(format!(
                "packetizer_compute_dec_msg_len(), no message length encodes to {} bytes \
                 (nearest are {} and {})",
                k, lo, k_hat,
            )));
        } else {
            n_hat += 1;
        }
    }

    // k == 0, which only a zero-length payload with no crc could produce
    Ok(0)
}

fn interleaver_depth(fs: FecScheme) -> Result<usize> {
    // no error correction applied: nothing to interleave
    if fs == FecScheme::None {
        return Ok(0);
    }

    // for backwards compat, use only a depth-4 interleaving even though it performs worse
    //    for Reed-Solomon.
    #[cfg(feature = "liquid-quirks")]
    {
        Ok(4)
    }

    // by default, interleave at the symbol width of the scheme
    #[cfg(not(feature = "liquid-quirks"))]
    match fs.symbol_bits() {
        8 => Ok(1),
        4 => Ok(2),
        // 2 is not supported. if we swapped the last two stages of the interleaver,
        //   we would get it, but then the interleaver would be incompat with liquid
        1 => Ok(4),
        w => Err(Error::Config(format!(
            "packetizer, interleaver cannot preserve {}-bit symbols",
            w
        ))),
    }
}

/// packetizer object
#[derive(Debug, Clone)]
pub struct Packetizer {
    msg_len: usize,
    packet_len: usize,
    check: CrcScheme,
    crc_length: usize,

    plan: [PacketizerPlan; PLAN_LEN],

    // scaled by 8 so the soft decoder can share them
    buffer_0: Vec<u8>,
    buffer_1: Vec<u8>,
}

impl Packetizer {
    /// create packetizer object
    ///
    ///  n      :   number of uncoded input bytes
    ///  crc    :   error-detecting scheme
    ///  fec0   :   inner forward error-correction code
    ///  fec1   :   outer forward error-correction code
    pub fn new(n: usize, crc: CrcScheme, fec0: FecScheme, fec1: FecScheme) -> Result<Self> {
        let msg_len = n;
        let packet_len = packetizer_compute_enc_msg_len(n, crc, fec0, fec1);
        let crc_length = crc.key_len();

        // create plan
        let mut n0 = n + crc_length;
        let mut plan = Vec::with_capacity(PLAN_LEN);

        for i in 0..PLAN_LEN {
            // set schemes
            let fs = if i == 0 { fec0 } else { fec1 };

            // compute lengths
            let dec_msg_len = n0;
            let enc_msg_len = fs.enc_msg_len(dec_msg_len);

            // create objects
            let f = Fec::new(fs)?;
            let mut q = Interleaver::new(enc_msg_len);

            // interleave no finer than this scheme's symbol
            q.set_depth(interleaver_depth(fs)?);

            plan.push(PacketizerPlan {
                fec_scheme: fs,
                dec_msg_len,
                enc_msg_len,
                fec: f,
                interleaver: q,
            });

            // update length
            n0 = enc_msg_len;
        }

        let plan: [PacketizerPlan; PLAN_LEN] = plan
            .try_into()
            .map_err(|_| Error::Config("failed to build packetizer plan".into()))?;

        // allocate memory for buffers (scale by 8 for soft decoding)
        Ok(Self {
            msg_len,
            packet_len,
            check: crc,
            crc_length,
            plan,
            buffer_0: vec![0u8; 8 * packet_len],
            buffer_1: vec![0u8; 8 * packet_len],
        })
    }

    /// get decoded message length
    pub fn dec_msg_len(&self) -> usize {
        self.msg_len
    }

    /// get encoded message length
    pub fn enc_msg_len(&self) -> usize {
        self.packet_len
    }

    /// get error-detecting scheme
    pub fn crc(&self) -> CrcScheme {
        self.check
    }

    /// get inner forward error-correction code
    pub fn fec0(&self) -> FecScheme {
        self.plan[0].fec_scheme
    }

    /// get outer forward error-correction code
    pub fn fec1(&self) -> FecScheme {
        self.plan[1].fec_scheme
    }

    /// execute the packetizer on an input message
    ///
    ///  msg    :   input message (uncoded bytes)
    ///  pkt    :   encoded output message
    pub fn encode(&mut self, msg: &[u8], pkt: &mut [u8]) -> Result<()> {
        if msg.len() != self.msg_len {
            return Err(Error::Config(format!(
                "input message has wrong length: {} != {}",
                msg.len(),
                self.msg_len
            )));
        }

        // copy input message to internal buffer[0]
        self.buffer_0[..self.msg_len].copy_from_slice(&msg[..self.msg_len]);

        self.encode_buffered(pkt)
    }

    /// execute the packetizer on an all-zero message
    /// 
    /// pkt    :   encoded output message
    pub fn encode_zero(&mut self, pkt: &mut [u8]) -> Result<()> {
        // initialize with zeros
        self.buffer_0[..self.msg_len].fill(0);

        self.encode_buffered(pkt)
    }

    /// append the crc, whiten, and run the fec/interleaver plans over whatever
    /// is already in buffer_0
    fn encode_buffered(&mut self, pkt: &mut [u8]) -> Result<()> {
        if pkt.len() < self.packet_len {
            return Err(Error::Config(format!(
                "output packet too short: {} < {}",
                pkt.len(),
                self.packet_len
            )));
        }

        // compute crc, append to buffer
        let mut key = crc::generate_key(self.check, &self.buffer_0[..self.msg_len]);
        for i in 0..self.crc_length {
            // append byte to buffer
            self.buffer_0[self.msg_len + self.crc_length - i - 1] = (key & 0xff) as u8;

            // shift key by 8 bits
            key >>= 8;
        }

        // whiten input sequence
        scramble_data(&mut self.buffer_0[..self.msg_len + self.crc_length]);

        // execute fec/interleaver plans
        for i in 0..PLAN_LEN {
            // run the encoder: buffer[0] > buffer[1]
            let plan = &mut self.plan[i];
            plan.fec
                .encode(&self.buffer_0[..plan.dec_msg_len], &mut self.buffer_1)?;

            // run the interleaver: buffer[1] > buffer[0]
            plan.interleaver.encode(
                &self.buffer_1[..plan.enc_msg_len],
                &mut self.buffer_0[..plan.enc_msg_len],
            );
        }

        // copy result to output
        pkt[..self.packet_len].copy_from_slice(&self.buffer_0[..self.packet_len]);
        Ok(())
    }

    /// execute the packetizer to decode an input message, returning whether the
    /// data integrity check passed
    ///
    ///  pkt    :   encoded input message
    ///  msg    :   decoded output message
    pub fn decode(&mut self, pkt: &[u8], msg: &mut [u8]) -> Result<bool> {
        if pkt.len() != self.packet_len {
            return Err(Error::Config(format!(
                "input packet has wrong length: {} != {}",
                pkt.len(),
                self.packet_len
            )));
        }
        if msg.len() < self.msg_len {
            return Err(Error::Config(format!(
                "output message too short: {} < {}",
                msg.len(),
                self.msg_len
            )));
        }

        // copy coded message to internal buffer[0]
        self.buffer_0[..self.packet_len].copy_from_slice(&pkt[..self.packet_len]);

        // execute fec/interleaver plans
        for i in (0..PLAN_LEN).rev() {
            // run the de-interleaver: buffer[0] > buffer[1]
            let plan = &mut self.plan[i];
            plan.interleaver.decode(
                &self.buffer_0[..plan.enc_msg_len],
                &mut self.buffer_1[..plan.enc_msg_len],
            );

            // run the decoder: buffer[1] > buffer[0]
            plan.fec
                .decode(plan.dec_msg_len, &self.buffer_1, &mut self.buffer_0)?;
        }

        Ok(self.finish_decode(msg))
    }

    /// execute the packetizer to decode an input message using soft bits
    ///
    ///  pkt    :   encoded input message, 8 bytes per hard bit
    ///  msg    :   decoded output message
    pub fn decode_soft(&mut self, pkt: &[u8], msg: &mut [u8]) -> Result<bool> {
        if pkt.len() != 8 * self.packet_len {
            return Err(Error::Config(format!(
                "input soft packet has wrong length: {} != {}",
                pkt.len(),
                8 * self.packet_len
            )));
        }
        if msg.len() < self.msg_len {
            return Err(Error::Config(format!(
                "output message too short: {} < {}",
                msg.len(),
                self.msg_len
            )));
        }

        // copy coded message to internal buffer[0]
        self.buffer_0[..8 * self.packet_len].copy_from_slice(&pkt[..8 * self.packet_len]);

        //
        // decode outer level using soft decoding
        //

        // run the de-interleaver: buffer[0] > buffer[1]
        let plan = &mut self.plan[1];
        plan.interleaver.decode_soft(
            &self.buffer_0[..8 * plan.enc_msg_len],
            &mut self.buffer_1[..8 * plan.enc_msg_len],
        );

        // run the decoder: buffer[1] > buffer[0]
        plan.fec
            .decode_soft(plan.dec_msg_len, &self.buffer_1, &mut self.buffer_0)?;

        //
        // decode inner level using hard decoding
        //

        // run the de-interleaver: buffer[0] > buffer[1]
        let plan = &mut self.plan[0];
        plan.interleaver.decode(
            &self.buffer_0[..plan.enc_msg_len],
            &mut self.buffer_1[..plan.enc_msg_len],
        );

        // run the decoder: buffer[1] > buffer[0]
        plan.fec
            .decode(plan.dec_msg_len, &self.buffer_1, &mut self.buffer_0)?;

        Ok(self.finish_decode(msg))
    }

    /// unwhiten, strip the crc, and validate; shared by both decode paths
    fn finish_decode(&mut self, msg: &mut [u8]) -> bool {
        // remove sequence whitening
        unscramble_data(&mut self.buffer_0[..self.msg_len + self.crc_length]);

        // strip crc, validate message
        let mut key = 0u32;
        for i in 0..self.crc_length {
            key <<= 8;
            key |= self.buffer_0[self.msg_len + i] as u32;
        }

        // copy result to output
        msg[..self.msg_len].copy_from_slice(&self.buffer_0[..self.msg_len]);

        // return crc validity
        crc::validate_message(self.check, &self.buffer_0[..self.msg_len], key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use test_macro::autotest_annotate;

    // Help function to keep code base small
    fn packetizer_test_codec(n: usize, crc: CrcScheme, fec0: FecScheme, fec1: FecScheme) {
        let pkt_len = packetizer_compute_enc_msg_len(n, crc, fec0, fec1);
        let mut packet = vec![0u8; pkt_len];

        // create object
        let mut p = Packetizer::new(n, crc, fec0, fec1).unwrap();

        // initialize data
        let msg_tx: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let mut msg_rx = vec![0u8; n];

        // encode/decode packet
        p.encode(&msg_tx, &mut packet).unwrap();
        let crc_pass = p.decode(&packet, &mut msg_rx).unwrap();

        assert_eq!(msg_tx, msg_rx);
        assert!(crc_pass);
    }

    #[test]
    #[autotest_annotate(autotest_packetizer_n16_0_0)]
    fn test_packetizer_n16_0_0() {
        packetizer_test_codec(16, CrcScheme::Crc32, FecScheme::None, FecScheme::None);
    }

    #[test]
    #[autotest_annotate(autotest_packetizer_n16_0_1)]
    fn test_packetizer_n16_0_1() {
        packetizer_test_codec(16, CrcScheme::Crc32, FecScheme::None, FecScheme::Rep3);
    }

    #[test]
    #[autotest_annotate(autotest_packetizer_n16_0_2)]
    fn test_packetizer_n16_0_2() {
        packetizer_test_codec(16, CrcScheme::Crc32, FecScheme::None, FecScheme::Hamming74);
    }

    #[test]
    #[autotest_annotate(autotest_packetizer_copy)]
    fn test_packetizer_copy() {
        let msg_len_dec = 57;
        let crc = CrcScheme::Crc32;
        let fec0 = FecScheme::Hamming128;
        let fec1 = FecScheme::Golay2412;

        // compute encoded message length
        let msg_len_enc = packetizer_compute_enc_msg_len(msg_len_dec, crc, fec0, fec1);

        // create object
        let mut q0 = Packetizer::new(msg_len_dec, crc, fec0, fec1).unwrap();

        // initialize random data
        let mut rng = rand::thread_rng();
        let msg_org: Vec<u8> = (0..msg_len_dec).map(|_| rng.gen::<u8>()).collect();

        // encode packet
        let mut msg_enc_0 = vec![0u8; msg_len_enc];
        q0.encode(&msg_org, &mut msg_enc_0).unwrap();

        // copy object, encode, and compare result
        let mut q1 = q0.clone();
        let mut msg_enc_1 = vec![0u8; msg_len_enc];
        q1.encode(&msg_org, &mut msg_enc_1).unwrap();
        assert_eq!(msg_enc_0, msg_enc_1);

        // initialize random data for decoder input
        for i in 0..msg_len_enc {
            msg_enc_0[i] = rng.gen::<u8>();
        }
        msg_enc_1.copy_from_slice(&msg_enc_0);

        // decode and compare
        // NOTE: we don't care if the output is valid; just that they match
        let mut msg_dec_0 = vec![0u8; msg_len_dec];
        let mut msg_dec_1 = vec![0u8; msg_len_dec];
        let crc_pass_0 = q0.decode(&msg_enc_0, &mut msg_dec_0).unwrap();
        let crc_pass_1 = q1.decode(&msg_enc_1, &mut msg_dec_1).unwrap();

        assert_eq!(msg_dec_0, msg_dec_1);
        assert_eq!(crc_pass_0, crc_pass_1);
    }

    #[test]
    fn test_packetizer_decode_soft() {
        for (fec0, fec1) in [
            (FecScheme::None, FecScheme::Hamming74),
            (FecScheme::Hamming128, FecScheme::Golay2412),
            (FecScheme::Rep3, FecScheme::Rep5),
        ] {
            let n = 24;
            let crc = CrcScheme::Crc32;
            let mut p = Packetizer::new(n, crc, fec0, fec1).unwrap();

            let mut rng = rand::thread_rng();
            let msg_tx: Vec<u8> = (0..n).map(|_| rng.gen::<u8>()).collect();

            let pkt_len = p.enc_msg_len();
            let mut packet = vec![0u8; pkt_len];
            p.encode(&msg_tx, &mut packet).unwrap();

            // expand to soft bits, 8 bytes per hard bit
            let mut soft = vec![0u8; 8 * pkt_len];
            for (i, byte) in packet.iter().enumerate() {
                for j in 0..8 {
                    soft[8 * i + j] = if byte & (0x80 >> j) != 0 { 255 } else { 0 };
                }
            }

            let mut msg_rx = vec![0u8; n];
            let crc_pass = p.decode_soft(&soft, &mut msg_rx).unwrap();

            assert_eq!(msg_tx, msg_rx, "{:?}/{:?}: soft decode", fec0, fec1);
            assert!(crc_pass, "{:?}/{:?}: crc failed", fec0, fec1);
        }
    }

    #[test]
    fn test_packetizer_encode_zero() {
        let n = 16;
        let crc = CrcScheme::Crc32;
        let mut p = Packetizer::new(n, crc, FecScheme::Hamming128, FecScheme::Rep3).unwrap();

        let pkt_len = p.enc_msg_len();
        let mut pkt_zero = vec![0u8; pkt_len];
        p.encode_zero(&mut pkt_zero).unwrap();

        // must match encoding an explicit zero message
        let mut pkt_explicit = vec![0u8; pkt_len];
        p.encode(&vec![0u8; n], &mut pkt_explicit).unwrap();
        assert_eq!(pkt_zero, pkt_explicit);

        // and it must round-trip
        let mut msg_rx = vec![0xffu8; n];
        assert!(p.decode(&pkt_zero, &mut msg_rx).unwrap());
        assert_eq!(msg_rx, vec![0u8; n]);
    }

    #[test]
    fn test_packetizer_accessors() {
        let p = Packetizer::new(57, CrcScheme::Crc16, FecScheme::Rep3, FecScheme::Golay2412)
            .unwrap();

        assert_eq!(p.dec_msg_len(), 57);
        assert_eq!(p.crc(), CrcScheme::Crc16);
        assert_eq!(p.fec0(), FecScheme::Rep3);
        assert_eq!(p.fec1(), FecScheme::Golay2412);
        assert_eq!(
            p.enc_msg_len(),
            packetizer_compute_enc_msg_len(57, CrcScheme::Crc16, FecScheme::Rep3, FecScheme::Golay2412)
        );
    }

    #[test]
    fn test_packetizer_dec_msg_len_unreachable() {
        let crc = CrcScheme::Crc32;
        let (fec0, fec1) = (FecScheme::Hamming128, FecScheme::Golay2412);

        // collect the lengths that are actually achievable
        let achievable: std::collections::HashSet<usize> = (0..40)
            .map(|n| packetizer_compute_enc_msg_len(n, crc, fec0, fec1))
            .collect();

        let mut rejected = 0;
        for k in 1..120usize {
            let result = packetizer_compute_dec_msg_len(k, crc, fec0, fec1);
            if achievable.contains(&k) {
                assert!(result.is_ok(), "k={} is achievable but was rejected", k);
            } else {
                assert!(result.is_err(), "k={} is not achievable but was accepted", k);
                rejected += 1;
            }
        }

        assert!(rejected > 0);
    }

    #[test]
    fn test_packetizer_compute_dec_msg_len() {
        let crc = CrcScheme::Crc32;
        for (fec0, fec1) in [
            (FecScheme::None, FecScheme::None),
            (FecScheme::None, FecScheme::Rep3),
            (FecScheme::Hamming128, FecScheme::Golay2412),
        ] {
            for n in 1..=40usize {
                let k = packetizer_compute_enc_msg_len(n, crc, fec0, fec1);
                let n_hat = packetizer_compute_dec_msg_len(k, crc, fec0, fec1).unwrap();

                assert!(n_hat <= n, "{:?}/{:?} n={}: got {}", fec0, fec1, n, n_hat);

                assert_eq!(
                    packetizer_compute_enc_msg_len(n_hat, crc, fec0, fec1),
                    k,
                    "{:?}/{:?} n={}: n_hat={} does not round-trip",
                    fec0,
                    fec1,
                    n,
                    n_hat
                );
            }
        }
    }

    #[test]
    fn test_packetizer_interleaver_depth() {
        // no coding: nothing to interleave
        assert_eq!(interleaver_depth(FecScheme::None).unwrap(), 0);

        // bit-oriented schemes take the finest interleaving
        for fs in [
            FecScheme::Rep3,
            FecScheme::Hamming74,
            FecScheme::Golay2412,
            FecScheme::Secded7264,
            FecScheme::ConvV27,
            FecScheme::ConvV615,
            FecScheme::ConvV29P78,
        ] {
            assert_eq!(fs.symbol_bits(), 1, "{:?}", fs);
            assert_eq!(interleaver_depth(fs).unwrap(), 4, "{:?}", fs);
        }

        // test the quirk case
        assert_eq!(FecScheme::RsM8.symbol_bits(), 8);
        let expected = if cfg!(feature = "liquid-quirks") { 4 } else { 1 };
        assert_eq!(interleaver_depth(FecScheme::RsM8).unwrap(), expected);
    }

    #[test]
    fn test_packetizer_reedsolomon_burst() {
        let n = 64;
        let crc = CrcScheme::Crc32;
        let (fec0, fec1) = (FecScheme::RsM8, FecScheme::None);

        let pkt_len = packetizer_compute_enc_msg_len(n, crc, fec0, fec1);
        let mut p = Packetizer::new(n, crc, fec0, fec1).unwrap();

        let msg_tx: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
        let mut packet = vec![0u8; pkt_len];
        p.encode(&msg_tx, &mut packet).unwrap();

        // corrupt a run of bytes well inside the rs budget
        for b in packet.iter_mut().skip(8).take(8) {
            *b ^= 0xff;
        }

        let mut msg_rx = vec![0u8; n];
        let result = p.decode(&packet, &mut msg_rx);

        if cfg!(feature = "liquid-quirks") {
            match result {
                Err(_) => {}
                Ok(crc_pass) => assert!(
                    !crc_pass || msg_rx != msg_tx,
                    "depth 4 should not survive an 8-byte burst"
                ),
            }
        } else {
            let crc_pass = result.expect("8-byte burst should be correctable at depth 1");
            assert!(crc_pass, "crc should pass after correction");
            assert_eq!(msg_tx, msg_rx);
        }
    }

    #[test]
    fn test_packetizer_uncorrectable_reports_crc_failure() {
        let n = 64;
        let crc = CrcScheme::Crc32;

        for fec0 in [
            FecScheme::RsM8,
            FecScheme::Hamming74,
            FecScheme::Golay2412,
            FecScheme::None,
        ] {
            let pkt_len = packetizer_compute_enc_msg_len(n, crc, fec0, FecScheme::None);
            let mut p = Packetizer::new(n, crc, fec0, FecScheme::None).unwrap();

            let msg_tx: Vec<u8> = (0..n).map(|i| ((i * 7 + 3) % 256) as u8).collect();
            let mut packet = vec![0u8; pkt_len];
            p.encode(&msg_tx, &mut packet).unwrap();

            // corrupt a third of the packet: far beyond any of these schemes
            for (i, b) in packet.iter_mut().enumerate() {
                if i % 3 == 0 {
                    *b ^= 0xff;
                }
            }

            let mut msg_rx = vec![0xAAu8; n];
            let crc_pass = p
                .decode(&packet, &mut msg_rx)
                .unwrap_or_else(|e| panic!("{fec0:?}: decode returned Err({e})"));

            assert!(!crc_pass, "{fec0:?}: crc should fail on a mangled packet");

            // and the payload buffer was actually written, not left untouched
            assert!(
                msg_rx.iter().any(|&b| b != 0xAA),
                "{fec0:?}: decode left the output buffer untouched"
            );
        }
    }
}
