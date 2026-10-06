//! Pilot injection (transmit side).

use crate::error::{Error, Result};
use crate::math::nextpow2;
use crate::sequence::MaximalLengthSequence;
use num_complex::Complex32;
use std::f32::consts::PI;

/// Returns the number of pilots inserted into a payload.
///
/// Each pilot is followed by up to `pilot_spacing - 1` payload symbols, so the
/// count is `payload_len / (pilot_spacing - 1)` rounded up. A valid
/// [`PilotGenerator`] needs a nonempty payload and a spacing of at least two.
/// Outside that range this returns zero.
#[doc(alias = "qpilot_num_pilots")]
pub fn pilot_count(payload_len: usize, pilot_spacing: usize) -> usize {
    if payload_len == 0 || pilot_spacing < 2 {
        return 0;
    }
    let quot = payload_len / (pilot_spacing - 1);
    let rem = payload_len % (pilot_spacing - 1);
    quot + if rem != 0 { 1 } else { 0 }
}

/// Returns the frame length in symbols after pilot insertion.
///
/// The frame holds the payload plus [`pilot_count`] pilots. For an empty
/// payload or a spacing below two, no pilots are counted and this returns
/// `payload_len`.
///
/// # Panics
///
/// Panics when the sum overflows `usize` and overflow checks are enabled.
/// Without overflow checks, the count wraps.
#[doc(alias = "qpilot_frame_len")]
pub fn pilot_frame_len(payload_len: usize, pilot_spacing: usize) -> usize {
    payload_len + pilot_count(payload_len, pilot_spacing)
}

/// Generate the deterministic pilot symbol sequence shared by the pilot
/// generator and synchronizer.
pub(super) fn generate_pilots(num_pilots: usize) -> Result<Vec<Complex32>> {
    let m = nextpow2(num_pilots as u32)?.max(2);

    let mut pilots = Vec::with_capacity(num_pilots);
    let mut seq = MaximalLengthSequence::from_degree(m)?;
    for _ in 0..num_pilots {
        // generate symbol
        let s = seq.generate_symbol(2);

        // save modulated symbol (QPSK, offset by pi/4)
        let theta = (2.0 * PI * s as f32 / 4.0) + PI / 4.0;
        pilots.push(Complex32::from_polar(1.0, theta));
    }
    Ok(pilots)
}

/// Inserts a known pilot sequence into a symbol payload.
///
/// A pilot occupies every `pilot_spacing`th frame position, starting with the
/// first, and payload symbols fill the positions between in order. The pilots
/// are unit-magnitude QPSK symbols from a fixed pseudo-random sequence that
/// depends only on the pilot count. A [`PilotSynchronizer`] built with the
/// same payload length and spacing uses them to recover the payload.
///
/// [`PilotSynchronizer`]: crate::framing::PilotSynchronizer
#[doc(alias = "qpilotgen")]
#[derive(Clone)]
pub struct PilotGenerator {
    payload_len: usize,     // number of symbols in payload
    pilot_spacing: usize,   // spacing between pilot symbols
    num_pilots: usize,      // total number of pilot symbols
    frame_len: usize,       // total number of frame symbols
    pilots: Vec<Complex32>, // pilot sequence
}

impl PilotGenerator {
    /// Creates a pilot generator for `payload_len` symbols with a pilot every
    /// `pilot_spacing` frame symbols.
    ///
    /// # Errors
    ///
    /// Returns an error when `payload_len` is zero, `pilot_spacing` is less
    /// than two, or the frame needs more than about `2^31` pilots.
    ///
    /// # Panics
    ///
    /// Panics when the frame length overflows `usize` and overflow checks are
    /// enabled. Without overflow checks, the length wraps.
    pub fn new(payload_len: usize, pilot_spacing: usize) -> Result<Self> {
        // validate input
        if payload_len == 0 {
            return Err(Error::Config("payload length must be at least 1 symbol".into()));
        }
        if pilot_spacing < 2 {
            return Err(Error::Config("pilot spacing must be at least 2 symbols".into()));
        }

        // derived values
        let num_pilots = pilot_count(payload_len, pilot_spacing);
        let frame_len = payload_len + num_pilots;

        // generate pilot sequence
        let pilots = generate_pilots(num_pilots)?;

        Ok(Self { payload_len, pilot_spacing, num_pilots, frame_len, pilots })
    }

    /// Returns the frame length in symbols.
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Inserts pilots into `payload` and writes the complete frame.
    ///
    /// `payload` must hold exactly the payload length given at construction,
    /// and `frame` exactly [`Self::frame_len`] symbols. Every frame gets the
    /// same pilots.
    ///
    /// # Errors
    ///
    /// Returns an error when either slice has the wrong length. `frame` is
    /// unchanged on error.
    pub fn execute(&self, payload: &[Complex32], frame: &mut [Complex32]) -> Result<()> {
        if payload.len() != self.payload_len {
            return Err(Error::Config("unexpected payload length".into()));
        }
        if frame.len() != self.frame_len {
            return Err(Error::Config("unexpected frame length".into()));
        }

        let mut n = 0; // payload index
        let mut p = 0; // pilot index
        for (i, out) in frame.iter_mut().enumerate() {
            if i % self.pilot_spacing == 0 {
                *out = self.pilots[p];
                p += 1;
            } else {
                *out = payload[n];
                n += 1;
            }
        }

        // internal-consistency checks (mirror liquid); slice-length validation
        // above already guarantees these, but keep them as a guard
        debug_assert_eq!(n, self.payload_len);
        debug_assert_eq!(p, self.num_pilots);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pilot_count_contract() {
        // degenerate cases
        assert_eq!(pilot_count(0, 4), 0);
        assert_eq!(pilot_count(16, 1), 0);

        // ceil(payload / (spacing - 1))
        assert_eq!(pilot_count(16, 5), 4); // 16 / 4 = 4 exactly
        assert_eq!(pilot_count(17, 5), 5); // 17 / 4 -> 5 (remainder)
        assert_eq!(pilot_count(1, 2), 1); // 1 / 1 = 1

        // frame_len = payload + pilots
        assert_eq!(pilot_frame_len(16, 5), 20);
        assert_eq!(pilot_frame_len(1, 2), 2);
    }

    #[test]
    fn pilot_placement_and_payload_roundtrip() {
        let payload_len = 40;
        let pilot_spacing = 5;
        let gen = PilotGenerator::new(payload_len, pilot_spacing).unwrap();

        let n_pilots = pilot_count(payload_len, pilot_spacing);
        assert_eq!(gen.frame_len(), payload_len + n_pilots);

        let payload: Vec<Complex32> = (0..payload_len).map(|i| Complex32::new(i as f32 + 1.0, -(i as f32))).collect();
        let mut frame = vec![Complex32::new(0.0, 0.0); gen.frame_len()];
        gen.execute(&payload, &mut frame).unwrap();

        let mut n = 0;
        for (i, &f) in frame.iter().enumerate() {
            if i % pilot_spacing == 0 {
                assert!((f.norm() - 1.0).abs() < 1e-6, "pilot at {} not unit magnitude", i);
            } else {
                assert_eq!(f, payload[n]);
                n += 1;
            }
        }
        assert_eq!(n, payload_len);
    }

    #[test]
    fn pilots_are_deterministic() {
        // two generators with the same parameters must produce identical pilots
        let g0 = PilotGenerator::new(40, 5).unwrap();
        let g1 = PilotGenerator::new(40, 5).unwrap();
        assert_eq!(g0.pilots, g1.pilots);
    }

    #[test]
    fn single_pilot_config_is_valid() {
        let gen = PilotGenerator::new(1, 2).unwrap();
        assert_eq!(gen.frame_len(), 2);

        let payload = [Complex32::new(0.5, -0.5)];
        let mut frame = [Complex32::new(0.0, 0.0); 2];
        gen.execute(&payload, &mut frame).unwrap();
        assert!((frame[0].norm() - 1.0).abs() < 1e-6); // pilot
        assert_eq!(frame[1], payload[0]); // payload
    }

    #[test]
    fn execute_rejects_wrong_lengths() {
        let gen = PilotGenerator::new(40, 5).unwrap();
        let mut frame = vec![Complex32::new(0.0, 0.0); gen.frame_len()];
        // wrong payload length
        assert!(gen.execute(&[Complex32::new(0.0, 0.0); 39], &mut frame).is_err());
    }

    #[test]
    #[test_macro::autotest_annotate(autotest_qpilotgen_config)]
    fn qpilotgen_config() {
        assert!(PilotGenerator::new(0, 100).is_err(), "invalid payload length");
        assert!(PilotGenerator::new(512, 0).is_err(), "invalid pilot spacing");
        // a valid configuration still constructs
        assert!(PilotGenerator::new(512, 16).is_ok());
    }
}
