//! Symbol recovery with interleaved pilots (receive side).

use super::pilot_generator::{generate_pilots, pilot_count};
use crate::error::{Error, Result};
use crate::fft::{Direction, Fft};
use crate::math::nextpow2;
use num_complex::{Complex32, ComplexFloat};
use std::f32::consts::PI;

/// Recovers a payload from a frame with interleaved pilots.
///
/// This is the receive side of [`PilotGenerator`]. Each frame is handled on
/// its own. The synchronizer compares the received pilots with the known
/// sequence to estimate carrier frequency, carrier phase, and gain. It then
/// corrects every frame symbol and extracts the payload. The frequency comes
/// from an interpolated FFT peak of the pilot errors, and the phase and gain
/// from their average after removing that frequency.
///
/// Pilots sample the carrier once every `pilot_spacing` symbols, so the
/// frequency estimate is unambiguous only for offsets strictly between
/// `-pi / pilot_spacing` and `pi / pilot_spacing` radians per frame symbol.
/// A larger offset aliases to another offset in that range, and
/// [`Self::execute`] still succeeds. Aliases differ by `2 * pi / pilot_spacing`
/// and so have the same phase at every pilot. A low [`Self::evm`] therefore
/// cannot rule out an aliased, corrupted payload. The synchronizer does not
/// check payload integrity.
///
/// [`PilotGenerator`]: crate::framing::PilotGenerator
#[doc(alias = "qpilotsync")]
#[derive(Clone)]
pub struct PilotSynchronizer {
    payload_len: usize,     // number of symbols in payload
    pilot_spacing: usize,   // spacing between pilot symbols
    num_pilots: usize,      // total number of pilot symbols
    frame_len: usize,       // total number of frame symbols
    pilots: Vec<Complex32>, // pilot sequence

    nfft: usize,              // FFT size
    buf_time: Vec<Complex32>, // FFT time buffer
    buf_freq: Vec<Complex32>, // FFT freq buffer
    fft: Fft<f32>,            // transform object

    dphi_hat: f32, // carrier frequency offset estimate
    phi_hat: f32,  // carrier phase offset estimate
    g_hat: f32,    // gain correction estimate
    evm_hat: f32,  // error-vector magnitude estimate (from pilots)
}

impl PilotSynchronizer {
    /// Creates a pilot synchronizer.
    ///
    /// `payload_len` and `pilot_spacing` must match the transmitting
    /// [`PilotGenerator`](crate::framing::PilotGenerator). A payload shorter
    /// than `pilot_spacing` gives a frame with a single pilot, which
    /// [`Self::execute`] cannot use.
    ///
    /// # Errors
    ///
    /// Returns an error when `payload_len` is zero, `pilot_spacing` is less
    /// than two, or the frame needs more than about `2^31` pilots.
    ///
    /// # Panics
    ///
    /// Panics when a derived frame or FFT length overflows `usize` and
    /// overflow checks are enabled. Without overflow checks, the lengths wrap
    /// and are wrong.
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

        // generate pilot sequence (shared with the generator)
        let pilots = generate_pilots(num_pilots)?;

        // compute fft size and create transform object
        let nfft = 1usize << nextpow2((num_pilots + (num_pilots >> 1)) as u32)?;
        let buf_time = vec![Complex32::new(0.0, 0.0); nfft];
        let buf_freq = vec![Complex32::new(0.0, 0.0); nfft];
        let fft = Fft::new(nfft, Direction::Forward);

        let mut q = Self {
            payload_len,
            pilot_spacing,
            num_pilots,
            frame_len,
            pilots,
            nfft,
            buf_time,
            buf_freq,
            fft,
            dphi_hat: 0.0,
            phi_hat: 0.0,
            g_hat: 1.0,
            evm_hat: 0.0,
        };

        q.reset();
        Ok(q)
    }

    /// Returns the carrier and gain estimates to zero offset and unit gain.
    ///
    /// [`Self::evm`] keeps its value until the next frame. Resetting between
    /// frames is not required, because each frame is estimated independently.
    pub fn reset(&mut self) {
        // clear FFT input buffer
        for x in self.buf_time.iter_mut() {
            *x = Complex32::new(0.0, 0.0);
        }

        // reset estimates
        self.dphi_hat = 0.0;
        self.phi_hat = 0.0;
        self.g_hat = 1.0;
    }

    /// Returns the frame length in symbols.
    pub fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Returns the carrier frequency offset of the last frame, in radians per
    /// frame symbol.
    pub fn dphi(&self) -> f32 {
        self.dphi_hat
    }

    /// Returns the carrier phase of the last frame at its first symbol, in
    /// radians.
    pub fn phi(&self) -> f32 {
        self.phi_hat
    }

    /// Returns the channel gain of the last frame, as a linear amplitude
    /// factor.
    pub fn gain(&self) -> f32 {
        self.g_hat
    }

    /// Returns the error vector magnitude of the corrected pilots in the last
    /// frame, in dB.
    ///
    /// This is the mean squared distance between each corrected pilot and its
    /// known value. It starts at zero before the first frame.
    pub fn evm(&self) -> f32 {
        self.evm_hat
    }

    /// Estimates offsets from the pilots in `frame` and writes the corrected
    /// payload.
    ///
    /// `frame` must hold exactly [`Self::frame_len`] symbols, and `payload`
    /// exactly the payload length given at construction. The estimate getters
    /// report this frame after the call. An offset outside the range given on
    /// the type yields a wrongly corrected payload without an error.
    ///
    /// The frequency estimate needs at least two pilots. With a single pilot,
    /// or when every received pilot is zero, the estimates and the payload
    /// become NaN and the call still succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error when either slice has the wrong length. The estimates
    /// and `payload` are unchanged on error.
    pub fn execute(&mut self, frame: &[Complex32], payload: &mut [Complex32]) -> Result<()> {
        if frame.len() != self.frame_len {
            return Err(Error::Config("unexpected frame length".into()));
        }
        if payload.len() != self.payload_len {
            return Err(Error::Config("unexpected payload length".into()));
        }

        // extract pilots and de-rotate with known sequence
        for i in 0..self.num_pilots {
            self.buf_time[i] = frame[i * self.pilot_spacing] * self.pilots[i].conj();
        }

        // compute frequency offset by computing transform and finding peak
        self.fft.run(&self.buf_time, &mut self.buf_freq);
        let mut i0 = 0;
        let mut y0 = 0.0f32;
        for i in 0..self.nfft {
            let mag = self.buf_freq[i].abs();
            if i == 0 || mag > y0 {
                i0 = i;
                y0 = mag;
            }
        }

        // interpolate and recover frequency
        let ineg = (i0 + self.nfft - 1) % self.nfft;
        let ipos = (i0 + 1) % self.nfft;
        let ypos = self.buf_freq[ipos].abs();
        let yneg = self.buf_freq[ineg].abs();
        let a = 0.5 * (ypos + yneg) - y0;
        let b = 0.5 * (ypos - yneg);
        let idx = -b / (2.0 * a);
        let index = i0 as f32 + idx;
        self.dphi_hat = (if i0 > self.nfft / 2 { index - self.nfft as f32 } else { index }) * 2.0 * PI
            / (self.nfft * self.pilot_spacing) as f32;

        // estimate carrier phase offset
        // METHOD 2: compute metric by de-rotating pilots and measuring resulting
        // phase (more accurate than interpolating the FFT phase directly)
        let mut metric = Complex32::new(0.0, 0.0);
        for i in 0..self.num_pilots {
            let phase = -self.dphi_hat * i as f32 * self.pilot_spacing as f32;
            metric += self.buf_time[i] * Complex32::from_polar(1.0, phase);
        }
        self.phi_hat = metric.arg();
        self.g_hat = metric.abs() / self.num_pilots as f32;

        // gain correction
        let g = 1.0 / self.g_hat;

        // recover frame symbols
        self.evm_hat = 0.0;
        let mut n = 0; // payload index
        let mut p = 0; // pilot index
        for (i, &fi) in frame.iter().enumerate() {
            let v = g * fi * Complex32::from_polar(1.0, -(self.dphi_hat * i as f32 + self.phi_hat));
            if i % self.pilot_spacing == 0 {
                // pilot symbol
                let e = self.pilots[p] - v;
                self.evm_hat += (e * e.conj()).re();
                p += 1;
            } else {
                // data symbol
                payload[n] = v;
                n += 1;
            }
        }
        self.evm_hat = 10.0 * (self.evm_hat / self.num_pilots as f32).log10();

        debug_assert_eq!(n, self.payload_len);
        debug_assert_eq!(p, self.num_pilots);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framing::PilotGenerator;
    use crate::modem::{Modem, ModulationScheme};
    use crate::random::randnf;
    use crate::utility::count_bit_errors;
    use approx::assert_abs_diff_eq;
    use std::f32::consts::FRAC_1_SQRT_2;
    use test_macro::autotest_annotate;

    // test simple recovery of frame in noise
    #[allow(clippy::too_many_arguments)]
    fn qpilotsync_test(
        ms: ModulationScheme,
        payload_len: usize,
        pilot_spacing: usize,
        dphi: f32,
        phi: f32,
        gamma: f32,
        snr_db: f32,
    ) {
        // derived values
        let nstd = 10f32.powf(-snr_db / 20.0);

        // create pilot generator and synchronizer objects
        let pg = PilotGenerator::new(payload_len, pilot_spacing).unwrap();
        let mut ps = PilotSynchronizer::new(payload_len, pilot_spacing).unwrap();

        // get frame length
        let frame_len = pg.frame_len();
        assert_eq!(frame_len, ps.frame_len());

        // create modem objects for payload
        let mut modulator = Modem::new(ms).unwrap();
        let mut demod = Modem::new(ms).unwrap();

        // assemble payload symbols
        let mut payload_sym_tx = vec![0u32; payload_len];
        let mut payload_tx = vec![Complex32::new(0.0, 0.0); payload_len];
        for i in 0..payload_len {
            payload_sym_tx[i] = modulator.random_symbol();
            payload_tx[i] = modulator.modulate(payload_sym_tx[i]).unwrap();
        }

        // assemble frame
        let mut frame_tx = vec![Complex32::new(0.0, 0.0); frame_len];
        pg.execute(&payload_tx, &mut frame_tx).unwrap();

        // add channel impairments
        let mut frame_rx = vec![Complex32::new(0.0, 0.0); frame_len];
        for i in 0..frame_len {
            // add carrier offset
            let mut r = frame_tx[i] * Complex32::from_polar(1.0, dphi * i as f32 + phi);
            // add noise
            r += nstd * Complex32::new(randnf(), randnf()) * FRAC_1_SQRT_2;
            // apply channel gain
            r *= gamma;
            frame_rx[i] = r;
        }

        // receive frame
        let mut payload_rx = vec![Complex32::new(0.0, 0.0); payload_len];
        ps.execute(&frame_rx, &mut payload_rx).unwrap();

        // demodulate and count errors
        let mut bit_errors = 0;
        for i in 0..payload_len {
            let sym_demod = demod.demodulate(payload_rx[i]).unwrap();
            bit_errors += count_bit_errors(sym_demod, payload_sym_tx[i]);
        }

        // check to see that frame was recovered
        assert_abs_diff_eq!(ps.dphi(), dphi, epsilon = 0.010);
        assert_abs_diff_eq!(ps.phi(), phi, epsilon = 0.087); // ~5 degrees
        assert_abs_diff_eq!(ps.gain(), gamma, epsilon = 0.010);
        assert_eq!(bit_errors, 0);
    }

    #[test]
    #[autotest_annotate(autotest_qpilotsync_100_16)]
    fn qpilotsync_100_16() {
        qpilotsync_test(ModulationScheme::Qpsk, 100, 16, 0.07, 1.2, 0.7, 40.0);
    }

    #[test]
    #[autotest_annotate(autotest_qpilotsync_200_20)]
    fn qpilotsync_200_20() {
        qpilotsync_test(ModulationScheme::Qpsk, 200, 20, 0.07, 1.2, 0.7, 40.0);
    }

    #[test]
    #[autotest_annotate(autotest_qpilotsync_300_24)]
    fn qpilotsync_300_24() {
        qpilotsync_test(ModulationScheme::Qpsk, 300, 24, 0.07, 1.2, 0.7, 40.0);
    }

    #[test]
    #[autotest_annotate(autotest_qpilotsync_400_28)]
    fn qpilotsync_400_28() {
        qpilotsync_test(ModulationScheme::Qpsk, 400, 28, 0.07, 1.2, 0.7, 40.0);
    }

    #[test]
    #[autotest_annotate(autotest_qpilotsync_500_32)]
    fn qpilotsync_500_32() {
        qpilotsync_test(ModulationScheme::Qpsk, 500, 32, 0.07, 1.2, 0.7, 40.0);
    }

    #[test]
    #[autotest_annotate(autotest_qpilotsync_config)]
    fn qpilotsync_config() {
        // invalid configurations
        assert!(PilotSynchronizer::new(0, 100).is_err()); // invalid payload length
        assert!(PilotSynchronizer::new(512, 0).is_err()); // invalid pilot spacing

        // create proper object
        assert!(PilotSynchronizer::new(512, 16).is_ok());
    }
}
