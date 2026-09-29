use crate::buffer::WindowedDelay;
use crate::error::{Error, Result};
use crate::filter::{FirFilter, FirHilbertFilter};
use crate::nco::{Nco, NcoBackend};
use num_complex::Complex32;

/// Sidebands sent by an [`AmplitudeModem`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AmpmodemType {
    /// Double sideband.
    Dsb,
    /// Upper sideband only.
    Usb,
    /// Lower sideband only.
    Lsb,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AmpmodemDemodType {
    DsbPllCostas,
    DsbPllCarrier,
    DemodSsb,
    DemodSsbPllCarrier,
}

#[derive(Debug, Clone)]
#[doc(alias = "Ampmodem")]
/// Amplitude modulator and demodulator for a real message.
///
/// The modulated signal is complex baseband. It is the message times the
/// modulation index, plus a carrier of amplitude one unless the carrier is
/// suppressed. The USB and LSB modes send the message through a Hilbert
/// filter first, so only one sideband remains. Demodulation divides by the
/// modulation index, so a round trip returns the message at its original
/// level, delayed by [`Self::delay_mod`] plus [`Self::delay_demod`].
///
/// Carrier recovery depends on the mode. Double sideband uses a
/// phase-locked loop: a Costas loop when the carrier is suppressed, or a loop
/// that tracks the carrier otherwise. A Costas loop can settle half a turn
/// off, which negates the recovered message. Single sideband with a carrier
/// tracks the carrier with a loop. Suppressed-carrier USB and LSB have no
/// carrier recovery, so their input must have no frequency or phase offset.
///
/// Demodulation loses low message frequencies in some modes. With a carrier,
/// the recovered message passes through a DC blocker that also attenuates
/// nearby frequencies. Suppressed-carrier USB and LSB lose the low end in
/// the Hilbert filter. Suppressed-carrier DSB has neither filter. In cycles
/// per sample, keep the message above about `0.02` in the carrier modes and
/// `0.01` in suppressed-carrier USB and LSB. With a carrier, the response
/// above that point still ripples by about 1.5 dB. These are not exact band
/// edges, so check the response in the band you need.
///
/// In the USB and LSB modes, modulation and demodulation share one Hilbert
/// filter. Use separate objects for the two directions.
pub struct AmplitudeModem {
    mod_index: f32,                     // modulation index
    mod_type: AmpmodemType,             // modulation type (e.g. DSB)
    suppressed_carrier: bool,           // suppressed carrier flag
    m: usize,                           // filter semi-length for all objects
    mixer: Nco,                         // mixer and phase-locked loop
    dcblock: FirFilter<f32>,            // carrier suppression filter
    hilbert: FirHilbertFilter,          // hilbert transform (single side-band)
    lowpass: FirFilter<Complex32, f32>, // low-pass filter for SSB PLL
    delay: WindowedDelay<Complex32>,    // delay buffer to align to low-pass filter delay
    demod_type: AmpmodemDemodType,
    phase_error: f32,
}

impl AmplitudeModem {
    /// Creates an amplitude modem.
    ///
    /// `modulation_index` scales the message before the carrier is added.
    /// `suppressed_carrier` leaves the carrier out. The carrier-recovery loop
    /// bandwidth starts at 0.001.
    ///
    /// # Errors
    ///
    /// Returns an error when `modulation_index` is zero or negative.
    pub fn new(modulation_index: f32, modulation_type: AmpmodemType, suppressed_carrier: bool) -> Result<Self> {
        // Validate input
        if modulation_index <= 0.0 {
            return Err(Error::Config(format!("modulation index {modulation_index:.4e} must be greater than 0")));
        }

        let m = 25;

        let mut nco = Nco::new(NcoBackend::LookupTable);
        nco.pll_set_bandwidth(0.001);

        let demod_type = match (modulation_type, suppressed_carrier) {
            (AmpmodemType::Dsb, true) => AmpmodemDemodType::DsbPllCostas,
            (AmpmodemType::Dsb, false) => AmpmodemDemodType::DsbPllCarrier,
            (AmpmodemType::Usb | AmpmodemType::Lsb, true) => AmpmodemDemodType::DemodSsb,
            (AmpmodemType::Usb | AmpmodemType::Lsb, false) => AmpmodemDemodType::DemodSsbPllCarrier,
        };

        let mut q = Self {
            mod_index: modulation_index,
            mod_type: modulation_type,
            suppressed_carrier,
            m,
            mixer: nco,
            dcblock: FirFilter::new_dc_blocker(m, 20.0)?,
            hilbert: FirHilbertFilter::new(m, 60.0)?,
            lowpass: FirFilter::new_kaiser(2 * m + 1, 0.01, 40.0, 0.0)?,
            delay: WindowedDelay::new(m)?,
            demod_type,
            phase_error: 0.0,
        };

        q.reset();
        Ok(q)
    }

    /// Clears the filter history and the carrier-recovery phase and
    /// frequency.
    ///
    /// The loop bandwidth is unchanged, and the value returned by
    /// [`Self::phase_error`] is not cleared.
    pub fn reset(&mut self) {
        self.mixer.reset();
        self.dcblock.reset();
        self.hilbert.reset();
        self.lowpass.reset();
        self.delay.reset();
    }

    /// Returns the delay added by modulation, in samples.
    ///
    /// This is 0 for DSB and 50 for USB and LSB.
    pub fn delay_mod(&self) -> usize {
        match self.mod_type {
            AmpmodemType::Dsb => 0,
            AmpmodemType::Usb | AmpmodemType::Lsb => 2 * self.m,
        }
    }

    /// Returns the delay added by demodulation, in samples.
    ///
    /// This is 0 for suppressed-carrier DSB, 50 for DSB with a carrier and
    /// for suppressed-carrier USB and LSB, and 100 for USB and LSB with a
    /// carrier.
    pub fn delay_demod(&self) -> usize {
        match self.mod_type {
            AmpmodemType::Dsb => {
                if self.suppressed_carrier {
                    0
                } else {
                    2 * self.m
                }
            }
            AmpmodemType::Usb | AmpmodemType::Lsb => {
                if self.suppressed_carrier {
                    2 * self.m
                } else {
                    4 * self.m
                }
            }
        }
    }

    /// Returns the phase error that the carrier-recovery loop last used.
    ///
    /// For DSB with a carrier it is in radians. In the other loop modes it is
    /// the imaginary part of a mixed-down sample, so it also scales with the
    /// signal amplitude. Suppressed-carrier USB and LSB have no loop, and the
    /// value stays at zero.
    pub fn phase_error(&self) -> f32 {
        self.phase_error
    }

    /// Sets the carrier-recovery loop gain.
    ///
    /// This is the same dimensionless gain as
    /// [`Nco::pll_set_bandwidth`](crate::nco::Nco::pll_set_bandwidth). Larger
    /// values lock faster. Suppressed-carrier USB and LSB have no loop, so the
    /// value has no effect in those modes.
    ///
    /// # Panics
    ///
    /// Panics when `bandwidth` is negative.
    pub fn set_pll_bandwidth(&mut self, bandwidth: f32) {
        self.mixer.pll_set_bandwidth(bandwidth);
    }

    /// Modulates one message sample into one complex sample.
    ///
    /// This never returns an error.
    pub fn modulate(&mut self, input: f32) -> Result<Complex32> {
        let x_hat = match self.mod_type {
            AmpmodemType::Dsb => Complex32::new(input, 0.0),
            AmpmodemType::Usb | AmpmodemType::Lsb => {
                let mut x_hat = self.hilbert.r2c_execute(input)?;
                if self.mod_type == AmpmodemType::Lsb {
                    x_hat = x_hat.conj();
                }
                x_hat
            }
        };

        Ok(x_hat * self.mod_index
            + if self.suppressed_carrier { Complex32::new(0.0, 0.0) } else { Complex32::new(1.0, 0.0) })
    }

    /// Modulates each sample of `input` into `output`.
    ///
    /// The results go to the first `input.len()` samples of `output`, and the
    /// rest of `output` is left unchanged. Returns `input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`. The modem and
    /// `output` are unchanged on error.
    pub fn modulate_block(&mut self, input: &[f32], output: &mut [Complex32]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Range("output length must be at least the input length".into()));
        }

        for (input_sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = self.modulate(*input_sample)?;
        }
        Ok(input.len())
    }

    /// Demodulates one complex sample into one message sample.
    ///
    /// This never returns an error.
    pub fn demodulate(&mut self, input: Complex32) -> Result<f32> {
        match self.demod_type {
            AmpmodemDemodType::DsbPllCarrier => self.demod_dsb_pll_carrier(input),
            AmpmodemDemodType::DsbPllCostas => self.demod_dsb_pll_costas(input),
            AmpmodemDemodType::DemodSsb => self.demod_ssb(input),
            AmpmodemDemodType::DemodSsbPllCarrier => self.demod_ssb_pll_carrier(input),
        }
    }

    /// Demodulates each sample of `input` into `output`.
    ///
    /// The results go to the first `input.len()` samples of `output`, and the
    /// rest of `output` is left unchanged. Returns `input.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error when `output` is shorter than `input`. The modem and
    /// `output` are unchanged on error.
    pub fn demodulate_block(&mut self, input: &[Complex32], output: &mut [f32]) -> Result<usize> {
        if output.len() < input.len() {
            return Err(Error::Range("output length must be at least the input length".into()));
        }

        for (input_sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = self.demodulate(*input_sample)?;
        }
        Ok(input.len())
    }

    fn demod_dsb_pll_carrier(&mut self, y: Complex32) -> Result<f32> {
        // split signal into two branches:
        //   0. low-pass filter for carrier recovery and
        //   1. delay to align signal output
        self.lowpass.push(y);
        let y0 = self.lowpass.execute();
        self.delay.push(y);
        let y1 = self.delay.read();

        // mix each signal down
        let v0 = self.mixer.mix_down(y0);
        let v1 = self.mixer.mix_down(y1);

        // compute phase error
        self.phase_error = v0.arg();

        // adjust nco, pll objects
        self.mixer.pll_step(self.phase_error);

        // step nco
        self.mixer.step();

        // keep in-phase component
        let m = v1.re / self.mod_index;

        // apply DC block, writing directly to output
        self.dcblock.push(m);
        Ok(self.dcblock.execute())
    }

    fn demod_dsb_pll_costas(&mut self, y: Complex32) -> Result<f32> {
        // mix down
        let v = self.mixer.mix_down(y);

        // compute phase error
        self.phase_error = v.im * if v.re > 0.0 { 1.0 } else { -1.0 };

        // adjust nco, pll objects
        self.mixer.pll_step(self.phase_error);

        // step nco
        self.mixer.step();

        // keep in-phase component
        Ok(v.re / self.mod_index)
    }

    fn demod_ssb_pll_carrier(&mut self, y: Complex32) -> Result<f32> {
        // split signal into two branches:
        //   0. low-pass filter for carrier recovery and
        //   1. delay to align signal output
        self.lowpass.push(y);
        let y0 = self.lowpass.execute();
        self.delay.push(y);
        let y1 = self.delay.read();

        // mix each signal down
        let v0 = self.mixer.mix_down(y0);
        let v1 = self.mixer.mix_down(y1);

        // compute phase error
        self.phase_error = v0.im;

        // adjust nco, pll objects
        self.mixer.pll_step(self.phase_error);

        // step nco
        self.mixer.step();

        let (m_lsb, m_usb) = self.hilbert.c2r_execute(v1)?;

        // recover message
        let m = 0.5 * if self.mod_type == AmpmodemType::Usb { m_usb } else { m_lsb } / self.mod_index;

        // apply DC block, writing directly to output
        self.dcblock.push(m);
        Ok(self.dcblock.execute())
    }

    fn demod_ssb(&mut self, y: Complex32) -> Result<f32> {
        // apply hilbert transform and retrieve both upper and lower side-bands
        let (m_lsb, m_usb) = self.hilbert.c2r_execute(y)?;

        // recover message
        let m = 0.5 * if self.mod_type == AmpmodemType::Usb { m_usb } else { m_lsb } / self.mod_index;
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility::test_helpers::validate_block_execute;
    use test_macro::autotest_annotate;

    // Help function to keep code base small
    fn ampmodem_test_harness(
        mod_index: f32,
        mod_type: AmpmodemType,
        suppressed_carrier: bool,
        dphi: f32,
        mut phi: f32,
    ) -> Result<()> {
        use crate::buffer::WindowedDelay;
        use crate::random::randnf;
        use num_complex::Complex;
        use std::f32::consts::{FRAC_1_SQRT_2, PI};

        // options
        let snr_db = 40.0; // signal-to-noise ratio (set very high for testing)

        // derived values
        let nstd = 10.0f32.powf(-snr_db / 20.0);

        // create mod/demod objects
        let mut mod_ = AmplitudeModem::new(mod_index, mod_type, suppressed_carrier)?;
        let mut demod = AmplitudeModem::new(mod_index, mod_type, suppressed_carrier)?;

        // compute end-to-end delay
        let delay = mod_.delay_mod() + demod.delay_demod();
        let mut message_delay = WindowedDelay::new(delay)?;

        // run trials
        let mut i = 0;
        let skip = 2400; // wait for PLL and filters to settle
        let mut num_samples_compare = 0; // number of samples compared
        let mut rmse_0 = 0.0;
        let mut rmse_1 = 0.0; // RMS error in phase and 180 out of phase
        let f0 = 1.0 / (1031.0f32).sqrt();
        let f1 = 1.0 / (1723.0f32).sqrt();

        while num_samples_compare < 8000 {
            // generate original message signal
            let msg_in = 0.6 * (2.0 * PI * f0 * i as f32).cos() + 0.4 * (2.0 * PI * f1 * i as f32).cos();
            message_delay.push(msg_in);

            // modulate
            let x = mod_.modulate(msg_in)?;

            // add channel impairments
            let y = x * Complex::new(phi.cos(), phi.sin()) + nstd * Complex::new(randnf(), randnf()) * FRAC_1_SQRT_2;

            // update phase
            phi += dphi;
            while phi > PI {
                phi -= 2.0 * PI;
            }
            while phi < -PI {
                phi += 2.0 * PI;
            }

            // demodulate signal
            let msg_out = demod.demodulate(y)?;

            // compute error
            let msg_in = message_delay.read();

            if i >= skip {
                let e0 = msg_in - msg_out;
                let e1 = msg_in + msg_out;
                rmse_0 += e0 * e0;
                rmse_1 += e1 * e1;
                num_samples_compare += 1;
            }
            i += 1;
        }

        // finally, check if test passed based on modulation type; for
        // double side-band suppressed carrier, we can have a 180 degree phase offset
        rmse_0 = 10.0 * (rmse_0 / num_samples_compare as f32).log10(); // in-phase
        rmse_1 = 10.0 * (rmse_1 / num_samples_compare as f32).log10(); // 180-degree out of phase

        let rmse = if mod_type == AmpmodemType::Dsb && suppressed_carrier { rmse_0.min(rmse_1) } else { rmse_0 };

        assert!(rmse < -18.0);
        Ok(())
    }

    #[test]
    #[autotest_annotate(autotest_ampmodem_dsb_carrier_on)]
    fn test_ampmodem_dsb_carrier_on() -> Result<()> {
        ampmodem_test_harness(0.8, AmpmodemType::Dsb, false, 0.02, 0.0)
    }

    #[test]
    #[autotest_annotate(autotest_ampmodem_usb_carrier_on)]
    fn test_ampmodem_usb_carrier_on() -> Result<()> {
        ampmodem_test_harness(0.8, AmpmodemType::Usb, false, 0.02, 0.0)
    }

    #[test]
    #[autotest_annotate(autotest_ampmodem_lsb_carrier_on)]
    fn test_ampmodem_lsb_carrier_on() -> Result<()> {
        ampmodem_test_harness(0.8, AmpmodemType::Lsb, false, 0.02, 0.0)
    }

    #[test]
    #[autotest_annotate(autotest_ampmodem_dsb_carrier_off)]
    fn test_ampmodem_dsb_carrier_off() -> Result<()> {
        ampmodem_test_harness(0.8, AmpmodemType::Dsb, true, 0.02, 0.0)
    }

    #[test]
    #[autotest_annotate(autotest_ampmodem_usb_carrier_off)]
    fn test_ampmodem_usb_carrier_off() -> Result<()> {
        ampmodem_test_harness(0.8, AmpmodemType::Usb, true, 0.00, 0.0)
    }

    #[test]
    #[autotest_annotate(autotest_ampmodem_lsb_carrier_off)]
    fn test_ampmodem_lsb_carrier_off() -> Result<()> {
        ampmodem_test_harness(0.8, AmpmodemType::Lsb, true, 0.00, 0.0)
    }

    #[test]
    fn test_ampmodem_block_contracts() {
        let modem = AmplitudeModem::new(0.5, AmpmodemType::Dsb, true).unwrap();
        validate_block_execute(&modem, &[0.25, -0.5, 1.0], Complex32::new(-9.0, -9.0), |q, x, y| {
            q.modulate_block(x, y)
        });

        let modem = AmplitudeModem::new(0.5, AmpmodemType::Dsb, true).unwrap();
        let input = [Complex32::new(1.0, -0.5), Complex32::new(0.25, 0.75), Complex32::new(-1.0, 0.0)];
        validate_block_execute(&modem, &input, -9.0, |q, x, y| q.demodulate_block(x, y));
    }
}
