use crate::error::{Error, Result};
use crate::nco::{Osc, OscScheme};
use crate::filter::{FirHilbertFilter, FirFilter};
use crate::buffer::WDelay;
use num_complex::Complex32;

// Modulation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AmpmodemType {
    Dsb,  // Double side-band
    Usb,  // Upper side-band
    Lsb,  // Lower side-band
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AmpmodemDemodType {
    DsbPllCostas,
    DsbPllCarrier,
    DemodSsb,
    DemodSsbPllCarrier,
}

#[derive(Debug, Clone)]
pub struct Ampmodem {
    mod_index: f32,              // modulation index
    mod_type: AmpmodemType,      // modulation type (e.g. DSB)
    suppressed_carrier: bool,    // suppressed carrier flag
    m: usize,                    // filter semi-length for all objects
    mixer: Osc,                  // mixer and phase-locked loop
    dcblock: FirFilter<f32>,     // carrier suppression filter
    hilbert: FirHilbertFilter,    // hilbert transform (single side-band)
    lowpass: FirFilter<Complex32, f32>,     // low-pass filter for SSB PLL
    delay: WDelay<Complex32>,          // delay buffer to align to low-pass filter delay
    demod_type: AmpmodemDemodType,
    phase_error: f32,
}

impl Ampmodem {
    pub fn new(mod_index: f32, mod_type: AmpmodemType, suppressed_carrier: bool) -> Result<Self> {
        // Validate input
        if mod_index <= 0.0 {
            return Err(Error::Config(format!(
                "modulation index {:.4e} must be greater than 0",
                mod_index
            )));
        }

        let m = 25;

        let mut nco = Osc::new(OscScheme::Nco);
        nco.pll_set_bandwidth(0.001);

        let demod_type = match (mod_type, suppressed_carrier) {
            (AmpmodemType::Dsb, true) => AmpmodemDemodType::DsbPllCostas,
            (AmpmodemType::Dsb, false) => AmpmodemDemodType::DsbPllCarrier,
            (AmpmodemType::Usb | AmpmodemType::Lsb, true) => AmpmodemDemodType::DemodSsb,
            (AmpmodemType::Usb | AmpmodemType::Lsb, false) => AmpmodemDemodType::DemodSsbPllCarrier,
        };

        let mut q = Self {
            mod_index,
            mod_type,
            suppressed_carrier,
            m,
            mixer: nco,
            dcblock: FirFilter::new_dc_blocker(m, 20.0)?,
            hilbert: FirHilbertFilter::new(m, 60.0)?,
            lowpass: FirFilter::new_kaiser(2 * m + 1, 0.01, 40.0, 0.0)?,
            delay: WDelay::create(m)?,
            demod_type,
            phase_error: 0.0,
        };

        q.reset();
        Ok(q)
    }

    pub fn reset(&mut self) {
        self.mixer.reset();
        self.dcblock.reset();
        self.hilbert.reset();
        self.lowpass.reset();
        self.delay.reset();
    }

    pub fn get_delay_mod(&self) -> usize {
        match self.mod_type {
            AmpmodemType::Dsb => 0,
            AmpmodemType::Usb | AmpmodemType::Lsb => 2 * self.m,
        }
    }

    pub fn get_delay_demod(&self) -> usize {
        match self.mod_type {
            AmpmodemType::Dsb => {
                if self.suppressed_carrier { 0 } else { 2 * self.m }
            },
            AmpmodemType::Usb | AmpmodemType::Lsb => {
                if self.suppressed_carrier { 2 * self.m } else { 4 * self.m }
            },
        }
    }

    pub fn get_phase_error(&self) -> f32 {
        self.phase_error
    }

    pub fn set_pll_bandwidth(&mut self, bandwidth: f32) {
        self.mixer.pll_set_bandwidth(bandwidth);
    }

    pub fn modulate(&mut self, x: f32) -> Result<Complex32> {
        let x_hat = match self.mod_type {
            AmpmodemType::Dsb => Complex32::new(x, 0.0),
            AmpmodemType::Usb | AmpmodemType::Lsb => {
                let mut x_hat = self.hilbert.r2c_execute(x)?;
                if self.mod_type == AmpmodemType::Lsb {
                    x_hat = x_hat.conj();
                }
                x_hat
            },
        };

        Ok(x_hat * self.mod_index + if self.suppressed_carrier { Complex32::new(0.0, 0.0) } else { Complex32::new(1.0, 0.0) })
    }

    pub fn modulate_block(&mut self, m: &[f32], s: &mut [Complex32]) -> Result<()> {
        if m.len() != s.len() {
            return Err(Error::Range(
                "input and output arrays must be same length".into()
            ));
        }

        for (x, y) in m.iter().zip(s.iter_mut()) {
            *y = self.modulate(*x)?;
        }
        Ok(())
    }

    pub fn demodulate(&mut self, y: Complex32) -> Result<f32> {
        match self.demod_type {
            AmpmodemDemodType::DsbPllCarrier => {
                self.demod_dsb_pll_carrier(y)
            },
            AmpmodemDemodType::DsbPllCostas => {
                self.demod_dsb_pll_costas(y)
            },
            AmpmodemDemodType::DemodSsb => {
                self.demod_ssb(y)
            },
            AmpmodemDemodType::DemodSsbPllCarrier => {
                self.demod_ssb_pll_carrier(y)
            },
        }
    }

    pub fn demodulate_block(&mut self, y: &[Complex32], x: &mut [f32]) -> Result<()> {
        if y.len() != x.len() {
            return Err(Error::Range(
                "input and output arrays must be same length".into()
            ));
        }

        for (y_val, x_val) in y.iter().zip(x.iter_mut()) {
            *x_val = self.demodulate(*y_val)?;
        }
        Ok(())
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
    use test_macro::autotest_annotate;

    // Help function to keep code base small
    fn ampmodem_test_harness(
        mod_index: f32,
        mod_type: AmpmodemType,
        suppressed_carrier: bool,
        dphi: f32,
        mut phi: f32,
    ) -> Result<()> {
        use std::f32::consts::{PI, FRAC_1_SQRT_2};
        use num_complex::Complex;
        use crate::buffer::WDelay;
        use crate::random::randnf;
        
        // options
        let snr_db = 40.0;    // signal-to-noise ratio (set very high for testing)

        // derived values
        let nstd = 10.0f32.powf(-snr_db / 20.0);

        // create mod/demod objects
        let mut mod_ = Ampmodem::new(mod_index, mod_type, suppressed_carrier)?;
        let mut demod = Ampmodem::new(mod_index, mod_type, suppressed_carrier)?;
        
        // compute end-to-end delay
        let delay = mod_.get_delay_mod() + demod.get_delay_demod();
        let mut message_delay = WDelay::create(delay)?;

        // run trials
        let mut i = 0;
        let skip = 2400; // wait for PLL and filters to settle
        let mut num_samples_compare = 0;   // number of samples compared
        let mut rmse_0 = 0.0;
        let mut rmse_1 = 0.0;           // RMS error in phase and 180 out of phase
        let f0 = 1.0 / (1031.0f32).sqrt();
        let f1 = 1.0 / (1723.0f32).sqrt();
        
        while num_samples_compare < 8000 {
            // generate original message signal
            let msg_in = 0.6 * (2.0 * PI * f0 * i as f32).cos() + 0.4 * (2.0 * PI * f1 * i as f32).cos();
            message_delay.push(msg_in);

            // modulate
            let x = mod_.modulate(msg_in)?;

            // add channel impairments
            let y = x * Complex::new(phi.cos(), phi.sin()) +
                nstd * Complex::new(randnf(), randnf()) * FRAC_1_SQRT_2;

            // update phase
            phi += dphi;
            while phi > PI { phi -= 2.0 * PI; }
            while phi < -PI { phi += 2.0 * PI; }

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
        rmse_0 = 10.0 * (rmse_0 / num_samples_compare as f32).log10();  // in-phase
        rmse_1 = 10.0 * (rmse_1 / num_samples_compare as f32).log10();  // 180-degree out of phase
        
        let rmse = if mod_type == AmpmodemType::Dsb && suppressed_carrier {
            rmse_0.min(rmse_1)
        } else {
            rmse_0
        };
        
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

}
