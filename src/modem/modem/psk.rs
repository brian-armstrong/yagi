use crate::modem::modem::*;

#[derive(Debug, Clone)]
pub(super) struct Psk {
    d_phi: f32,            // half of phase between symbols
    alpha: f32,            // scaling factor for phase symbols
}

impl Modem {
    pub(super) fn new_psk(scheme: ModulationScheme) -> Result<Self> {
        let bits_per_symbol = scheme.bits_per_symbol();
        let mut modem = Modem::_new(bits_per_symbol, scheme)?;

        let data = Psk {
            alpha: PI / modem.constellation_size as f32,
            d_phi: PI * (1.0 - 1.0 / modem.constellation_size as f32),
        };

        modem.reference = Some([0.0; MAX_MOD_BITS_PER_SYMBOL]);
        let reference = modem.reference.as_mut().unwrap();

        for k in 0..modem.bits_per_symbol {
            reference[k] = (1 << k) as f32 * data.alpha;
        }

        modem.data = Some(ModemData::Psk(data));

        modem.symbol_map = Some(vec![Complex32::new(0.0, 0.0); modem.constellation_size]);
        modem.init_map()?;

        if modem.bits_per_symbol >= 3 {
            modem.init_demod_soft_tab(2)?;
        }

        Ok(modem)
    }

    pub(super) fn modulate_psk(&mut self, symbol_in: u32) -> Result<Complex32> {
        let symbol_in = gray_decode(symbol_in);
        if let ModemData::Psk(Psk { alpha, .. }) = self.data.as_ref().unwrap() {
            let theta = Complex32::from_polar(1.0, symbol_in as f32 * 2.0 * alpha);
            Ok(theta)
        } else {
            Err(Error::Internal("modem data is not of type Psk".into()))
        }
    }

    pub(super) fn demodulate_psk(&mut self, symbol_in: Complex32) -> Result<u32> {
        let d_phi = match self.data.as_ref().unwrap() {
            ModemData::Psk(Psk { d_phi, .. }) => d_phi,
            _ => return Err(Error::Internal("modem data is not of type Psk".into())),
        };
        // f64 for stability
        let mut theta = symbol_in.arg() - d_phi;
        if (theta as f64) < -std::f64::consts::PI {
            theta = (theta as f64 + 2.0 * std::f64::consts::PI) as f32;
        }
        let (s, _demod_phase_error) = self.demodulate_linear_array_ref(theta, self.bits_per_symbol)?;
        let symbol_out = gray_encode(s);
        self.x_hat = self.modulate_psk(symbol_out)?;
        self.r = symbol_in;
        Ok(symbol_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modem_psk_negative_pi_boundary() {
        let cases: [(ModulationScheme, u32, u32); 4] = [
            (ModulationScheme::Psk2, 2, 1),
            (ModulationScheme::Psk4, 4, 2),
            (ModulationScheme::Psk8, 8, 4),
            (ModulationScheme::Psk16, 16, 8),
        ];

        for (scheme, m, want) in cases {
            let d_phi = std::f32::consts::PI * (1.0 - 1.0 / m as f32);
            let arg = -std::f32::consts::PI + d_phi;

            let mut q = Modem::new(scheme).unwrap();
            let sym = q.demodulate(Complex32::from_polar(1.0, arg)).unwrap();
            assert_eq!(sym, want, "PSK{} at the -pi boundary", m);
        }
    }
}