use crate::modem::modem::*;

#[derive(Debug, Clone)]
pub(super) struct Dpsk {
    d_phi: f32,            // half of phase between symbols
    phi: f32,              // angle state for differential PSK
    alpha: f32,            // scaling factor for phase symbols
}

impl Dpsk {
    pub(super) fn reset(&mut self) {
        self.phi = 0.0;
    }
}

impl Modem {
    pub(super) fn new_dpsk(scheme: ModulationScheme) -> Result<Self> {
        let bits_per_symbol = scheme.bits_per_symbol();
        let mut modem = Self::_new(bits_per_symbol, scheme)?;

        let alpha = PI / modem.constellation_size as f32;
        
        modem.data = Some(ModemData::Dpsk(Dpsk {
            alpha,
            phi: 0.0,
            d_phi: PI * (1.0 - 1.0 / modem.constellation_size as f32),
        }));

        modem.reference = Some([0.0; MAX_MOD_BITS_PER_SYMBOL]);
        let reference = modem.reference.as_mut().unwrap();
        for k in 0..modem.bits_per_symbol {
            reference[k] = (1 << k) as f32 * alpha;
        }

        Ok(modem)
    }

    pub(super) fn modulate_dpsk(&mut self, sym_in: u32) -> Result<Complex32> {
        if let ModemData::Dpsk(dpsk) = self.data.as_mut().unwrap() {
            // 'encode' input symbol (actually gray decoding)
            let sym_in = gray_decode(sym_in);

            // compute phase difference between this symbol and the previous
            dpsk.phi += sym_in as f32 * 2.0 * dpsk.alpha;

            // limit phase
            if dpsk.phi > 2.0 * PI {
                dpsk.phi -= 2.0 * PI;
            }
            
            // compute output sample
            let y = Complex32::from_polar(1.0, dpsk.phi);

            // save symbol state
            self.r = y;
            Ok(y)
        } else {
            Err(Error::Internal("modem data is not of type Dpsk".into()))
        }
    }

    pub(super) fn demodulate_dpsk(&mut self, x: Complex32) -> Result<u32> {
        let theta = x.arg();
        let d_theta = {
            if let ModemData::Dpsk(dpsk) = self.data.as_mut().unwrap() {
                // compute angle difference
                let mut d_theta = theta - dpsk.phi;
                dpsk.phi = theta;

                // subtract phase offset, ensuring phase is in [-pi,pi)
                d_theta -= dpsk.d_phi;
                if d_theta > PI {
                    d_theta -= 2.0 * PI;
                } else if d_theta < -PI {
                    d_theta += 2.0 * PI;
                }
                d_theta
            } else {
                return Err(Error::Internal("modem data is not of type Dpsk".into()));
            }
        };

        // demodulate on linearly-spaced array
        let (s, demod_phase_error) = self.demodulate_linear_array_ref(d_theta, self.bits_per_symbol)?;

        // 'decode' output symbol (actually gray encoding)
        let sym_out = gray_encode(s);

        // re-modulate symbol (accounting for differential rotation)
        // and store state
        self.x_hat = Complex32::from_polar(1.0, theta - demod_phase_error);
        self.r = x;
        Ok(sym_out)
    }
}