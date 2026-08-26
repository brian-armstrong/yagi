use crate::modem::modem::*;

#[derive(Debug, Clone)]
pub(super) struct Pi4Dqpsk {
    theta: f32,    // phase state
}

impl Pi4Dqpsk {
    pub(super) fn reset(&mut self) {
        self.theta = 0.0;
    }
}

impl Modem {
    pub(super) fn new_pi4dqpsk() -> Result<Self> {
        let mut modem = Modem::_new(2, ModulationScheme::Pi4Dqpsk)?;
        let data = Pi4Dqpsk { theta: 0.0 };
        modem.data = Some(ModemData::Pi4Dqpsk(data));
        modem.demodulate_soft_func = Some(Self::demodulate_soft_pi4dqpsk);
        Ok(modem)
    }

    pub(super) fn modulate_pi4dqpsk(&mut self, sym_in: u32) -> Result<Complex32> {
        if let ModemData::Pi4Dqpsk(pi4dqpsk) = self.data.as_mut().unwrap() {
            let d_theta = match sym_in {
                0 => 0.25 * PI,
                1 => 0.75 * PI,
                2 => -0.25 * PI,
                3 => -0.75 * PI,
                _ => return Err(Error::Config("invalid input symbol".to_string())),
            };

            pi4dqpsk.theta += d_theta;

            // f64 for numerical stability
            let theta = pi4dqpsk.theta as f64;
            if theta > std::f64::consts::PI {
                pi4dqpsk.theta = (theta - 2.0 * std::f64::consts::PI) as f32;
            } else if theta < -std::f64::consts::PI {
                pi4dqpsk.theta = (theta + 2.0 * std::f64::consts::PI) as f32;
            }

            let y = Complex32::from_polar(1.0, pi4dqpsk.theta);
            Ok(y)
        } else {
            Err(Error::Internal("invalid modem data".to_string()))
        }
    }

    pub(super) fn demodulate_pi4dqpsk(&mut self, x: Complex32) -> Result<u32> {
        if let ModemData::Pi4Dqpsk(pi4dqpsk) = self.data.as_mut().unwrap() {
            let theta = x.arg();

            // f64 for numerical stability
            let mut d_theta = (theta - pi4dqpsk.theta) as f64;
            while d_theta > std::f64::consts::PI {
                d_theta -= 2.0 * std::f64::consts::PI;
            }
            while d_theta < -std::f64::consts::PI {
                d_theta += 2.0 * std::f64::consts::PI;
            }

            let sym_out = match d_theta {
                d if d > 0.5 * std::f64::consts::PI => 1,
                d if d > 0.0 => 0,
                d if d < -0.5 * std::f64::consts::PI => 3,
                _ => 2,
            };

            let d_theta_ideal = match sym_out {
                0 => 0.25 * PI,
                1 => 0.75 * PI,
                2 => -0.25 * PI,
                3 => -0.75 * PI,
                _ => return Err(Error::Internal("invalid output symbol".to_string())),
            };

            self.x_hat = Complex32::from_polar(1.0, pi4dqpsk.theta + d_theta_ideal);
            self.r = x;
            pi4dqpsk.theta = theta;
            Ok(sym_out)
        } else {
            Err(Error::Internal("invalid modem data".to_string()))
        }
    }

    fn demodulate_soft_pi4dqpsk(&mut self, x: Complex32, soft_bits: &mut [u8]) -> Result<u32> {
        let s = self.demodulate_pi4dqpsk(x)?;
        soft_bits[0] = if s & 2 == 0 { 0 } else { 255 };
        soft_bits[1] = if s & 1 == 0 { 0 } else { 255 };
        Ok(s)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modem_pi4dqpsk_negative_real_axis() {
        // purely real, negative values expose a bug with f32 arithmetic above
        assert_eq!(Complex32::new(-1.0, 0.0).arg(), std::f32::consts::PI);

        for re in [-1.0f32, -0.5, -2.0, -0.25] {
            let mut q = Modem::new(ModulationScheme::Pi4Dqpsk).unwrap();
            let sym = q.demodulate(Complex32::new(re, 0.0)).unwrap();
            assert_eq!(sym, 3, "pi4dqpsk, re = {}, expected symbol 3, got {}", re, sym);
        }
    }

    #[test]
    fn test_modem_pi4dqpsk_slicer_boundary() {
        let half_pi = std::f32::consts::FRAC_PI_2;
        let cases: [(f32, u32); 6] = [
            (half_pi, 1),
            (-half_pi, 3),
            (f32::from_bits(half_pi.to_bits() - 1), 0), // one ulp below +pi/2
            (f32::from_bits(half_pi.to_bits() + 1), 1), // one ulp above
            (f32::from_bits((-half_pi).to_bits() - 1), 2),
            (f32::from_bits((-half_pi).to_bits() + 1), 3),
        ];

        for (d_theta, want) in cases {
            let mut q = Modem::new(ModulationScheme::Pi4Dqpsk).unwrap();
            // first symbol establishes the reference phase at 0
            q.demodulate(Complex32::from_polar(1.0, 0.0)).unwrap();
            let sym = q.demodulate(Complex32::from_polar(1.0, d_theta)).unwrap();
            assert_eq!(sym, want, "d_theta = {:+.9}", d_theta);
        }
    }
}