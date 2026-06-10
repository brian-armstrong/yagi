use crate::modem::modem::*;

impl Modem {
    pub(super) fn new_ook() -> Result<Self> {
        let mut modem = Modem::_new(1, ModulationScheme::Ook)?;
        modem.symbol_map = Some(vec![Complex32::new(0.0, 0.0); modem.constellation_size]);
        modem.init_map()?;
        Ok(modem)
    }

    pub(super) fn modulate_ook(&mut self, symbol_in: u32) -> Result<Complex32> {
        if symbol_in != 0 {
            Ok(Complex32::new(0.0, 0.0))
        } else {
            Ok(Complex32::new(SQRT_2, 0.0))
        }
    }

    pub(super) fn demodulate_ook(&mut self, symbol_in: Complex32) -> Result<u32> {
        let symbol_out = if symbol_in.re > FRAC_1_SQRT_2 {
            0
        } else {
            1
        };
        self.x_hat = self.modulate_ook(symbol_out)?;
        self.r = symbol_in;
        Ok(symbol_out)
    }
}