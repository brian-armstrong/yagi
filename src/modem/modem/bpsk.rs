use crate::modem::modem::*;

impl Modem {
    pub(super) fn new_bpsk() -> Result<Self> {
        let mut modem = Self::_new(1, ModulationScheme::Bpsk)?;
        modem.demodulate_soft_func = Some(Self::demodulate_soft_bpsk);
        modem.symbol_map = Some(vec![Complex32::new(0.0, 0.0); modem.constellation_size]);
        modem.init_map()?;
        Ok(modem)
    }

    pub(super) fn modulate_bpsk(&mut self, sym_in: u32) -> Result<Complex32> {
        let y = if sym_in == 0 { 1.0 } else { -1.0 };
        Ok(Complex32::new(y, 0.0))
    }

    pub(super) fn demodulate_bpsk(&mut self, x: Complex32) -> Result<u32> {
        let sym_out = if x.re > 0.0 { 0 } else { 1 };
        self.x_hat = self.modulate_bpsk(sym_out)?;
        self.r = x;
        Ok(sym_out)
    }

    fn demodulate_soft_bpsk(&mut self, x: Complex32, soft_bits: &mut [u8]) -> Result<u32> {
        let gamma = 4.0;

        let llr = -2.0 * x.re * gamma;
        let soft_bit = ((llr * 16.0 + 127.0)).clamp(0.0, 255.0) as u8;
        soft_bits[0] = soft_bit;

        let sym_out = if x.re > 0.0 { 0 } else { 1 };
        self.x_hat = self.modulate_bpsk(sym_out)?;
        self.r = x;
        Ok(sym_out)
    }
}
