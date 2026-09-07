use crate::modem::modem::*;

impl Modem {
    pub(super) fn new_qpsk() -> Result<Self> {
        let mut modem = Self::_new(2, ModulationScheme::Qpsk)?;
        modem.demodulate_soft_func = Some(Self::demodulate_soft_qpsk);
        Ok(modem)
    }

    pub(super) fn modulate_qpsk(&mut self, sym_in: u32) -> Result<Complex32> {
        let re = if sym_in & 0x01 == 0 { FRAC_1_SQRT_2 } else { -FRAC_1_SQRT_2 };
        let im = if sym_in & 0x02 == 0 { FRAC_1_SQRT_2 } else { -FRAC_1_SQRT_2 };
        Ok(Complex32::new(re, im))
    }

    pub(super) fn demodulate_qpsk(&mut self, x: Complex32) -> Result<u32> {
        let sym_out = if x.re > 0.0 { 0 } else { 1 } + if x.im > 0.0 { 0 } else { 2 };
        self.x_hat = self.modulate_qpsk(sym_out)?;
        self.r = x;
        Ok(sym_out)
    }

    fn demodulate_soft_qpsk(&mut self, x: Complex32, soft_bits: &mut [u8]) -> Result<u32> {
        let gamma = 5.8;

        let llr = -2.0 * x.im * gamma;
        let soft_bit = (llr * 16.0 + 127.0).clamp(0.0, 255.0) as u8;
        soft_bits[0] = soft_bit;

        let llr = -2.0 * x.re * gamma;
        let soft_bit = (llr * 16.0 + 127.0).clamp(0.0, 255.0) as u8;
        soft_bits[1] = soft_bit;

        let sym_out = if x.re > 0.0 { 0 } else { 1 } + if x.im > 0.0 { 0 } else { 2 };
        self.x_hat = self.modulate_qpsk(sym_out)?;
        self.r = x;
        Ok(sym_out)
    }
}
