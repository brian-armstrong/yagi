use crate::modem::modem::*;

#[derive(Debug, Clone)]
pub(super) struct Ask {
    alpha: f32,            // scaling factor to ensure unity energy
}

impl Modem {
    pub(super) fn new_ask(bits_per_symbol: usize) -> Result<Self> {
        let (alpha, scheme) = match 1 << bits_per_symbol {
            2 => (1.0, ModulationScheme::Ask2),
            4 => (1.0 / 5.0_f32.sqrt(), ModulationScheme::Ask4),
            8 => (1.0 / 21.0_f32.sqrt(), ModulationScheme::Ask8),
            16 => (1.0 / 85.0_f32.sqrt(), ModulationScheme::Ask16),
            32 => (1.0 / 341.0_f32.sqrt(), ModulationScheme::Ask32),
            64 => (1.0 / 1365.0_f32.sqrt(), ModulationScheme::Ask64),
            128 => (1.0 / 5461.0_f32.sqrt(), ModulationScheme::Ask128),
            256 => (1.0 / 21845.0_f32.sqrt(), ModulationScheme::Ask256),
            _ => unreachable!(),
        };

        let mut modem = Self::_new(bits_per_symbol, scheme)?;
        modem.data = Some(ModemData::Ask(Ask { alpha }));

        modem.reference = Some([0.0; MAX_MOD_BITS_PER_SYMBOL]);
        let reference = modem.reference.as_mut().unwrap();

        for k in 0..bits_per_symbol {
            reference[k] = (1 << k) as f32 * alpha;
        }

        if bits_per_symbol >= 2 && bits_per_symbol < 8 {
            modem.init_demod_soft_tab(2)?;
        }

        modem.symbol_map = Some(vec![Complex32::new(0.0, 0.0); modem.constellation_size]);
        modem.init_map()?;

        Ok(modem)
    }

    pub(super) fn modulate_ask(&mut self, sym_in: u32) -> Result<Complex32> {
        let ask = match &self.data {
            Some(ModemData::Ask(ask)) => ask,
            _ => return Err(Error::Internal("modem data is not of type Ask".into())),
        };

        // 'encode' input symbol (actually gray decoding)
        let sym_in = gray_decode(sym_in);

        // modulate symbol
        let y = (2 * sym_in as i32 - self.constellation_size as i32 + 1) as f32 * ask.alpha;
        Ok(Complex32::new(y, 0.0))
    }

    pub(super) fn demodulate_ask(&mut self, x: Complex32) -> Result<u32> {
        // demodulate on linearly-spaced array
        let (s, _res_i) = self.demodulate_linear_array_ref(x.re, self.bits_per_symbol)?;

        // 'decode' output symbol (actually gray encoding)
        let sym_out = gray_encode(s);

        // re-modulate symbol and store state
        self.x_hat = self.modulate_ask(sym_out)?;
        self.r = x;
        Ok(sym_out)
    }

}