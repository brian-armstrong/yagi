use crate::modem::modem::*;

impl Modem {
    pub(super) fn new_arb_preset(table: &[Complex32]) -> Result<Self> {
        if !table.len().is_power_of_two() {
            return Err(Error::Config("constellation table size must be power of 2".into()));
        }
        Modem::new_arb(ModulationScheme::Arb, table, table.len().ilog2() as usize)
    }

    pub(super) fn new_arb(scheme: ModulationScheme, table: &[Complex32], m: usize) -> Result<Self> {
        let mut modem = Modem::_new(m, scheme)?;
        modem.symbol_map = Some(table.to_vec());
        if scheme == ModulationScheme::Arb {
            modem.arb_balance_iq()?;
            modem.demodulate_soft_func = Some(Self::demodulate_soft_arb);
        }
        modem.arb_scale()?;
        Ok(modem)
    }

    pub(super) fn modulate_arb(&mut self, symbol: u32) -> Result<Complex32> {
        let symbol_map = self.symbol_map.as_ref().unwrap();
        let symbol = symbol_map[symbol as usize];
        Ok(symbol)
    }

    pub(super) fn demodulate_arb(&mut self, symbol_in: Complex32) -> Result<u32> {
        let symbol_map = self.symbol_map.as_ref().unwrap();
        let mut min_distance = f32::INFINITY;
        let mut min_index = 0;
        for (index, symbol) in symbol_map.iter().enumerate() {
            let distance = (symbol_in - symbol).norm();
            if distance < min_distance {
                min_distance = distance;
                min_index = index;
            }
        }
        self.x_hat = self.modulate_arb(min_index as u32)?;
        self.r = symbol_in;
        Ok(min_index as u32)
    }

    fn demodulate_soft_arb(&mut self, symbol_in: Complex32, soft_bits: &mut [u8]) -> Result<u32> {
        let gamma = 1.2 * self.bits_per_symbol as f32;

        let mut dmin_0 = vec![4.0; self.bits_per_symbol];
        let mut dmin_1 = vec![4.0; self.bits_per_symbol];
        let mut dmin = f32::INFINITY;
        let mut s = 0;
        for (index, symbol) in self.symbol_map.as_ref().unwrap().iter().enumerate() {
            let d = ((symbol_in - symbol) * (symbol_in - symbol).conj()).re;

            if d < dmin {
                dmin = d;
                s = index as u32;
            }

            for k in 0..self.bits_per_symbol {
                if (s >> (self.bits_per_symbol - k - 1)) & 0x01 == 0 {
                    if d < dmin_0[k] {
                        dmin_0[k] = d;
                    }
                } else {
                    if d < dmin_1[k] {
                        dmin_1[k] = d;
                    }
                }
            }
        }

        for k in 0..self.bits_per_symbol {
            let soft_bit = ((dmin_0[k] - dmin_1[k]) * gamma) * 16.0 + 127.0;
            let soft_bit = soft_bit.clamp(0.0, 255.0) as u8;
            soft_bits[k] = soft_bit;
        }

        self.x_hat = self.modulate_arb(s)?;
        self.r = symbol_in;
        Ok(s)
    }

    fn arb_scale(&mut self) -> Result<()> {
        let symbol_map = self.symbol_map.as_mut().unwrap();

        let energy = symbol_map.iter().map(|x| x.norm_sqr()).sum::<f32>();
        let scale = (energy / symbol_map.len() as f32).sqrt();
        for symbol in symbol_map.iter_mut() {
            *symbol /= scale;
        }
        Ok(())
    }
    
    fn arb_balance_iq(&mut self) -> Result<()> {
        let symbol_map = self.symbol_map.as_mut().unwrap();
        let mean = symbol_map.iter().sum::<Complex32>() / symbol_map.len() as f32;
        for symbol in symbol_map.iter_mut() {
            *symbol -= mean;
        }
        Ok(())
    }
}
