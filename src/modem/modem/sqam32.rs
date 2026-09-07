#![allow(clippy::excessive_precision)]
use crate::modem::modem::*;

#[derive(Debug, Clone)]
pub(super) struct Sqam32 {
    map: Vec<Complex32>, // 8-sample sub-map (first quadrant)
}

impl Modem {
    pub(super) fn new_sqam32() -> Result<Self> {
        let mut modem = Modem::_new(5, ModulationScheme::Sqam32)?;

        let symbol_map = MODEM_ARB_SQAM32.to_vec();
        modem.data = Some(ModemData::Sqam32(Sqam32 { map: symbol_map }));
        Ok(modem)
    }

    pub(super) fn modulate_sqam32(&mut self, symbol_in: u32) -> Result<Complex32> {
        let quad = (symbol_in >> 3) & 0x03;

        let s = symbol_in & 0x07;
        let symbol_map = if let Some(ModemData::Sqam32(sqam32)) = self.data.as_ref() {
            &sqam32.map
        } else {
            return Err(Error::Internal("modem data is not of type Sqam32".into()));
        };
        let p = symbol_map[s as usize];

        match quad {
            0 => Ok(p),
            1 => Ok(p.conj()),
            2 => Ok(-p.conj()),
            3 => Ok(-p),
            _ => Err(Error::Internal("invalid quadrant".into())),
        }
    }

    pub(super) fn demodulate_sqam32(&mut self, symbol_in: Complex32) -> Result<u32> {
        // determine quadrant and de-rotate to first quadrant
        // 10 | 00
        // ---+---
        // 11 | 01
        let quad = if symbol_in.re < 0.0 { 2 } else { 0 } + if symbol_in.im < 0.0 { 1 } else { 0 };

        let x_prime = match quad {
            0 => symbol_in,
            1 => symbol_in.conj(),
            2 => -symbol_in.conj(),
            3 => -symbol_in,
            _ => return Err(Error::Internal("invalid quadrant".into())),
        };

        let mut dmin = f32::INFINITY;
        let mut s = 0;
        let symbol_map = if let Some(ModemData::Sqam32(sqam32)) = self.data.as_ref() {
            &sqam32.map
        } else {
            return Err(Error::Internal("modem data is not of type Sqam32".into()));
        };
        for (i, symbol) in symbol_map.iter().enumerate() {
            let d = (x_prime - symbol).norm();
            if d < dmin {
                dmin = d;
                s = i as u32;
            }
        }

        s |= (quad as u32) << 3;

        self.x_hat = self.modulate_sqam32(s)?;
        self.r = symbol_in;
        Ok(s)
    }
}

// 'square' 32-QAM (first quadrant)
#[rustfmt::skip]
const MODEM_ARB_SQAM32: [Complex32; 8] = [
      Complex32::new(0.22361000,  0.22361000),   Complex32::new(0.67082000,  0.22361000), 
      Complex32::new(0.67082000,  1.11800000),   Complex32::new(1.11800000,  0.22361000), 
      Complex32::new(0.22361000,  0.67082000),   Complex32::new(0.67082000,  0.67082000), 
      Complex32::new(0.22361000,  1.11800000),   Complex32::new(1.11800000,  0.67082000)
];
