#![allow(clippy::excessive_precision)]
use crate::modem::modem::*;

#[derive(Debug, Clone)]
pub(super) struct Sqam128 {
    map: Vec<Complex32>, // 32-sample sub-map (first quadrant)
}

impl Modem {
    pub(super) fn new_sqam128() -> Result<Self> {
        let mut modem = Modem::new_base(7, ModulationScheme::Sqam128)?;

        let symbol_map = MODEM_ARB_SQAM128.to_vec();
        modem.data = Some(ModemData::Sqam128(Sqam128 { map: symbol_map }));
        Ok(modem)
    }

    pub(super) fn modulate_sqam128(&mut self, symbol_in: u32) -> Result<Complex32> {
        let quad = (symbol_in >> 5) & 0x03;

        let s = symbol_in & 0x1f;
        let symbol_map = if let Some(ModemData::Sqam128(sqam128)) = self.data.as_ref() {
            &sqam128.map
        } else {
            return Err(Error::Internal("modem data is not of type Sqam128".into()));
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

    pub(super) fn demodulate_sqam128(&mut self, symbol_in: Complex32) -> Result<u32> {
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
        let symbol_map = if let Some(ModemData::Sqam128(sqam128)) = self.data.as_ref() {
            &sqam128.map
        } else {
            return Err(Error::Internal("modem data is not of type Sqam128".into()));
        };
        for (i, symbol) in symbol_map.iter().enumerate() {
            let d = (x_prime - symbol).norm();
            if d < dmin {
                dmin = d;
                s = i as u32;
            }
        }

        s |= (quad as u32) << 5;

        self.x_hat = self.modulate_sqam128(s)?;
        self.r = symbol_in;
        Ok(s)
    }
}

// 'square' 128-QAM (first quadrant)
#[rustfmt::skip]
const MODEM_ARB_SQAM128: [Complex32; 32] = [
    Complex32::new( 0.11043000,  0.11043000), Complex32::new( 0.33129000,  0.11043000),
    Complex32::new( 0.11043000,  0.33129000), Complex32::new( 0.33129000,  0.33129000),
    Complex32::new( 0.77302000,  0.11043000), Complex32::new( 0.55216000,  0.11043000),
    Complex32::new( 0.77302000,  0.33129000), Complex32::new( 0.55216000,  0.33129000),
    Complex32::new( 0.77302000,  0.99388000), Complex32::new( 0.55216000,  0.99388000),
    Complex32::new( 0.77302000,  1.21470000), Complex32::new( 0.55216000,  1.21470000),
    Complex32::new( 0.99388000,  0.11043000), Complex32::new( 1.21470000,  0.11043000),
    Complex32::new( 0.99388000,  0.33129000), Complex32::new( 1.21470000,  0.33129000),
    Complex32::new( 0.11043000,  0.77302000), Complex32::new( 0.33129000,  0.77302000),
    Complex32::new( 0.11043000,  0.55216000), Complex32::new( 0.33129000,  0.55216000),
    Complex32::new( 0.77302000,  0.77302000), Complex32::new( 0.55216000,  0.77302000),
    Complex32::new( 0.77302000,  0.55216000), Complex32::new( 0.55216000,  0.55216000),
    Complex32::new( 0.11043000,  0.99388000), Complex32::new( 0.33129000,  0.99388000),
    Complex32::new( 0.11043000,  1.21470000), Complex32::new( 0.33129000,  1.21470000),
    Complex32::new( 0.99388000,  0.77302000), Complex32::new( 1.21470000,  0.77302000),
    Complex32::new( 0.99388000,  0.55216000), Complex32::new( 1.21470000,  0.55216000)
];
