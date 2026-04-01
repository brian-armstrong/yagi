use crate::{error::{Error, Result}, sequence::MSequence};
use num_complex::Complex32;
use std::f32::consts::{SQRT_2, FRAC_1_SQRT_2, PI};

mod apsk;
mod arb;
mod arb_v29;
mod arb_opt;
mod arb_ui;
mod arb_vt;
mod ask;
mod bpsk;
mod dpsk;
mod ook;
mod pi4dqpsk;
mod psk;
mod qam;
mod qpsk;
mod sqam32;
mod sqam128;

const MAX_MOD_BITS_PER_SYMBOL: usize = 8;
const SOFTBIT_ERASURE: u8 = 127;
const SOFTBIT_0: u8 = 0;
const SOFTBIT_1: u8 = 255;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModulationScheme {
    Unknown, // Unknown modulation scheme

    // Phase-shift keying (PSK)
    Psk2,      Psk4,
    Psk8,      Psk16,
    Psk32,     Psk64,
    Psk128,    Psk256,

    // Differential phase-shift keying (DPSK)
    Dpsk2,     Dpsk4,
    Dpsk8,     Dpsk16,
    Dpsk32,    Dpsk64,
    Dpsk128,   Dpsk256,

    // amplitude-shift keying
    Ask2,      Ask4,
    Ask8,      Ask16,
    Ask32,     Ask64,
    Ask128,    Ask256,

    // rectangular quadrature amplitude-shift keying (QAM)
    Qam4,
    Qam8,      Qam16,
    Qam32,     Qam64,
    Qam128,    Qam256,

    // amplitude phase-shift keying (APSK)
    Apsk4,
    Apsk8,     Apsk16,
    Apsk32,    Apsk64,
    Apsk128,   Apsk256,

    // specific modem types
    Bpsk,      // Specific: binary PSK
    Qpsk,      // specific: quaternary PSK
    Ook,       // Specific: on/off keying
    Sqam32,    // 'square' 32-QAM
    Sqam128,   // 'square' 128-QAM
    V29,       // V.29 star constellation
    Arb16Opt,  // optimal 16-QAM
    Arb32Opt,  // optimal 32-QAM
    Arb64Opt,  // optimal 64-QAM
    Arb128Opt, // optimal 128-QAM
    Arb256Opt, // optimal 256-QAM
    Arb64Vt,   // Virginia Tech logo
    Arb64Ui,   // University of Illinois logo
    Pi4Dqpsk,  // pi/4 differential QPSK

    // arbitrary modem type
    Arb        // arbitrary QAM
}

#[derive(Debug, Clone)]
pub struct Modem {
    // common data
    scheme: ModulationScheme,   // modulation scheme
    bits_per_symbol: usize,       // bits per symbol (modulation depth)
    constellation_size: usize,   // constellation size, M=2^m

    // Reference vector for demodulating linear arrays
    //
    // By storing these values in an array they do not need to be
    // calculated during run-time.  This speeds up the demodulation by
    // approximately 8%.
    reference: Option<[f32; MAX_MOD_BITS_PER_SYMBOL]>,

    // modulation
    symbol_map: Option<Vec<Complex32>>,     // complete symbol map
    modulate_using_map: bool,         // modulate using map (look-up table) flag

    // demodulation
    r: Complex32,                // received state vector
    x_hat: Complex32,            // estimated symbol (demodulator)

    // common data structure shared between specific modem types
    data: Option<ModemData>,

    // modulate function pointer
    modulate_func: fn(&mut Modem, u32) -> Result<Complex32>,

    // demodulate function pointer
    demodulate_func: fn(&mut Modem, Complex32) -> Result<u32>,

    // soft demodulation
    demodulate_soft_func: Option<fn(&mut Modem, Complex32, &mut [u8]) -> Result<u32>>,

    // soft demodulation
    // neighbors array
    demod_soft_neighbors: Option<Vec<u8>>,   // array of nearest neighbors
    demod_soft_p: u32,              // number of neighbors in array

    randomizer: MSequence,
}


#[derive(Debug, Clone)]
enum ModemData {
    Psk(psk::Psk),
    Dpsk(dpsk::Dpsk),
    Ask(ask::Ask),
    Qam(qam::Qam),
    Apsk(apsk::Apsk),
    Sqam32(sqam32::Sqam32),
    Sqam128(sqam128::Sqam128),
    Pi4Dqpsk(pi4dqpsk::Pi4Dqpsk),
}

impl ModemData {
    pub fn reset(&mut self) {
        match self {
            ModemData::Dpsk(dpsk) => {
                dpsk.reset();
            }
            ModemData::Pi4Dqpsk(pi4dqpsk) => {
                pi4dqpsk.reset();
            }
            _ => {}
        }
    }
}

impl Modem {
    pub fn new(scheme: ModulationScheme) -> Result<Self> {
        match scheme {
            ModulationScheme::Psk2 => Modem::new_psk(1),
            ModulationScheme::Psk4 => Modem::new_psk(2),
            ModulationScheme::Psk8 => Modem::new_psk(3),
            ModulationScheme::Psk16 => Modem::new_psk(4),
            ModulationScheme::Psk32 => Modem::new_psk(5),
            ModulationScheme::Psk64 => Modem::new_psk(6),
            ModulationScheme::Psk128 => Modem::new_psk(7),
            ModulationScheme::Psk256 => Modem::new_psk(8),
            ModulationScheme::Dpsk2 => Modem::new_dpsk(1),
            ModulationScheme::Dpsk4 => Modem::new_dpsk(2),
            ModulationScheme::Dpsk8 => Modem::new_dpsk(3),
            ModulationScheme::Dpsk16 => Modem::new_dpsk(4),
            ModulationScheme::Dpsk32 => Modem::new_dpsk(5),
            ModulationScheme::Dpsk64 => Modem::new_dpsk(6),
            ModulationScheme::Dpsk128 => Modem::new_dpsk(7),
            ModulationScheme::Dpsk256 => Modem::new_dpsk(8),
            ModulationScheme::Ask2 => Modem::new_ask(1),
            ModulationScheme::Ask4 => Modem::new_ask(2),
            ModulationScheme::Ask8 => Modem::new_ask(3),
            ModulationScheme::Ask16 => Modem::new_ask(4),
            ModulationScheme::Ask32 => Modem::new_ask(5),
            ModulationScheme::Ask64 => Modem::new_ask(6),
            ModulationScheme::Ask128 => Modem::new_ask(7),
            ModulationScheme::Ask256 => Modem::new_ask(8),
            ModulationScheme::Qam4 => Modem::new_qam(2),
            ModulationScheme::Qam8 => Modem::new_qam(3),
            ModulationScheme::Qam16 => Modem::new_qam(4),
            ModulationScheme::Qam32 => Modem::new_qam(5),
            ModulationScheme::Qam64 => Modem::new_qam(6),
            ModulationScheme::Qam128 => Modem::new_qam(7),
            ModulationScheme::Qam256 => Modem::new_qam(8),
            ModulationScheme::Apsk4 => Modem::new_apsk(2),
            ModulationScheme::Apsk8 => Modem::new_apsk(3),
            ModulationScheme::Apsk16 => Modem::new_apsk(4),
            ModulationScheme::Apsk32 => Modem::new_apsk(5),
            ModulationScheme::Apsk64 => Modem::new_apsk(6),
            ModulationScheme::Apsk128 => Modem::new_apsk(7),
            ModulationScheme::Apsk256 => Modem::new_apsk(8),
            ModulationScheme::Bpsk => Modem::new_bpsk(),
            ModulationScheme::Qpsk => Modem::new_qpsk(),
            ModulationScheme::Ook => Modem::new_ook(),
            ModulationScheme::Sqam32 => Modem::new_sqam32(),
            ModulationScheme::Sqam128 => Modem::new_sqam128(),
            ModulationScheme::V29 => Modem::new_arb_v29(),
            ModulationScheme::Arb16Opt => Modem::new_arb_opt16(),
            ModulationScheme::Arb32Opt => Modem::new_arb_opt32(),
            ModulationScheme::Arb64Opt => Modem::new_arb_opt64(),
            ModulationScheme::Arb128Opt => Modem::new_arb_opt128(),
            ModulationScheme::Arb256Opt => Modem::new_arb_opt256(),
            ModulationScheme::Arb64Vt => Modem::new_arb_vt(),
            ModulationScheme::Arb64Ui => Modem::new_arb_ui(),
            ModulationScheme::Pi4Dqpsk => Modem::new_pi4dqpsk(),
            _ => Err(Error::Config("modulation scheme not supported".into())),
        }
    }

    pub fn from_table(table: &[Complex32]) -> Result<Self> {
        let bits_per_symbol = crate::math::nextpow2(table.len() as u32)? as usize;
        if (1 << bits_per_symbol) != table.len() {
            return Err(Error::Config("table size must be power of 2".into()));
        }
        let modem = Modem::new_arb(ModulationScheme::Arb, table, bits_per_symbol)?;
        Ok(modem)
    }

    pub fn reset(&mut self) {
        self.r = Complex32::new(1.0, 0.0);
        self.x_hat = self.r;
        if let Some(data) = &mut self.data {
            data.reset();
        }
    }

    pub fn get_bps(&self) -> usize {
        self.bits_per_symbol
    }

    pub fn get_scheme(&self) -> ModulationScheme {
        self.scheme
    }

    pub fn get_constellation_size(&self) -> usize {
        self.constellation_size
    }

    pub fn random_symbol(&mut self) -> u32 {
        self.randomizer.generate_symbol(self.bits_per_symbol as u32)
        // rand::random::<u32>() % self.constellation_size as u32
    }

    pub fn modulate(&mut self, symbol_in: u32) -> Result<Complex32> {
        if symbol_in >= self.constellation_size as u32 {
            return Err(Error::Config("input symbol exceeds constellation size".into()));
        }

        if self.modulate_using_map {
            self.modulate_map(symbol_in)
        } else {
            (self.modulate_func)(self, symbol_in)
        }
    }

    pub fn demodulate(&mut self, symbol_in: Complex32) -> Result<u32> {
        (self.demodulate_func)(self, symbol_in)
    }

    pub fn demodulate_soft(&mut self, symbol_in: Complex32, soft_bits: &mut [u8]) -> Result<u32> {
        if let Some(demodulate_soft_func) = self.demodulate_soft_func {
            return demodulate_soft_func(self, symbol_in, soft_bits);
        }

        if self.demod_soft_neighbors.is_some() && self.demod_soft_p > 0 {
            return self.demodulate_soft_table(symbol_in, soft_bits);
        }

        let symbol_out = (self.demodulate_func)(self, symbol_in)?;
        unpack_soft_bits(symbol_out, self.bits_per_symbol, soft_bits)?;
        Ok(symbol_out)
    }

    pub fn get_demodulator_sample(&self) -> Complex32 {
        self.x_hat
    }

    pub fn get_demodulator_phase_error(&self) -> f32 {
        (self.r * self.x_hat.conj()).im
    }

    pub fn get_demodulator_evm(&self) -> f32 {
        (self.x_hat - self.r).norm()
    }

    fn modulate_map(&mut self, symbol_in: u32) -> Result<Complex32> {
        if symbol_in >= self.constellation_size as u32 {
            return Err(Error::Config("input symbol exceeds maximum".into()));
        }
        if let Some(ref symbol_map) = self.symbol_map {
            Ok(symbol_map[symbol_in as usize])
        } else {
            Err(Error::Config("symbol table not initialized".into()))
        }
    }

    fn demodulate_linear_array_ref(
        &mut self,
        v: f32,
        m: usize
    ) -> Result<(u32, f32)> {
        let mut s = 0;
        let mut v = v;
        let ref_table = self.reference.as_ref().unwrap();
        for k in 0..m {
            s <<= 1;
            if v > 0.0 {
                s |= 1;
                v -= ref_table[m - k - 1];
            } else {
                s |= 0;
                v += ref_table[m - k - 1];
            }
        }
        Ok((s, v))
    }    

    fn demodulate_soft_table(&mut self, symbol_in: Complex32, soft_bits: &mut [u8]) -> Result<u32> {
        let symbol_out = (self.demodulate_func)(self, symbol_in)?;
        let bps = self.bits_per_symbol;
        let gamma = 1.2 * self.constellation_size as f32;
        let mut dmin_0 = vec![8.0; bps];
        let mut dmin_1 = vec![8.0; bps];
        
        let d = ((symbol_in - self.x_hat) * (symbol_in - self.x_hat).conj()).re;
        for k in 0..bps {
            let bit = (symbol_out >> (bps - k - 1)) & 0x01;
            if bit != 0 {
                dmin_1[k] = d;
            } else {
                dmin_0[k] = d;
            }
        }

        let p = self.demod_soft_p;

        for i in 0..p {
            let soft_neighbors = self.demod_soft_neighbors.as_ref().unwrap();
            let neighbor = soft_neighbors[(symbol_out * p + i) as usize] as u32;
            let x_hat = self.modulate(neighbor)?;
            let e = symbol_in - x_hat;
            let d = (e.re * e.re) + (e.im * e.im);
            let soft_neighbors = self.demod_soft_neighbors.as_ref().unwrap();
            for k in 0..bps {
                let bit = (soft_neighbors[(symbol_out * p + i) as usize] >> (bps - k - 1)) & 0x01;
                if bit != 0 {
                    if d < dmin_1[k] {
                        dmin_1[k] = d;
                    }
                } else {
                    if d < dmin_0[k] {
                        dmin_0[k] = d;
                    }
                }
            }
        }

        for k in 0..bps {
            let soft_bit = ((dmin_0[k] - dmin_1[k]) * gamma) * 16.0 + 127.0;
            let soft_bit = soft_bit.clamp(0.0, 255.0) as u8;
            soft_bits[k] = soft_bit;
        }

        Ok(symbol_out)
    }

    fn _new(bits_per_symbol: usize, scheme: ModulationScheme) -> Result<Self> {
        if bits_per_symbol < 1 {
            return Err(Error::Config("modem must have at least 1 bit per symbol".into()));
        }
        if bits_per_symbol > MAX_MOD_BITS_PER_SYMBOL {
            return Err(Error::Config("maximum number of bits per symbol exceeded".into()));
        }

        let (mod_fn, demod_fn) = match scheme {
            ModulationScheme::Psk2 => (Self::modulate_psk as _, Self::demodulate_psk as _),
            ModulationScheme::Psk4 => (Self::modulate_psk as _, Self::demodulate_psk as _),
            ModulationScheme::Psk8 => (Self::modulate_psk as _, Self::demodulate_psk as _),
            ModulationScheme::Psk16 => (Self::modulate_psk as _, Self::demodulate_psk as _),
            ModulationScheme::Psk32 => (Self::modulate_psk as _, Self::demodulate_psk as _),
            ModulationScheme::Psk64 => (Self::modulate_psk as _, Self::demodulate_psk as _),
            ModulationScheme::Psk128 => (Self::modulate_psk as _, Self::demodulate_psk as _),
            ModulationScheme::Psk256 => (Self::modulate_psk as _, Self::demodulate_psk as _),
            ModulationScheme::Dpsk2 => (Self::modulate_dpsk as _, Self::demodulate_dpsk as _),
            ModulationScheme::Dpsk4 => (Self::modulate_dpsk as _, Self::demodulate_dpsk as _),
            ModulationScheme::Dpsk8 => (Self::modulate_dpsk as _, Self::demodulate_dpsk as _),
            ModulationScheme::Dpsk16 => (Self::modulate_dpsk as _, Self::demodulate_dpsk as _),
            ModulationScheme::Dpsk32 => (Self::modulate_dpsk as _, Self::demodulate_dpsk as _),
            ModulationScheme::Dpsk64 => (Self::modulate_dpsk as _, Self::demodulate_dpsk as _),
            ModulationScheme::Dpsk128 => (Self::modulate_dpsk as _, Self::demodulate_dpsk as _),
            ModulationScheme::Dpsk256 => (Self::modulate_dpsk as _, Self::demodulate_dpsk as _),
            ModulationScheme::Ask2 => (Self::modulate_ask as _, Self::demodulate_ask as _),
            ModulationScheme::Ask4 => (Self::modulate_ask as _, Self::demodulate_ask as _),
            ModulationScheme::Ask8 => (Self::modulate_ask as _, Self::demodulate_ask as _),
            ModulationScheme::Ask16 => (Self::modulate_ask as _, Self::demodulate_ask as _),
            ModulationScheme::Ask32 => (Self::modulate_ask as _, Self::demodulate_ask as _),
            ModulationScheme::Ask64 => (Self::modulate_ask as _, Self::demodulate_ask as _),
            ModulationScheme::Ask128 => (Self::modulate_ask as _, Self::demodulate_ask as _),
            ModulationScheme::Ask256 => (Self::modulate_ask as _, Self::demodulate_ask as _),
            ModulationScheme::Qam4 => (Self::modulate_qam as _, Self::demodulate_qam as _),
            ModulationScheme::Qam8 => (Self::modulate_qam as _, Self::demodulate_qam as _),
            ModulationScheme::Qam16 => (Self::modulate_qam as _, Self::demodulate_qam as _),
            ModulationScheme::Qam32 => (Self::modulate_qam as _, Self::demodulate_qam as _),
            ModulationScheme::Qam64 => (Self::modulate_qam as _, Self::demodulate_qam as _),
            ModulationScheme::Qam128 => (Self::modulate_qam as _, Self::demodulate_qam as _),
            ModulationScheme::Qam256 => (Self::modulate_qam as _, Self::demodulate_qam as _),
            ModulationScheme::Apsk4 => (Self::modulate_apsk as _, Self::demodulate_apsk as _),
            ModulationScheme::Apsk8 => (Self::modulate_apsk as _, Self::demodulate_apsk as _),
            ModulationScheme::Apsk16 => (Self::modulate_apsk as _, Self::demodulate_apsk as _),
            ModulationScheme::Apsk32 => (Self::modulate_apsk as _, Self::demodulate_apsk as _),
            ModulationScheme::Apsk64 => (Self::modulate_apsk as _, Self::demodulate_apsk as _),
            ModulationScheme::Apsk128 => (Self::modulate_apsk as _, Self::demodulate_apsk as _),
            ModulationScheme::Apsk256 => (Self::modulate_apsk as _, Self::demodulate_apsk as _),
            ModulationScheme::Bpsk => (Self::modulate_bpsk as _, Self::demodulate_bpsk as _),
            ModulationScheme::Qpsk => (Self::modulate_qpsk as _, Self::demodulate_qpsk as _),
            ModulationScheme::Ook => (Self::modulate_ook as _, Self::demodulate_ook as _),
            ModulationScheme::Sqam32 => (Self::modulate_sqam32 as _, Self::demodulate_sqam32 as _),
            ModulationScheme::Sqam128 => (Self::modulate_sqam128 as _, Self::demodulate_sqam128 as _),
            ModulationScheme::V29 => (Self::modulate_arb as _, Self::demodulate_arb as _),
            ModulationScheme::Arb16Opt => (Self::modulate_arb as _, Self::demodulate_arb as _),
            ModulationScheme::Arb32Opt => (Self::modulate_arb as _, Self::demodulate_arb as _),
            ModulationScheme::Arb64Opt => (Self::modulate_arb as _, Self::demodulate_arb as _),
            ModulationScheme::Arb128Opt => (Self::modulate_arb as _, Self::demodulate_arb as _),
            ModulationScheme::Arb256Opt => (Self::modulate_arb as _, Self::demodulate_arb as _),
            ModulationScheme::Arb64Vt => (Self::modulate_arb as _, Self::demodulate_arb as _),
            ModulationScheme::Arb64Ui => (Self::modulate_arb as _, Self::demodulate_arb as _),
            ModulationScheme::Pi4Dqpsk => (Self::modulate_pi4dqpsk as _, Self::demodulate_pi4dqpsk as _),
            ModulationScheme::Arb => (Self::modulate_arb as _, Self::demodulate_arb as _),
            _ => return Err(Error::Config("modulation scheme not supported".into())),
        };

        let mut modem = Modem {
            scheme,
            bits_per_symbol,
            constellation_size: 1 << bits_per_symbol,
            reference: None,
            symbol_map: None,
            modulate_using_map: false,
            modulate_func: mod_fn,
            demodulate_func: demod_fn,
            demodulate_soft_func: None,
            demod_soft_neighbors: None,
            demod_soft_p: 0,
            data: None,
            r: Complex32::new(0.0, 0.0),
            x_hat: Complex32::new(0.0, 0.0),
            randomizer: MSequence::create_default(11)?,
        };
        modem.reset();
        Ok(modem)
    }

    fn init_map(&mut self) -> Result<()> {
        if self.symbol_map.is_none() {
            return Err(Error::Internal("symbol map not initialized".into()));
        }
        if self.constellation_size == 0 || self.constellation_size > (1 << MAX_MOD_BITS_PER_SYMBOL) {
            return Err(Error::Internal("constellation size out of range".into()));
        }
        for i in 0..self.constellation_size {
            self.symbol_map.as_mut().unwrap()[i] = (self.modulate_func)(self, i as u32)?;
        }
        Ok(())
    }

    fn init_demod_soft_tab(&mut self, p: u32) -> Result<()> {
        if p > (self.constellation_size - 1) as u32 {
            return Err(Error::Internal("too many neighbors".into()));
        }
        self.demod_soft_p = p;
        self.demod_soft_neighbors = Some(vec![0; self.constellation_size * p as usize]);

        let constellation_size = self.constellation_size;
        let mut c = vec![Complex32::new(0.0, 0.0); constellation_size];
        for i in 0..constellation_size {
            c[i] = self.modulate(i as u32)?;
        }

        let neighbors = self.demod_soft_neighbors.as_mut().unwrap();

        for i in 0..constellation_size {
            for k in 0..p {
                neighbors[i * p as usize + k as usize] = constellation_size as u8;
            }
        }

        for i in 0..constellation_size {
            for k in 0..p {
                let mut dmin = 1e9f32;
                for j in 0..constellation_size {
                    let mut symbol_valid = true;
                    if i == j {
                        symbol_valid = false;
                    }

                    for l in 0..p {
                        if neighbors[i * p as usize + l as usize] == j as u8 {
                            symbol_valid = false;
                        }
                    }

                    let d = (c[i] - c[j]).norm();

                    if d < dmin && symbol_valid {
                        dmin = d;
                        neighbors[i * p as usize + k as usize] = j as u8;
                    }
                }
            }
        }
        Ok(())
    }
    
}

/// gray encoding
pub fn gray_encode(symbol_in: u32) -> u32 {
    symbol_in ^ (symbol_in >> 1)
}

/// gray decoding
pub fn gray_decode(symbol_in: u32) -> u32 {
    let mut mask = symbol_in;
    let mut symbol_out = symbol_in;

    // Run loop in blocks of 4 to reduce number of comparisons. Running
    // loop more times than MAX_MOD_BITS_PER_SYMBOL will not result in
    // decoding errors.
    for _ in (0..MAX_MOD_BITS_PER_SYMBOL).step_by(4) {
        symbol_out ^= mask >> 1;
        symbol_out ^= mask >> 2;
        symbol_out ^= mask >> 3;
        symbol_out ^= mask >> 4;
        mask >>= 4;
    }

    symbol_out
}

/// pack soft bits into symbol
///  soft_bits  :   soft input bits [size: bps x 1]
///  bps        :   bits per symbol
///  sym_out    :   output symbol, value in [0,2^bps)
pub fn pack_soft_bits(soft_bits: &[u8], bps: usize) -> Result<u32> {
    // validate input
    if bps > MAX_MOD_BITS_PER_SYMBOL {
        return Err(Error::Config(format!("pack_soft_bits(), bits/symbol exceeds maximum ({})", MAX_MOD_BITS_PER_SYMBOL)));
    }

    let mut s = 0;
    for &bit in soft_bits.iter().take(bps as usize) {
        s <<= 1;
        s |= if bit > SOFTBIT_ERASURE { 1 } else { 0 };
    }
    Ok(s)
}

/// unpack soft bits into symbol
///  sym_in     :   input symbol, value in [0,2^bps)
///  bps        :   bits per symbol
///  soft_bits  :   soft output bits [size: bps x 1]
pub fn unpack_soft_bits(sym_in: u32, bps: usize, soft_bits: &mut [u8]) -> Result<()> {
    // validate input
    if bps > MAX_MOD_BITS_PER_SYMBOL {
        return Err(Error::Config(format!("unpack_soft_bits(), bits/symbol exceeds maximum ({})", MAX_MOD_BITS_PER_SYMBOL)));
    }

    for i in 0..bps {
        soft_bits[i as usize] = if (sym_in >> (bps - i - 1)) & 0x0001 != 0 {
            SOFTBIT_1
        } else {
            SOFTBIT_0
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macro::autotest_annotate;
    use approx::assert_relative_eq;
    
    fn modemcf_test_mod_demod(ms: ModulationScheme) {
        // generate mod/demod
        let mut modem = Modem::new(ms).unwrap();
        let mut demod = Modem::new(ms).unwrap();

        // run the test
        let m = 1 << modem.get_bps();
        let mut e = 0.0f32;
        for i in 0..m {
            let x = modem.modulate(i as u32).unwrap();
            let s = demod.demodulate(x).unwrap();
            // println!("i: {}, x: {}, s: {}", i, x, s);
            assert_eq!(s, i as u32);

            assert_relative_eq!(demod.get_demodulator_phase_error(), 0.0f32, epsilon = 1e-3);
            
            assert_relative_eq!(demod.get_demodulator_evm(), 0.0f32, epsilon = 1e-3);

            e += (x * x.conj()).re;
        }
        e = (e / m as f32).sqrt();

        assert_relative_eq!(e, 1.0f32, epsilon = 1e-3);
    }

    // AUTOTESTS: generic PSK
    #[test]
    #[autotest_annotate(autotest_mod_demod_psk2)]
    fn test_mod_demod_psk2() { modemcf_test_mod_demod(ModulationScheme::Psk2); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_psk4)]
    fn test_mod_demod_psk4() { modemcf_test_mod_demod(ModulationScheme::Psk4); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_psk8)]
    fn test_mod_demod_psk8() { modemcf_test_mod_demod(ModulationScheme::Psk8); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_psk16)]
    fn test_mod_demod_psk16() { modemcf_test_mod_demod(ModulationScheme::Psk16); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_psk32)]
    fn test_mod_demod_psk32() { modemcf_test_mod_demod(ModulationScheme::Psk32); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_psk64)]
    fn test_mod_demod_psk64() { modemcf_test_mod_demod(ModulationScheme::Psk64); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_psk128)]
    fn test_mod_demod_psk128() { modemcf_test_mod_demod(ModulationScheme::Psk128); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_psk256)]
    fn test_mod_demod_psk256() { modemcf_test_mod_demod(ModulationScheme::Psk256); }

    // AUTOTESTS: generic DPSK
    #[test]
    #[autotest_annotate(autotest_mod_demod_dpsk2)]
    fn test_mod_demod_dpsk2() { modemcf_test_mod_demod(ModulationScheme::Dpsk2); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_dpsk4)]
    fn test_mod_demod_dpsk4() { modemcf_test_mod_demod(ModulationScheme::Dpsk4); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_dpsk8)]
    fn test_mod_demod_dpsk8() { modemcf_test_mod_demod(ModulationScheme::Dpsk8); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_dpsk16)]
    fn test_mod_demod_dpsk16() { modemcf_test_mod_demod(ModulationScheme::Dpsk16); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_dpsk32)]
    fn test_mod_demod_dpsk32() { modemcf_test_mod_demod(ModulationScheme::Dpsk32); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_dpsk64)]
    fn test_mod_demod_dpsk64() { modemcf_test_mod_demod(ModulationScheme::Dpsk64); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_dpsk128)]
    fn test_mod_demod_dpsk128() { modemcf_test_mod_demod(ModulationScheme::Dpsk128); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_dpsk256)]
    fn test_mod_demod_dpsk256() { modemcf_test_mod_demod(ModulationScheme::Dpsk256); }

    // AUTOTESTS: generic ASK
    #[test]
    #[autotest_annotate(autotest_mod_demod_ask2)]
    fn test_mod_demod_ask2() { modemcf_test_mod_demod(ModulationScheme::Ask2); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_ask4)]
    fn test_mod_demod_ask4() { modemcf_test_mod_demod(ModulationScheme::Ask4); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_ask8)]
    fn test_mod_demod_ask8() { modemcf_test_mod_demod(ModulationScheme::Ask8); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_ask16)]
    fn test_mod_demod_ask16() { modemcf_test_mod_demod(ModulationScheme::Ask16); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_ask32)]
    fn test_mod_demod_ask32() { modemcf_test_mod_demod(ModulationScheme::Ask32); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_ask64)]
    fn test_mod_demod_ask64() { modemcf_test_mod_demod(ModulationScheme::Ask64); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_ask128)]
    fn test_mod_demod_ask128() { modemcf_test_mod_demod(ModulationScheme::Ask128); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_ask256)]
    fn test_mod_demod_ask256() { modemcf_test_mod_demod(ModulationScheme::Ask256); }

    // AUTOTESTS: generic QAM
    #[test]
    #[autotest_annotate(autotest_mod_demod_qam4)]
    fn test_mod_demod_qam4() { modemcf_test_mod_demod(ModulationScheme::Qam4); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_qam8)]
    fn test_mod_demod_qam8() { modemcf_test_mod_demod(ModulationScheme::Qam8); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_qam16)]
    fn test_mod_demod_qam16() { modemcf_test_mod_demod(ModulationScheme::Qam16); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_qam32)]
    fn test_mod_demod_qam32() { modemcf_test_mod_demod(ModulationScheme::Qam32); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_qam64)]
    fn test_mod_demod_qam64() { modemcf_test_mod_demod(ModulationScheme::Qam64); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_qam128)]
    fn test_mod_demod_qam128() { modemcf_test_mod_demod(ModulationScheme::Qam128); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_qam256)]
    fn test_mod_demod_qam256() { modemcf_test_mod_demod(ModulationScheme::Qam256); }

    // AUTOTESTS: generic APSK (maps to specific APSK modems internally)
    #[test]
    #[autotest_annotate(autotest_mod_demod_apsk4)]
    fn test_mod_demod_apsk4() { modemcf_test_mod_demod(ModulationScheme::Apsk4); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_apsk8)]
    fn test_mod_demod_apsk8() { modemcf_test_mod_demod(ModulationScheme::Apsk8); }
    
    #[test]
    #[autotest_annotate(autotest_mod_demod_apsk16)]
    fn test_mod_demod_apsk16() { modemcf_test_mod_demod(ModulationScheme::Apsk16); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_apsk32)]
    fn test_mod_demod_apsk32() { modemcf_test_mod_demod(ModulationScheme::Apsk32); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_apsk64)]
    fn test_mod_demod_apsk64() { modemcf_test_mod_demod(ModulationScheme::Apsk64); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_apsk128)]
    fn test_mod_demod_apsk128() { modemcf_test_mod_demod(ModulationScheme::Apsk128); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_apsk256)]
    fn test_mod_demod_apsk256() { modemcf_test_mod_demod(ModulationScheme::Apsk256); }

    // AUTOTESTS: Specific modems
    #[test]
    #[autotest_annotate(autotest_mod_demod_bpsk)]
    fn test_mod_demod_bpsk() { modemcf_test_mod_demod(ModulationScheme::Bpsk); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_qpsk)]
    fn test_mod_demod_qpsk() { modemcf_test_mod_demod(ModulationScheme::Qpsk); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_ook)]
    fn test_mod_demod_ook() { modemcf_test_mod_demod(ModulationScheme::Ook); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_sqam32)]
    fn test_mod_demod_sqam32() { modemcf_test_mod_demod(ModulationScheme::Sqam32); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_sqam128)]
    fn test_mod_demod_sqam128() { modemcf_test_mod_demod(ModulationScheme::Sqam128); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_V29)]
    fn test_mod_demod_v29() { modemcf_test_mod_demod(ModulationScheme::V29); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_arb16opt)]
    fn test_mod_demod_arb16opt() { modemcf_test_mod_demod(ModulationScheme::Arb16Opt); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_arb32opt)]
    fn test_mod_demod_arb32opt() { modemcf_test_mod_demod(ModulationScheme::Arb32Opt); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_arb64opt)]
    fn test_mod_demod_arb64opt() { modemcf_test_mod_demod(ModulationScheme::Arb64Opt); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_arb128opt)]
    fn test_mod_demod_arb128opt() { modemcf_test_mod_demod(ModulationScheme::Arb128Opt); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_arb256opt)]
    fn test_mod_demod_arb256opt() { modemcf_test_mod_demod(ModulationScheme::Arb256Opt); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_arb64vt)]
    fn test_mod_demod_arb64vt() { modemcf_test_mod_demod(ModulationScheme::Arb64Vt); }

    #[test]
    fn test_mod_demod_arb64ui() { modemcf_test_mod_demod(ModulationScheme::Arb64Ui); }

    #[test]
    #[autotest_annotate(autotest_mod_demod_pi4dqpsk)]
    fn test_mod_demod_pi4dqpsk() { modemcf_test_mod_demod(ModulationScheme::Pi4Dqpsk); }

    fn modemcf_test_demodsoft(ms: ModulationScheme) {
        // generate mod/demod
        let mut modulator = Modem::new(ms).unwrap();
        let mut demodulator = Modem::new(ms).unwrap();

        // get bits per symbol
        let bps = demodulator.get_bps();

        // run the test
        let m = 1 << bps;
        let mut soft_bits = vec![0u8; bps];
        
        for i in 0..m {
            // modulate symbol
            let x = modulator.modulate(i).unwrap();

            // demodulate using soft-decision
            let s = demodulator.demodulate_soft(x, &mut soft_bits).unwrap();

            // check hard-decision output
            assert_eq!(s, i);

            // check soft bits
            let sym_soft = pack_soft_bits(&soft_bits, bps).unwrap();
            assert_eq!(sym_soft, i);

            // check phase error, evm, etc.
            assert_relative_eq!(demodulator.get_demodulator_phase_error(), 0.0, epsilon = 1e-3);
            assert_relative_eq!(demodulator.get_demodulator_evm(), 0.0, epsilon = 1e-3);
        }
    }

    // AUTOTESTS: generic PSK
    #[test]
    #[autotest_annotate(autotest_demodsoft_psk2)]
    fn test_demodsoft_psk2() { modemcf_test_demodsoft(ModulationScheme::Psk2); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_psk4)]
    fn test_demodsoft_psk4() { modemcf_test_demodsoft(ModulationScheme::Psk4); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_psk8)]
    fn test_demodsoft_psk8() { modemcf_test_demodsoft(ModulationScheme::Psk8); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_psk16)]
    fn test_demodsoft_psk16() { modemcf_test_demodsoft(ModulationScheme::Psk16); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_psk32)]
    fn test_demodsoft_psk32() { modemcf_test_demodsoft(ModulationScheme::Psk32); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_psk64)]
    fn test_demodsoft_psk64() { modemcf_test_demodsoft(ModulationScheme::Psk64); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_psk128)]
    fn test_demodsoft_psk128() { modemcf_test_demodsoft(ModulationScheme::Psk128); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_psk256)]
    fn test_demodsoft_psk256() { modemcf_test_demodsoft(ModulationScheme::Psk256); }

    // AUTOTESTS: generic DPSK
    #[test]
    #[autotest_annotate(autotest_demodsoft_dpsk2)]
    fn test_demodsoft_dpsk2() { modemcf_test_demodsoft(ModulationScheme::Dpsk2); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_dpsk4)]
    fn test_demodsoft_dpsk4() { modemcf_test_demodsoft(ModulationScheme::Dpsk4); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_dpsk8)]
    fn test_demodsoft_dpsk8() { modemcf_test_demodsoft(ModulationScheme::Dpsk8); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_dpsk16)]
    fn test_demodsoft_dpsk16() { modemcf_test_demodsoft(ModulationScheme::Dpsk16); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_dpsk32)]
    fn test_demodsoft_dpsk32() { modemcf_test_demodsoft(ModulationScheme::Dpsk32); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_dpsk64)]
    fn test_demodsoft_dpsk64() { modemcf_test_demodsoft(ModulationScheme::Dpsk64); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_dpsk128)]
    fn test_demodsoft_dpsk128() { modemcf_test_demodsoft(ModulationScheme::Dpsk128); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_dpsk256)]
    fn test_demodsoft_dpsk256() { modemcf_test_demodsoft(ModulationScheme::Dpsk256); }

    // AUTOTESTS: generic ASK
    #[test]
    #[autotest_annotate(autotest_demodsoft_ask2)]
    fn test_demodsoft_ask2() { modemcf_test_demodsoft(ModulationScheme::Ask2); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_ask4)]
    fn test_demodsoft_ask4() { modemcf_test_demodsoft(ModulationScheme::Ask4); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_ask8)]
    fn test_demodsoft_ask8() { modemcf_test_demodsoft(ModulationScheme::Ask8); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_ask16)]
    fn test_demodsoft_ask16() { modemcf_test_demodsoft(ModulationScheme::Ask16); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_ask32)]
    fn test_demodsoft_ask32() { modemcf_test_demodsoft(ModulationScheme::Ask32); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_ask64)]
    fn test_demodsoft_ask64() { modemcf_test_demodsoft(ModulationScheme::Ask64); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_ask128)]
    fn test_demodsoft_ask128() { modemcf_test_demodsoft(ModulationScheme::Ask128); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_ask256)]
    fn test_demodsoft_ask256() { modemcf_test_demodsoft(ModulationScheme::Ask256); }

    // AUTOTESTS: generic QAM
    #[test]
    #[autotest_annotate(autotest_demodsoft_qam4)]
    fn test_demodsoft_qam4() { modemcf_test_demodsoft(ModulationScheme::Qam4); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_qam8)]
    fn test_demodsoft_qam8() { modemcf_test_demodsoft(ModulationScheme::Qam8); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_qam16)]
    fn test_demodsoft_qam16() { modemcf_test_demodsoft(ModulationScheme::Qam16); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_qam32)]
    fn test_demodsoft_qam32() { modemcf_test_demodsoft(ModulationScheme::Qam32); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_qam64)]
    fn test_demodsoft_qam64() { modemcf_test_demodsoft(ModulationScheme::Qam64); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_qam128)]
    fn test_demodsoft_qam128() { modemcf_test_demodsoft(ModulationScheme::Qam128); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_qam256)]
    fn test_demodsoft_qam256() { modemcf_test_demodsoft(ModulationScheme::Qam256); }

    // AUTOTESTS: generic APSK (maps to specific APSK modems internally)
    #[test]
    #[autotest_annotate(autotest_demodsoft_apsk4)]
    fn test_demodsoft_apsk4() { modemcf_test_demodsoft(ModulationScheme::Apsk4); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_apsk8)]
    fn test_demodsoft_apsk8() { modemcf_test_demodsoft(ModulationScheme::Apsk8); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_apsk16)]
    fn test_demodsoft_apsk16() { modemcf_test_demodsoft(ModulationScheme::Apsk16); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_apsk32)]
    fn test_demodsoft_apsk32() { modemcf_test_demodsoft(ModulationScheme::Apsk32); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_apsk64)]
    fn test_demodsoft_apsk64() { modemcf_test_demodsoft(ModulationScheme::Apsk64); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_apsk128)]
    fn test_demodsoft_apsk128() { modemcf_test_demodsoft(ModulationScheme::Apsk128); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_apsk256)]
    fn test_demodsoft_apsk256() { modemcf_test_demodsoft(ModulationScheme::Apsk256); }

    // AUTOTESTS: Specific modems
    #[test]
    #[autotest_annotate(autotest_demodsoft_bpsk)]
    fn test_demodsoft_bpsk() { modemcf_test_demodsoft(ModulationScheme::Bpsk); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_qpsk)]
    fn test_demodsoft_qpsk() { modemcf_test_demodsoft(ModulationScheme::Qpsk); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_ook)]
    fn test_demodsoft_ook() { modemcf_test_demodsoft(ModulationScheme::Ook); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_sqam32)]
    fn test_demodsoft_sqam32() { modemcf_test_demodsoft(ModulationScheme::Sqam32); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_sqam128)]
    fn test_demodsoft_sqam128() { modemcf_test_demodsoft(ModulationScheme::Sqam128); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_V29)]
    fn test_demodsoft_v29() { modemcf_test_demodsoft(ModulationScheme::V29); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_arb16opt)]
    fn test_demodsoft_arb16opt() { modemcf_test_demodsoft(ModulationScheme::Arb16Opt); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_arb32opt)]
    fn test_demodsoft_arb32opt() { modemcf_test_demodsoft(ModulationScheme::Arb32Opt); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_arb64opt)]
    fn test_demodsoft_arb64opt() { modemcf_test_demodsoft(ModulationScheme::Arb64Opt); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_arb128opt)]
    fn test_demodsoft_arb128opt() { modemcf_test_demodsoft(ModulationScheme::Arb128Opt); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_arb256opt)]
    fn test_demodsoft_arb256opt() { modemcf_test_demodsoft(ModulationScheme::Arb256Opt); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_arb64vt)]
    fn test_demodsoft_arb64vt() { modemcf_test_demodsoft(ModulationScheme::Arb64Vt); }

    #[test]
    fn test_demodsoft_arb64ui() { modemcf_test_demodsoft(ModulationScheme::Arb64Ui); }

    #[test]
    #[autotest_annotate(autotest_demodsoft_pi4dqpsk)]
    fn test_demodsoft_pi4dqpsk() { modemcf_test_demodsoft(ModulationScheme::Pi4Dqpsk); }

    fn modemcf_test_demodstats(ms: ModulationScheme) {
        // generate mod/demod
        let mut modulator = Modem::new(ms).unwrap();
        let mut demodulator = Modem::new(ms).unwrap();

        // run the test
        let m = modulator.get_bps();
        let constellation_size = 1 << m;
        let phi = 0.01f32;

        for i in 0..constellation_size {
            // reset modem objects
            modulator.reset();
            demodulator.reset();

            // modulate symbol
            let x = modulator.modulate(i).unwrap();

            // ignore rare condition where modulated symbol is (0,0)
            // (e.g. APSK-8)
            if x.norm() < 1e-3f32 {
                continue;
            }

            // add phase offsets
            let x_hat = x * Complex32::from_polar(1.0, phi);

            // demod positive phase signal, and ensure demodulator
            // maps to appropriate symbol
            let s = demodulator.demodulate(x_hat).unwrap();
            // nb this assert isn't in original autotests
            assert_eq!(s, i);
            if s != i {
                println!("Warning: modem_test_demodstats(), output symbol does not match");
            }

            let demodstats = demodulator.get_demodulator_phase_error();
            assert!(demodstats > 0.0);
        }

        // repeat with negative phase error
        for i in 0..constellation_size {
            // reset modem objects
            modulator.reset();
            demodulator.reset();

            // modulate symbol
            let x = modulator.modulate(i).unwrap();

            // ignore rare condition where modulated symbol is (0,0)
            // (e.g. APSK-8)
            if x.norm() < 1e-3f32 {
                continue;
            }

            // add phase offsets
            let x_hat = x * Complex32::from_polar(1.0, -phi);

            // demod positive phase signal, and ensure demodulator
            // maps to appropriate symbol
            let s = demodulator.demodulate(x_hat).unwrap();
            // nb this assert isn't in original autotests
            assert_eq!(s, i);
            if s != i {
                println!("Warning: modem_test_demodstats(), output symbol does not match");
            }

            let demodstats = demodulator.get_demodulator_phase_error();
            assert!(demodstats < 0.0);
        }
    }

    // AUTOTESTS: generic PSK
    #[test]
    #[autotest_annotate(autotest_demodstats_psk2)]
    fn test_demodstats_psk2() { modemcf_test_demodstats(ModulationScheme::Psk2); }

    #[test]
    #[autotest_annotate(autotest_demodstats_psk4)]
    fn test_demodstats_psk4() { modemcf_test_demodstats(ModulationScheme::Psk4); }

    #[test]
    #[autotest_annotate(autotest_demodstats_psk8)]
    fn test_demodstats_psk8() { modemcf_test_demodstats(ModulationScheme::Psk8); }

    #[test]
    #[autotest_annotate(autotest_demodstats_psk16)]
    fn test_demodstats_psk16() { modemcf_test_demodstats(ModulationScheme::Psk16); }

    #[test]
    #[autotest_annotate(autotest_demodstats_psk32)]
    fn test_demodstats_psk32() { modemcf_test_demodstats(ModulationScheme::Psk32); }

    #[test]
    #[autotest_annotate(autotest_demodstats_psk64)]
    fn test_demodstats_psk64() { modemcf_test_demodstats(ModulationScheme::Psk64); }

    #[test]
    #[autotest_annotate(autotest_demodstats_psk128)]
    fn test_demodstats_psk128() { modemcf_test_demodstats(ModulationScheme::Psk128); }

    #[test]
    #[autotest_annotate(autotest_demodstats_psk256)]
    fn test_demodstats_psk256() { modemcf_test_demodstats(ModulationScheme::Psk256); }

    // AUTOTESTS: generic DPSK
    #[test]
    #[autotest_annotate(autotest_demodstats_dpsk2)]
    fn test_demodstats_dpsk2() { modemcf_test_demodstats(ModulationScheme::Dpsk2); }

    #[test]
    #[autotest_annotate(autotest_demodstats_dpsk4)]
    fn test_demodstats_dpsk4() { modemcf_test_demodstats(ModulationScheme::Dpsk4); }

    #[test]
    #[autotest_annotate(autotest_demodstats_dpsk8)]
    fn test_demodstats_dpsk8() { modemcf_test_demodstats(ModulationScheme::Dpsk8); }

    #[test]
    #[autotest_annotate(autotest_demodstats_dpsk16)]
    fn test_demodstats_dpsk16() { modemcf_test_demodstats(ModulationScheme::Dpsk16); }

    #[test]
    #[autotest_annotate(autotest_demodstats_dpsk32)]
    fn test_demodstats_dpsk32() { modemcf_test_demodstats(ModulationScheme::Dpsk32); }

    #[test]
    #[autotest_annotate(autotest_demodstats_dpsk64)]
    fn test_demodstats_dpsk64() { modemcf_test_demodstats(ModulationScheme::Dpsk64); }

    #[test]
    #[autotest_annotate(autotest_demodstats_dpsk128)]
    fn test_demodstats_dpsk128() { modemcf_test_demodstats(ModulationScheme::Dpsk128); }

    #[test]
    #[autotest_annotate(autotest_demodstats_dpsk256)]
    fn test_demodstats_dpsk256() { modemcf_test_demodstats(ModulationScheme::Dpsk256); }

    // AUTOTESTS: generic ASK
    #[test]
    #[autotest_annotate(autotest_demodstats_ask2)]
    fn test_demodstats_ask2() { modemcf_test_demodstats(ModulationScheme::Ask2); }

    #[test]
    #[autotest_annotate(autotest_demodstats_ask4)]
    fn test_demodstats_ask4() { modemcf_test_demodstats(ModulationScheme::Ask4); }

    #[test]
    #[autotest_annotate(autotest_demodstats_ask8)]
    fn test_demodstats_ask8() { modemcf_test_demodstats(ModulationScheme::Ask8); }

    #[test]
    #[autotest_annotate(autotest_demodstats_ask16)]
    fn test_demodstats_ask16() { modemcf_test_demodstats(ModulationScheme::Ask16); }

    #[test]
    #[autotest_annotate(autotest_demodstats_ask32)]
    fn test_demodstats_ask32() { modemcf_test_demodstats(ModulationScheme::Ask32); }

    #[test]
    #[autotest_annotate(autotest_demodstats_ask64)]
    fn test_demodstats_ask64() { modemcf_test_demodstats(ModulationScheme::Ask64); }

    #[test]
    #[autotest_annotate(autotest_demodstats_ask128)]
    fn test_demodstats_ask128() { modemcf_test_demodstats(ModulationScheme::Ask128); }

    #[test]
    #[autotest_annotate(autotest_demodstats_ask256)]
    fn test_demodstats_ask256() { modemcf_test_demodstats(ModulationScheme::Ask256); }

    // AUTOTESTS: generic QAM
    #[test]
    #[autotest_annotate(autotest_demodstats_qam4)]
    fn test_demodstats_qam4() { modemcf_test_demodstats(ModulationScheme::Qam4); }

    #[test]
    #[autotest_annotate(autotest_demodstats_qam8)]
    fn test_demodstats_qam8() { modemcf_test_demodstats(ModulationScheme::Qam8); }

    #[test]
    #[autotest_annotate(autotest_demodstats_qam16)]
    fn test_demodstats_qam16() { modemcf_test_demodstats(ModulationScheme::Qam16); }

    #[test]
    #[autotest_annotate(autotest_demodstats_qam32)]
    fn test_demodstats_qam32() { modemcf_test_demodstats(ModulationScheme::Qam32); }

    #[test]
    #[autotest_annotate(autotest_demodstats_qam64)]
    fn test_demodstats_qam64() { modemcf_test_demodstats(ModulationScheme::Qam64); }

    #[test]
    #[autotest_annotate(autotest_demodstats_qam128)]
    fn test_demodstats_qam128() { modemcf_test_demodstats(ModulationScheme::Qam128); }

    #[test]
    #[autotest_annotate(autotest_demodstats_qam256)]
    fn test_demodstats_qam256() { modemcf_test_demodstats(ModulationScheme::Qam256); }

    // AUTOTESTS: generic APSK (maps to specific APSK modems internally)
    #[test]
    #[autotest_annotate(autotest_demodstats_apsk4)]
    fn test_demodstats_apsk4() { modemcf_test_demodstats(ModulationScheme::Apsk4); }

    #[test]
    #[autotest_annotate(autotest_demodstats_apsk8)]
    fn test_demodstats_apsk8() { modemcf_test_demodstats(ModulationScheme::Apsk8); }

    #[test]
    #[autotest_annotate(autotest_demodstats_apsk16)]
    fn test_demodstats_apsk16() { modemcf_test_demodstats(ModulationScheme::Apsk16); }

    #[test]
    #[autotest_annotate(autotest_demodstats_apsk32)]
    fn test_demodstats_apsk32() { modemcf_test_demodstats(ModulationScheme::Apsk32); }

    #[test]
    #[autotest_annotate(autotest_demodstats_apsk64)]
    fn test_demodstats_apsk64() { modemcf_test_demodstats(ModulationScheme::Apsk64); }

    #[test]
    #[autotest_annotate(autotest_demodstats_apsk128)]
    fn test_demodstats_apsk128() { modemcf_test_demodstats(ModulationScheme::Apsk128); }

    #[test]
    #[autotest_annotate(autotest_demodstats_apsk256)]
    fn test_demodstats_apsk256() { modemcf_test_demodstats(ModulationScheme::Apsk256); }

    // AUTOTESTS: Specific modems
    #[test]
    #[autotest_annotate(autotest_demodstats_bpsk)]
    fn test_demodstats_bpsk() { modemcf_test_demodstats(ModulationScheme::Bpsk); }

    #[test]
    #[autotest_annotate(autotest_demodstats_qpsk)]
    fn test_demodstats_qpsk() { modemcf_test_demodstats(ModulationScheme::Qpsk); }
    
    #[test]
    #[autotest_annotate(autotest_demodstats_ook)]
    fn test_demodstats_ook() { modemcf_test_demodstats(ModulationScheme::Ook); }

    #[test]
    #[autotest_annotate(autotest_demodstats_sqam32)]
    fn test_demodstats_sqam32() { modemcf_test_demodstats(ModulationScheme::Sqam32); }

    #[test]
    #[autotest_annotate(autotest_demodstats_sqam128)]
    fn test_demodstats_sqam128() { modemcf_test_demodstats(ModulationScheme::Sqam128); }

    #[test]
    #[autotest_annotate(autotest_demodstats_V29)]
    fn test_demodstats_v29() { modemcf_test_demodstats(ModulationScheme::V29); }

    #[test]
    #[autotest_annotate(autotest_demodstats_arb16opt)]
    fn test_demodstats_arb16opt() { modemcf_test_demodstats(ModulationScheme::Arb16Opt); }

    #[test]
    #[autotest_annotate(autotest_demodstats_arb32opt)]
    fn test_demodstats_arb32opt() { modemcf_test_demodstats(ModulationScheme::Arb32Opt); }
    
    #[test]
    #[autotest_annotate(autotest_demodstats_arb64opt)]
    fn test_demodstats_arb64opt() { modemcf_test_demodstats(ModulationScheme::Arb64Opt); }

    #[test]
    #[autotest_annotate(autotest_demodstats_arb128opt)]
    fn test_demodstats_arb128opt() { modemcf_test_demodstats(ModulationScheme::Arb128Opt); }

    #[test]
    #[autotest_annotate(autotest_demodstats_arb256opt)]
    fn test_demodstats_arb256opt() { modemcf_test_demodstats(ModulationScheme::Arb256Opt); }

    #[test]
    #[autotest_annotate(autotest_demodstats_arb64vt)]
    fn test_demodstats_arb64vt() { modemcf_test_demodstats(ModulationScheme::Arb64Vt); }

    #[test]
    fn test_demodstats_arb64ui() { modemcf_test_demodstats(ModulationScheme::Arb64Ui); }

    #[test]
    #[autotest_annotate(autotest_demodstats_pi4dqpsk)]
    fn test_demodstats_pi4dqpsk() { modemcf_test_demodstats(ModulationScheme::Pi4Dqpsk); }

    fn modemcf_test_copy(ms: ModulationScheme) {
        // create modem and randomize internal state
        let mut modem_0 = Modem::new(ms).unwrap();
        let m = 1 << modem_0.get_bps();
        
        for _ in 0..10 {
            // modulate random symbol
            let _ = modem_0.modulate(rand::random::<u32>() % m);

            // demodulate random sample
            let _ = modem_0.demodulate(Complex32::new(crate::random::randnf(), crate::random::randnf()));
        }

        // copy modem
        let mut modem_1 = modem_0.clone();

        // ...
        for _ in 0..10 {
            // modulate random symbol
            let s = rand::random::<u32>() % m;
            let x0 = modem_0.modulate(s).unwrap();
            let x1 = modem_1.modulate(s).unwrap();
            assert_eq!(x0, x1);

            // demodulate random sample
            let x = Complex32::new(crate::random::randnf(), crate::random::randnf());
            let s0 = modem_0.demodulate(x).unwrap();
            let s1 = modem_1.demodulate(x).unwrap();
            assert_eq!(s0, s1);
        }
    }

    // AUTOTESTS: generic PSK
    #[test]
    #[autotest_annotate(autotest_modem_copy_psk2)]
    fn test_modem_copy_psk2() { modemcf_test_copy(ModulationScheme::Psk2); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_psk4)]
    fn test_modem_copy_psk4() { modemcf_test_copy(ModulationScheme::Psk4); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_psk8)]
    fn test_modem_copy_psk8() { modemcf_test_copy(ModulationScheme::Psk8); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_psk16)]
    fn test_modem_copy_psk16() { modemcf_test_copy(ModulationScheme::Psk16); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_psk32)]
    fn test_modem_copy_psk32() { modemcf_test_copy(ModulationScheme::Psk32); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_psk64)]
    fn test_modem_copy_psk64() { modemcf_test_copy(ModulationScheme::Psk64); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_psk128)]
    fn test_modem_copy_psk128() { modemcf_test_copy(ModulationScheme::Psk128); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_psk256)]
    fn test_modem_copy_psk256() { modemcf_test_copy(ModulationScheme::Psk256); }

    // AUTOTESTS: generic DPSK
    #[test]
    #[autotest_annotate(autotest_modem_copy_dpsk2)]
    fn test_modem_copy_dpsk2() { modemcf_test_copy(ModulationScheme::Dpsk2); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_dpsk4)]
    fn test_modem_copy_dpsk4() { modemcf_test_copy(ModulationScheme::Dpsk4); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_dpsk8)]
    fn test_modem_copy_dpsk8() { modemcf_test_copy(ModulationScheme::Dpsk8); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_dpsk16)]
    fn test_modem_copy_dpsk16() { modemcf_test_copy(ModulationScheme::Dpsk16); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_dpsk32)]
    fn test_modem_copy_dpsk32() { modemcf_test_copy(ModulationScheme::Dpsk32); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_dpsk64)]
    fn test_modem_copy_dpsk64() { modemcf_test_copy(ModulationScheme::Dpsk64); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_dpsk128)]
    fn test_modem_copy_dpsk128() { modemcf_test_copy(ModulationScheme::Dpsk128); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_dpsk256)]
    fn test_modem_copy_dpsk256() { modemcf_test_copy(ModulationScheme::Dpsk256); }

    // AUTOTESTS: generic ASK
    #[test]
    #[autotest_annotate(autotest_modem_copy_ask2)]
    fn test_modem_copy_ask2() { modemcf_test_copy(ModulationScheme::Ask2); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_ask4)]
    fn test_modem_copy_ask4() { modemcf_test_copy(ModulationScheme::Ask4); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_ask8)]
    fn test_modem_copy_ask8() { modemcf_test_copy(ModulationScheme::Ask8); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_ask16)]
    fn test_modem_copy_ask16() { modemcf_test_copy(ModulationScheme::Ask16); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_ask32)]
    fn test_modem_copy_ask32() { modemcf_test_copy(ModulationScheme::Ask32); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_ask64)]
    fn test_modem_copy_ask64() { modemcf_test_copy(ModulationScheme::Ask64); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_ask128)]
    fn test_modem_copy_ask128() { modemcf_test_copy(ModulationScheme::Ask128); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_ask256)]
    fn test_modem_copy_ask256() { modemcf_test_copy(ModulationScheme::Ask256); }

    // AUTOTESTS: generic QAM
    #[test]
    #[autotest_annotate(autotest_modem_copy_qam4)]
    fn test_modem_copy_qam4() { modemcf_test_copy(ModulationScheme::Qam4); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_qam8)]
    fn test_modem_copy_qam8() { modemcf_test_copy(ModulationScheme::Qam8); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_qam16)]
    fn test_modem_copy_qam16() { modemcf_test_copy(ModulationScheme::Qam16); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_qam32)]
    fn test_modem_copy_qam32() { modemcf_test_copy(ModulationScheme::Qam32); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_qam64)]
    fn test_modem_copy_qam64() { modemcf_test_copy(ModulationScheme::Qam64); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_qam128)]
    fn test_modem_copy_qam128() { modemcf_test_copy(ModulationScheme::Qam128); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_qam256)]
    fn test_modem_copy_qam256() { modemcf_test_copy(ModulationScheme::Qam256); }

    // AUTOTESTS: generic APSK (maps to specific APSK modems internally)
    #[test]
    #[autotest_annotate(autotest_modem_copy_apsk4)]
    fn test_modem_copy_apsk4() { modemcf_test_copy(ModulationScheme::Apsk4); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_apsk8)]
    fn test_modem_copy_apsk8() { modemcf_test_copy(ModulationScheme::Apsk8); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_apsk16)]
    fn test_modem_copy_apsk16() { modemcf_test_copy(ModulationScheme::Apsk16); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_apsk32)]
    fn test_modem_copy_apsk32() { modemcf_test_copy(ModulationScheme::Apsk32); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_apsk64)]
    fn test_modem_copy_apsk64() { modemcf_test_copy(ModulationScheme::Apsk64); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_apsk128)]
    fn test_modem_copy_apsk128() { modemcf_test_copy(ModulationScheme::Apsk128); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_apsk256)]
    fn test_modem_copy_apsk256() { modemcf_test_copy(ModulationScheme::Apsk256); }

    // AUTOTESTS: Specific modems
    #[test]
    #[autotest_annotate(autotest_modem_copy_bpsk)]
    fn test_modem_copy_bpsk() { modemcf_test_copy(ModulationScheme::Bpsk); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_qpsk)]
    fn test_modem_copy_qpsk() { modemcf_test_copy(ModulationScheme::Qpsk); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_ook)]
    fn test_modem_copy_ook() { modemcf_test_copy(ModulationScheme::Ook); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_sqam32)]
    fn test_modem_copy_sqam32() { modemcf_test_copy(ModulationScheme::Sqam32); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_sqam128)]
    fn test_modem_copy_sqam128() { modemcf_test_copy(ModulationScheme::Sqam128); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_V29)]
    fn test_modem_copy_v29() { modemcf_test_copy(ModulationScheme::V29); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_arb16opt)]
    fn test_modem_copy_arb16opt() { modemcf_test_copy(ModulationScheme::Arb16Opt); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_arb32opt)]
    fn test_modem_copy_arb32opt() { modemcf_test_copy(ModulationScheme::Arb32Opt); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_arb64opt)]
    fn test_modem_copy_arb64opt() { modemcf_test_copy(ModulationScheme::Arb64Opt); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_arb128opt)]
    fn test_modem_copy_arb128opt() { modemcf_test_copy(ModulationScheme::Arb128Opt); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_arb256opt)]
    fn test_modem_copy_arb256opt() { modemcf_test_copy(ModulationScheme::Arb256Opt); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_arb64vt)]
    fn test_modem_copy_arb64vt() { modemcf_test_copy(ModulationScheme::Arb64Vt); }

    #[test]
    fn test_modem_copy_arb64ui() { modemcf_test_copy(ModulationScheme::Arb64Ui); }

    #[test]
    #[autotest_annotate(autotest_modem_copy_pi4dqpsk)]
    fn test_modem_copy_pi4dqpsk() { modemcf_test_copy(ModulationScheme::Pi4Dqpsk); }

}
