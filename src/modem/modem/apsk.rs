use crate::modem::modem::*;

#[derive(Debug, Clone)]
pub(super) struct Apsk {
    num_levels: usize,    // number of levels
    p: &'static [usize],   // number of symbols per level
    r: &'static [f32],    // radii of levels
    r_slicer: &'static [f32], // slicer radii of levels
    phi: &'static [f32],   // phase offset of levels
    map: &'static [u8],   // symbol mapping (allocated)
}

impl Modem {
    pub(super) fn new_apsk(scheme: ModulationScheme) -> Result<Self> {
        let bits_per_symbol = scheme.bits_per_symbol();
        let apsk_def = match bits_per_symbol {
            2 => &APSK4,
            3 => &APSK8,
            4 => &APSK16,
            5 => &APSK32,
            6 => &APSK64,
            7 => &APSK128,
            8 => &APSK256,
            _ => unreachable!(),
        };

        debug_assert_eq!(apsk_def.modulation, scheme);

        let mut modem = Modem::_new(bits_per_symbol, apsk_def.modulation)?;

        let data = Apsk {
            num_levels: apsk_def.num_levels,
            p: apsk_def.p,
            r: apsk_def.r,
            phi: apsk_def.phi,
            r_slicer: apsk_def.r_slicer,
            map: apsk_def.map,
        };

        modem.data = Some(ModemData::Apsk(data));

        match modem.bits_per_symbol {
            2 => modem.init_demod_soft_tab(3)?,
            3 => modem.init_demod_soft_tab(3)?,
            4 => modem.init_demod_soft_tab(4)?,
            5 => modem.init_demod_soft_tab(4)?,
            6 => modem.init_demod_soft_tab(4)?,
            7 => modem.init_demod_soft_tab(5)?,
            8 => modem.init_demod_soft_tab(5)?,
            _ => return Err(Error::Internal("Unsupported bits per symbol".into())),
        }

        modem.symbol_map = Some(vec![Complex32::new(0.0, 0.0); modem.constellation_size]);
        modem.init_map()?;

        Ok(modem)
    }

    pub(super) fn modulate_apsk(&mut self, symbol_in: u32) -> Result<Complex32> {
        if let Some(ModemData::Apsk(apsk)) = &self.data {
            let s = apsk.map[symbol_in as usize] as usize;

            let mut p = 0;
            let mut t = 0;
            for i in 0..apsk.num_levels {
                if s < t + apsk.p[i] {
                    p = i;
                    break;
                }
                t += apsk.p[i];
            }
            let s0 = s - t;
            let s1 = apsk.p[p];

            let r = apsk.r[p];
            let phi = apsk.phi[p] + (s0 as f32) * 2.0 * PI / (s1 as f32);

            let symbol_out = Complex32::from_polar(r, phi);
            return Ok(symbol_out);
        } else {
            return Err(Error::Internal("Apsk data not initialized".into()));
        }
    }

    pub(super) fn demodulate_apsk(&mut self, symbol_in: Complex32) -> Result<u32> {
        if let Some(ModemData::Apsk(apsk)) = &self.data {
            let r = symbol_in.norm();
            let mut p = 0;
            for i in 0..apsk.num_levels - 1 {
                if r < apsk.r_slicer[i] {
                    p = i;
                    break;
                } else {
                    p = apsk.num_levels - 1;
                }
            }

            let mut theta = symbol_in.arg();
            if theta < 0.0 {
                theta += 2.0 * PI;
            }

            let dphi = 2.0 * PI / (apsk.p[p] as f32);
            let mut s_hat = ((theta - apsk.phi[p]) / dphi).round() as usize;
            s_hat %= apsk.p[p];

            for i in 0..p {
                s_hat += apsk.p[i];
            }

            let mut s_prime = 0;
            for i in 0..apsk.map.len() {
                if apsk.map[i] == s_hat as u8 {
                    s_prime = i;
                    break;
                }
            }

            self.x_hat = self.modulate(s_prime as u32)?;
            self.r = symbol_in;

            return Ok(s_prime as u32);
        } else {
            return Err(Error::Internal("Apsk data not initialized".into()));
        }
    }
}

struct ApskDef {
    modulation: ModulationScheme,
    num_levels: usize,
    p: &'static [usize],
    r: &'static [f32],
    phi: &'static [f32],
    r_slicer: &'static [f32],
    map: &'static [u8],
}

// APSK4(1,3)
const APSK4_NUM_LEVELS: usize = 2;
const APSK4_P: [usize; 2] = [1, 3];
const APSK4_R: [f32; 2] = [0.0, 1.15470052];
const APSK4_PHI: [f32; 2] = [0.0, 0.0];
const APSK4_R_SLICER: [f32; 1] = [0.57735026];
const APSK4_MAP: [u8; 4] = [3, 2, 1, 0];
const APSK4: ApskDef = ApskDef {
    modulation: ModulationScheme::Apsk4,
    num_levels: APSK4_NUM_LEVELS,
    p: &APSK4_P,
    r: &APSK4_R,
    phi: &APSK4_PHI,
    r_slicer: &APSK4_R_SLICER,
    map: &APSK4_MAP,
};

// APSK8(1,7)
const APSK8_NUM_LEVELS: usize = 2;
const APSK8_P: [usize; 2] = [1, 7];
const APSK8_R: [f32; 2] = [0.0, 1.06904495];
const APSK8_PHI: [f32; 2] = [0.0, 0.0];
const APSK8_R_SLICER: [f32; 1] = [0.53452247];
const APSK8_MAP: [u8; 8] = [0, 2, 4, 3, 1, 7, 5, 6];
const APSK8: ApskDef = ApskDef {
    modulation: ModulationScheme::Apsk8,
    num_levels: APSK8_NUM_LEVELS,
    p: &APSK8_P,
    r: &APSK8_R,
    phi: &APSK8_PHI,
    r_slicer: &APSK8_R_SLICER,
    map: &APSK8_MAP,
};

// APSK16(4,12)
const APSK16_NUM_LEVELS: usize = 2;
const APSK16_P: [usize; 2] = [4, 12];
const APSK16_R: [f32; 2] = [0.43246540, 1.12738252];
const APSK16_PHI: [f32; 2] = [0.0, 0.0];
const APSK16_R_SLICER: [f32; 1] = [0.77992396];
const APSK16_MAP: [u8; 16] = [
    11, 10, 8, 9, 12, 2, 7, 1,
    14, 15, 5, 4, 13, 3, 6, 0
];
const APSK16: ApskDef = ApskDef {
    modulation: ModulationScheme::Apsk16,
    num_levels: APSK16_NUM_LEVELS,
    p: &APSK16_P,
    r: &APSK16_R,
    phi: &APSK16_PHI,
    r_slicer: &APSK16_R_SLICER,
    map: &APSK16_MAP,
};

// APSK32(4,12,16)
const APSK32_NUM_LEVELS: usize = 3;
const APSK32_P: [usize; 3] = [4, 12, 16];
const APSK32_R: [f32; 3] = [
    0.27952856,
    0.72980529,
    1.25737989
];
const APSK32_PHI: [f32; 3] = [0.0, 0.0, 0.0];
const APSK32_R_SLICER: [f32; 2] = [0.504666925, 0.993592590];
const APSK32_MAP: [u8; 32] = [
    26,  25,  22,  23,  27,  11,  21,   9,
    13,   3,   7,   1,  12,  10,   8,  24,
    30,  31,  18,  17,  29,  15,  19,   5,
    28,   0,  20,   2,  14,  16,   6,   4
];
const APSK32: ApskDef = ApskDef {
    modulation: ModulationScheme::Apsk32,
    num_levels: APSK32_NUM_LEVELS,
    p: &APSK32_P,
    r: &APSK32_R,
    phi: &APSK32_PHI,
    r_slicer: &APSK32_R_SLICER,
    map: &APSK32_MAP,
};

// APSK64(4,14,20,26)
const APSK64_NUM_LEVELS: usize = 4;
const APSK64_P: [usize; 4] = [4, 14, 20, 26];
const APSK64_R: [f32; 4] = [
    0.18916586,
    0.52466476,
    0.88613129,
    1.30529201
];
const APSK64_PHI: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
const APSK64_R_SLICER: [f32; 3] = [
    0.35691531,
    0.70539802,
    1.09571165
];
const APSK64_MAP: [u8; 64] = [
    54,  53,  51,  52,    48,  49,  28,  50,
    55,  30,  11,  29,    47,  25,  27,  26,
    57,  32,   2,  14,    45,  23,   1,   8,
    56,  31,  12,  13,    46,  24,  10,   9,
    61,  62,  38,  63,    41,  40,  18,  39,
    60,  35,  37,  36,    42,  20,   4,  19,
    58,  33,   3,  15,    44,  22,   0,   7,
    59,  34,  17,  16,    43,  21,   5,   6
];
const APSK64: ApskDef = ApskDef {
    modulation: ModulationScheme::Apsk64,
    num_levels: APSK64_NUM_LEVELS,
    p: &APSK64_P,
    r: &APSK64_R,
    phi: &APSK64_PHI,
    r_slicer: &APSK64_R_SLICER,
    map: &APSK64_MAP,
};

// APSK128(8,18,24,36,42)
const APSK128_NUM_LEVELS: usize = 5;
const APSK128_P: [usize; 5] = [8, 18, 24, 36, 42];
const APSK128_R: [f32; 5] = [
    0.20241030,
    0.46255755,
    0.70972824,
    0.99172282,
    1.34806108
];
const APSK128_PHI: [f32; 5] = [0.0, 0.0, 0.0, 0.0, 0.0];
const APSK128_R_SLICER: [f32; 4] = [
    0.33248392,
    0.58614290,
    0.85072553,
    1.16989195
];
const APSK128_MAP: [u8; 128] = [
    112,  111,  108,  109,    102,  103,  106,  105,
    113,  110,  107,   71,    101,  104,   67,   66,
    115,   73,   39,   41,     99,   63,   38,   36,
    114,   72,   69,   70,    100,   64,   68,   65,
    117,   77,    1,   21,     97,   59,    2,   13,
     76,   43,    4,   20,     60,   33,    3,   14,
    116,   74,   18,   40,     98,   62,   37,   35,
     75,   42,   17,   19,     61,   34,   16,   15,
    123,  124,  127,  126,     91,   90,   87,   88,
    122,  125,   85,   84,     92,   89,   51,   53,
    120,   81,   49,   48,     94,   55,   26,   29,
    121,   82,   50,   83,     93,   54,   27,   52,
    118,   44,    7,    5,     96,   32,    0,   10,
     78,   45,    6,   22,     58,   30,   86,   12,
     80,   79,   25,   47,     57,   56,    9,   28,
    119,   46,   24,   23,     95,   31,    8,   11
];
const APSK128: ApskDef = ApskDef {
    modulation: ModulationScheme::Apsk128,
    num_levels: APSK128_NUM_LEVELS,
    p: &APSK128_P,
    r: &APSK128_R,
    phi: &APSK128_PHI,
    r_slicer: &APSK128_R_SLICER,
    map: &APSK128_MAP,
};

// APSK256(6,18,32,36,46,54,64)
const APSK256_NUM_LEVELS: usize = 7;
const APSK256_P: [usize; 7] = [6, 18, 32, 36, 46, 54, 64];
const APSK256_R: [f32; 7] = [
    0.19219166,
    0.41951191,
    0.60772800,
    0.77572918,
    0.94819963,
    1.12150347,
    1.31012368
];
const APSK256_PHI: [f32; 7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
const APSK256_R_SLICER: [f32; 6] = [
    0.30585179,
    0.51361996,
    0.69172859,
    0.86196440,
    1.03485155,
    1.21581364
];
const APSK256_MAP: [u8; 256] = [
    232,  231,  229,  230,    224,  225,  227,  226,
    216,  217,  219,  218,    164,  223,  221,  222,
    233,  228,  170,  171,    166,  167,  169,  168,
    215,  220,  160,  159,    165,  163,  161,  162,
    235,  173,  123,  121,     75,   76,   78,   77,
    213,  157,   70,  109,     73,  115,   71,   72,
    234,  172,  120,  174,    116,  117,  119,  118,
    214,  158,  110,  156,    114,  113,  111,  112,
    239,  178,   82,  126,    255,   19,   47,    5,
    209,  152,   66,  105,     56,   12,   33,   13,
    238,  177,   81,  125,    138,  193,   46,    4,
    210,  153,   67,  106,      7,    1,   34,    2,
    236,  175,   79,  122,     74,   42,   44,   43,
    212,  155,   69,  108,     39,   38,   36,   37,
    237,  176,   80,  124,      3,   40,   45,   41,
    211,  154,   68,  107,     16,   15,   35,   14,
    248,  249,  251,  250,    191,  190,  252,  253,
    200,  199,  197,  198,    139,  140,  196,  195,
    247,  185,  187,  186,    137,  136,  188,  189,
    201,  145,  143,  144,     93,   94,  142,  141,
    245,  183,   87,  131,     54,   53,   88,   89,
    203,  147,   61,   99,     26,   27,   60,   59,
    246,  184,  133,  132,     91,   90,  134,  135,
    202,  146,   97,   98,     57,   58,   96,   95,
    240,  179,   83,  127,     23,  192,   48,   17,
    208,  151,   65,  104,      6,   11,   32,  206,
    241,  180,   84,  128,    254,  242,   49,   18,
    207,  150,   64,  102,     92,  194,   31,   10,
    244,  182,   86,  130,     55,   21,   52,   51,
    204,  148,   62,  100,     25,    8,   28,   29,
    243,  181,   85,  129,      0,   22,   50,   20,
    205,  149,   63,  101,     24,    9,   30,  103
];
const APSK256: ApskDef = ApskDef {
    modulation: ModulationScheme::Apsk256,
    num_levels: APSK256_NUM_LEVELS,
    p: &APSK256_P,
    r: &APSK256_R,
    phi: &APSK256_PHI,
    r_slicer: &APSK256_R_SLICER,
    map: &APSK256_MAP,
};
