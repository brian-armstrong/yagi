use std::f32::consts::PI;

// Constants
const VCO_STATIC_LUT_WORDBITS: u32 = 32;
const VCO_STATIC_LUT_NBITS: u32 = 10;
const VCO_STATIC_LUT_SIZE: usize = 1 << VCO_STATIC_LUT_NBITS;
const VCO_STATIC_LUT_HSIZE: usize = VCO_STATIC_LUT_SIZE >> 1;
const VCO_STATIC_LUT_QSIZE: usize = VCO_STATIC_LUT_SIZE >> 2;

fn vco_static_lut_index_shifted_pi2(index: usize) -> usize {
    (index + VCO_STATIC_LUT_QSIZE) & (VCO_STATIC_LUT_SIZE - 1)
}

fn vco_static_lut_theta_shifted_pi2(theta: u32) -> u32 {
    theta.wrapping_add(1 << (VCO_STATIC_LUT_WORDBITS - 2))
}

fn vco_static_lut_theta_accum(theta: u32) -> u32 {
    theta & ((1 << (VCO_STATIC_LUT_WORDBITS - VCO_STATIC_LUT_NBITS)) - 1)
}

#[derive(Debug, Clone)]
pub struct VcoLut {
    value: f32,
    skew: f32,
}

#[derive(Debug, Clone)]
pub struct Vco {
    vco_sintab: Vec<VcoLut>,
}

impl Vco {
    pub fn new() -> Self {
        let mut vco = Vco { vco_sintab: vec![VcoLut { value: 0.0, skew: 0.0 }; VCO_STATIC_LUT_SIZE] };

        let mut theta = 0;
        let d_theta = u32::MAX / VCO_STATIC_LUT_SIZE as u32;

        // Initialize sine table
        for i in 0..VCO_STATIC_LUT_QSIZE {
            let value = Vco::fp_sin(theta);
            let next_value = Vco::fp_sin(theta + d_theta);
            let skew = (next_value - value) / d_theta as f32;
            let index = i;
            let index_pi = index + VCO_STATIC_LUT_HSIZE;
            vco.vco_sintab[index] = VcoLut { value, skew };
            vco.vco_sintab[index_pi] = VcoLut { value: -value, skew: -skew };
            theta = theta.wrapping_add(d_theta);
        }

        let index_pi_2 = VCO_STATIC_LUT_QSIZE;
        let index_3_pi_2 = VCO_STATIC_LUT_QSIZE + VCO_STATIC_LUT_HSIZE;
        vco.vco_sintab[index_pi_2].value = 1.0;
        vco.vco_sintab[index_pi_2].skew = -vco.vco_sintab[index_pi_2 - 1].skew;
        vco.vco_sintab[index_3_pi_2].value = -vco.vco_sintab[index_pi_2].value;
        vco.vco_sintab[index_3_pi_2].skew = vco.vco_sintab[index_pi_2 - 1].skew;

        // Mirror [0, PI/2] range to [PI/2, PI] range

        for i in 1..VCO_STATIC_LUT_QSIZE {
            let index_pi_2 = i + VCO_STATIC_LUT_QSIZE;
            let value = vco.vco_sintab[VCO_STATIC_LUT_QSIZE - i].value;
            let skew = vco.vco_sintab[VCO_STATIC_LUT_QSIZE - i - 1].skew;
            vco.vco_sintab[index_pi_2].value = value;
            vco.vco_sintab[index_pi_2].skew = -skew;
            vco.vco_sintab[index_pi_2 + VCO_STATIC_LUT_HSIZE].value = -value;
            vco.vco_sintab[index_pi_2 + VCO_STATIC_LUT_HSIZE].skew = skew;
        }

        vco
    }

    fn fp_sin(theta: u32) -> f32 {
        (theta as f32 * PI / (i32::MAX as u32 + 1) as f32).sin()
    }

    pub fn sin(&self, theta: u32) -> f32 {
        let index = self.static_index(theta);
        let v = self.vco_sintab[index].value;
        let s = self.vco_sintab[index].skew;
        v + vco_static_lut_theta_accum(theta) as f32 * s
    }

    pub fn cos(&self, theta: u32) -> f32 {
        let index = self.static_index(theta);
        let index_pi2 = vco_static_lut_index_shifted_pi2(index);
        let theta_pi2 = vco_static_lut_theta_shifted_pi2(theta);
        let v = self.vco_sintab[index_pi2].value;
        let s = self.vco_sintab[index_pi2].skew;
        v + vco_static_lut_theta_accum(theta_pi2) as f32 * s
    }

    pub fn sin_cos(&self, theta: u32) -> (f32, f32) {
        let index = self.static_index(theta);
        let index_pi2 = vco_static_lut_index_shifted_pi2(index);
        let theta_pi2 = vco_static_lut_theta_shifted_pi2(theta);
        let s_v = self.vco_sintab[index].value;
        let s_s = self.vco_sintab[index].skew;
        let c_v = self.vco_sintab[index_pi2].value;
        let c_s = self.vco_sintab[index_pi2].skew;
        (
            s_v + vco_static_lut_theta_accum(theta) as f32 * s_s,
            c_v + vco_static_lut_theta_accum(theta_pi2) as f32 * c_s,
        )
    }

    fn static_index(&self, theta: u32) -> usize {
        (theta as usize >> (VCO_STATIC_LUT_WORDBITS - VCO_STATIC_LUT_NBITS)) & (VCO_STATIC_LUT_SIZE - 1)
    }
}

impl Default for Vco {
    fn default() -> Self {
        Self::new()
    }
}
