use std::f32::consts::PI;

// Constants
const INTERPOLATED_LUT_WORDBITS: u32 = 32;
const INTERPOLATED_LUT_NBITS: u32 = 10;
const INTERPOLATED_LUT_SIZE: usize = 1 << INTERPOLATED_LUT_NBITS;
const INTERPOLATED_LUT_HSIZE: usize = INTERPOLATED_LUT_SIZE >> 1;
const INTERPOLATED_LUT_QSIZE: usize = INTERPOLATED_LUT_SIZE >> 2;

fn interpolated_lut_index_shifted_pi2(index: usize) -> usize {
    (index + INTERPOLATED_LUT_QSIZE) & (INTERPOLATED_LUT_SIZE - 1)
}

fn interpolated_lut_theta_shifted_pi2(theta: u32) -> u32 {
    theta.wrapping_add(1 << (INTERPOLATED_LUT_WORDBITS - 2))
}

fn interpolated_lut_theta_accum(theta: u32) -> u32 {
    theta & ((1 << (INTERPOLATED_LUT_WORDBITS - INTERPOLATED_LUT_NBITS)) - 1)
}

#[derive(Debug, Clone)]
struct InterpolatedEntry {
    value: f32,
    skew: f32,
}

#[derive(Debug, Clone)]
pub(super) struct InterpolatedLookupTableBackend {
    sine_table: Vec<InterpolatedEntry>,
}

impl InterpolatedLookupTableBackend {
    pub fn new() -> Self {
        let mut backend = InterpolatedLookupTableBackend {
            sine_table: vec![InterpolatedEntry { value: 0.0, skew: 0.0 }; INTERPOLATED_LUT_SIZE],
        };

        let mut theta = 0;
        let d_theta = u32::MAX / INTERPOLATED_LUT_SIZE as u32;

        // Initialize sine table
        for i in 0..INTERPOLATED_LUT_QSIZE {
            let value = InterpolatedLookupTableBackend::fp_sin(theta);
            let next_value = InterpolatedLookupTableBackend::fp_sin(theta + d_theta);
            let skew = (next_value - value) / d_theta as f32;
            let index = i;
            let index_pi = index + INTERPOLATED_LUT_HSIZE;
            backend.sine_table[index] = InterpolatedEntry { value, skew };
            backend.sine_table[index_pi] = InterpolatedEntry { value: -value, skew: -skew };
            theta = theta.wrapping_add(d_theta);
        }

        let index_pi_2 = INTERPOLATED_LUT_QSIZE;
        let index_3_pi_2 = INTERPOLATED_LUT_QSIZE + INTERPOLATED_LUT_HSIZE;
        backend.sine_table[index_pi_2].value = 1.0;
        backend.sine_table[index_pi_2].skew = -backend.sine_table[index_pi_2 - 1].skew;
        backend.sine_table[index_3_pi_2].value = -backend.sine_table[index_pi_2].value;
        backend.sine_table[index_3_pi_2].skew = backend.sine_table[index_pi_2 - 1].skew;

        // Mirror [0, PI/2] range to [PI/2, PI] range

        for i in 1..INTERPOLATED_LUT_QSIZE {
            let index_pi_2 = i + INTERPOLATED_LUT_QSIZE;
            let value = backend.sine_table[INTERPOLATED_LUT_QSIZE - i].value;
            let skew = backend.sine_table[INTERPOLATED_LUT_QSIZE - i - 1].skew;
            backend.sine_table[index_pi_2].value = value;
            backend.sine_table[index_pi_2].skew = -skew;
            backend.sine_table[index_pi_2 + INTERPOLATED_LUT_HSIZE].value = -value;
            backend.sine_table[index_pi_2 + INTERPOLATED_LUT_HSIZE].skew = skew;
        }

        backend
    }

    fn fp_sin(theta: u32) -> f32 {
        (theta as f32 * PI / (i32::MAX as u32 + 1) as f32).sin()
    }

    pub fn sin(&self, theta: u32) -> f32 {
        let index = self.static_index(theta);
        let v = self.sine_table[index].value;
        let s = self.sine_table[index].skew;
        v + interpolated_lut_theta_accum(theta) as f32 * s
    }

    pub fn cos(&self, theta: u32) -> f32 {
        let index = self.static_index(theta);
        let index_pi2 = interpolated_lut_index_shifted_pi2(index);
        let theta_pi2 = interpolated_lut_theta_shifted_pi2(theta);
        let v = self.sine_table[index_pi2].value;
        let s = self.sine_table[index_pi2].skew;
        v + interpolated_lut_theta_accum(theta_pi2) as f32 * s
    }

    pub fn sin_cos(&self, theta: u32) -> (f32, f32) {
        let index = self.static_index(theta);
        let index_pi2 = interpolated_lut_index_shifted_pi2(index);
        let theta_pi2 = interpolated_lut_theta_shifted_pi2(theta);
        let s_v = self.sine_table[index].value;
        let s_s = self.sine_table[index].skew;
        let c_v = self.sine_table[index_pi2].value;
        let c_s = self.sine_table[index_pi2].skew;
        (
            s_v + interpolated_lut_theta_accum(theta) as f32 * s_s,
            c_v + interpolated_lut_theta_accum(theta_pi2) as f32 * c_s,
        )
    }

    fn static_index(&self, theta: u32) -> usize {
        (theta as usize >> (INTERPOLATED_LUT_WORDBITS - INTERPOLATED_LUT_NBITS)) & (INTERPOLATED_LUT_SIZE - 1)
    }
}

impl Default for InterpolatedLookupTableBackend {
    fn default() -> Self {
        Self::new()
    }
}
