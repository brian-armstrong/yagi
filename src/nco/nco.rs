use std::f32::consts::PI;

// Constants
const NCO_STATIC_LUT_WORDBITS: u32 = 32;
const NCO_STATIC_LUT_NBITS: u32 = 10;
const NCO_STATIC_LUT_SIZE: usize = 1 << NCO_STATIC_LUT_NBITS;
const NCO_STATIC_LUT_QSIZE: usize = NCO_STATIC_LUT_SIZE >> 2;

fn nco_static_lut_index_shifted_pi2(index: usize) -> usize {
    (index + NCO_STATIC_LUT_QSIZE) & (NCO_STATIC_LUT_SIZE - 1)
}

#[derive(Debug, Clone)]
pub(super) struct Nco {
    sintab: Vec<f32>,
}

impl Nco {
    pub fn new() -> Self {
        let mut nco = Nco { sintab: vec![0.0; NCO_STATIC_LUT_SIZE] };

        // Initialize sine table
        for i in 0..NCO_STATIC_LUT_SIZE {
            nco.sintab[i] = (2.0 * PI * i as f32 / NCO_STATIC_LUT_SIZE as f32).sin();
        }

        nco
    }

    pub fn sin(&self, theta: u32) -> f32 {
        let index = self.static_index(theta);
        self.sintab[index]
    }

    pub fn cos(&self, theta: u32) -> f32 {
        let index = self.static_index(theta);
        let index_pi2 = nco_static_lut_index_shifted_pi2(index);
        self.sintab[index_pi2]
    }

    pub fn sin_cos(&self, theta: u32) -> (f32, f32) {
        let index = self.static_index(theta);
        let index_pi2 = nco_static_lut_index_shifted_pi2(index);
        (self.sintab[index], self.sintab[index_pi2])
    }

    pub fn static_index(&self, theta: u32) -> usize {
        (((theta as usize) + (1 << (NCO_STATIC_LUT_WORDBITS - NCO_STATIC_LUT_NBITS - 1)))
            >> (NCO_STATIC_LUT_WORDBITS - NCO_STATIC_LUT_NBITS)) as usize
            & (NCO_STATIC_LUT_SIZE - 1)
    }
}
