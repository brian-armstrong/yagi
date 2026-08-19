//
// noise.rs
//
// Gaussian source shared by the channel emulators.
//

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::random::randnf;

/// source of Gaussian samples for the noise-driven impairments
#[derive(Debug, Clone)]
pub(crate) enum NoiseSource {
    Global,
    Seeded(Box<StdRng>),
}

impl NoiseSource {
    pub(crate) fn seeded(seed: u64) -> Self {
        NoiseSource::Seeded(Box::new(StdRng::seed_from_u64(seed)))
    }

    pub(crate) fn randnf(&mut self) -> f32 {
        match self {
            NoiseSource::Global => randnf(),
            NoiseSource::Seeded(rng) => {
                // reimpl of randnf on seeded rng
                let u1: f32 = loop {
                    let u = rng.gen::<f32>();
                    if u != 0.0 {
                        break u;
                    }
                };
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).sin()
            }
        }
    }
}
