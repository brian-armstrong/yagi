// NCO module
// Current state:
// - Nco ready (+autotests)
// - oscillator backends are internal implementation details
// - TableOscillator ready to use

mod direct;
mod interpolated;
mod nco;
pub mod osc;
pub mod synth;
pub mod utilities;

pub use osc::{Nco, NcoBackend};
pub use synth::TableOscillator;
pub use utilities::{unwrap_phase, unwrap_phase2};
