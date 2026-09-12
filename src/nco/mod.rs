// NCO module
// Current state:
// - Nco ready (+autotests)
// - oscillator backends are internal implementation details
// - TableOscillator ready to use

#[path = "direct_backend.rs"]
mod direct;
#[path = "interpolated_lookup_table_backend.rs"]
mod interpolated;
#[path = "lookup_table_backend.rs"]
mod nco;
#[path = "nco.rs"]
pub mod osc;
pub mod phase;
#[path = "table_oscillator.rs"]
pub mod synth;

pub use osc::{Nco, NcoBackend};
pub use phase::{unwrap_phase, unwrap_phase2};
pub use synth::TableOscillator;
