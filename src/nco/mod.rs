// NCO module
// Current state:
// - Nco ready (+autotests)
// - oscillator backends are internal implementation details
// - TableOscillator ready to use

mod direct_backend;
mod interpolated_lookup_table_backend;
mod lookup_table_backend;
mod nco;
mod phase;
mod table_oscillator;

pub use nco::{Nco, NcoBackend};
pub use phase::{unwrap_phase, unwrap_phase2};
pub use table_oscillator::TableOscillator;
