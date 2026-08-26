// NCO module
// Current state:
// - Nco ready (+autotests)
// - Vco/vcoi/vcod TBD
// - Synth ready to use

pub mod direct;
pub mod nco;
pub mod osc;
pub mod synth;
pub mod utilities;
pub mod vco;

pub use osc::{Osc, OscScheme};
pub use synth::Synth;
pub use utilities::{unwrap_phase, unwrap_phase2};
