//! Numerically controlled oscillators, mixers, and phase-unwrapping helpers.
//!
//! [`Nco`] synthesizes a complex sinusoid from a fixed-point
//! phase accumulator. [`TableOscillator`] advances
//! through a caller-provided complex sequence. Frequencies and phases are
//! measured in radians per sample and radians, respectively.

mod direct_backend;
mod interpolated_lookup_table_backend;
mod lookup_table_backend;
mod nco;
mod phase;
mod table_oscillator;

pub use nco::{Nco, NcoBackend};
pub use phase::{unwrap_phase, unwrap_phase2};
pub use table_oscillator::TableOscillator;
