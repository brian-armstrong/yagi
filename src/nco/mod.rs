//! Numerically controlled oscillators, mixers, and phase unwrapping.
//!
//! [`Nco`] synthesizes a complex sinusoid from a 32-bit fixed-point phase
//! accumulator. [`TableOscillator`] steps through a caller-supplied complex
//! sequence, such as a spreading code. Both mix samples up or down, and both
//! can track the phase of an input with a simple phase-locked loop. See
//! [`TableOscillator::pll_step`] for measuring the phase error when the table
//! is not a sinusoid.
//!
//! Phases are in radians and frequencies in radians per sample.

mod direct_backend;
mod interpolated_lookup_table_backend;
mod lookup_table_backend;
mod nco;
mod phase;
mod table_oscillator;

pub use nco::{Nco, NcoBackend};
pub use phase::{unwrap_phase, unwrap_phase2};
pub use table_oscillator::TableOscillator;
