//! Automatic gain control for real and complex sample streams.
//!
//! [`Agc`] tracks the input magnitude toward unity, then
//! applies an optional output scale. It also provides an optional RSSI-based
//! squelch state machine.

mod agc;

pub use agc::{Agc, AgcSquelchMode};
