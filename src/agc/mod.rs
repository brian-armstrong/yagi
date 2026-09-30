//! Automatic gain control for real and complex sample streams.
//!
//! [`Agc`] adjusts gain to bring the output magnitude toward one, then applies
//! a configurable output scale. Its optional RSSI-based squelch reports signal
//! state but does not mute output samples.

mod agc;

pub use agc::{Agc, AgcSquelchMode};
