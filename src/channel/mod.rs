// Channel module
// Current state:
// - channel: generic channel emulator (multipath, shadowing, carrier offset,
//   AWGN)
// - tvmpch: time-varying multi-path channel emulator (Rayleigh-fading taps)

mod channel;
mod noise;
mod tvmpch;

pub use channel::*;
pub use tvmpch::*;
