// Channel module
// Current state:
// - channel: generic channel emulator (multipath, shadowing, carrier offset,
//   AWGN)
// - tvmpch: time-varying multi-path channel emulator (Rayleigh-fading taps)

mod channel;
mod noise;
#[path = "time_varying_multipath_channel.rs"]
mod tvmpch;

pub use channel::*;
pub use tvmpch::*;
