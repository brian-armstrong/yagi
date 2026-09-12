// Channel module
// Current state:
// - Channel: generic channel emulator (multipath, shadowing, carrier offset,
//   AWGN)
// - TimeVaryingMultipathChannel: time-varying multipath channel emulator
//   (Rayleigh-fading taps)

mod channel;
mod noise;
mod time_varying_multipath_channel;

pub use channel::Channel;
pub use time_varying_multipath_channel::TimeVaryingMultipathChannel;
