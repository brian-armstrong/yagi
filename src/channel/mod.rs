//! Complex baseband channel emulators.
//!
//! [`Channel`] combines selected fixed multipath, shadowing, carrier-offset,
//! and additive-noise impairments. [`TimeVaryingMultipathChannel`] models
//! fading echo paths that change as samples are processed. Both process
//! `Complex32` samples, one sample at a time or in blocks.

mod channel;
mod noise;
mod time_varying_multipath_channel;

pub use channel::Channel;
pub use time_varying_multipath_channel::TimeVaryingMultipathChannel;
