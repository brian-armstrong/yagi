//! Complex baseband channel emulators.
//!
//! [`Channel`] combines selected fixed multipath,
//! shadowing, carrier-offset, and additive-noise impairments.
//! [`TimeVaryingMultipathChannel`]
//! models fading echo paths that change as samples are processed.

mod channel;
mod noise;
mod time_varying_multipath_channel;

pub use channel::Channel;
pub use time_varying_multipath_channel::TimeVaryingMultipathChannel;
