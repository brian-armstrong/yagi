//! Stateful audio-sample processors and a CVSD codec.
//!
//! These types operate on scalar `f32` samples. They do not store a sample
//! rate, so delay and hold lengths are measured in samples.

mod cvsd_codec;
mod limiter;
mod peak_hold;

pub use cvsd_codec::CvsdCodec;
pub use limiter::Limiter;
pub use peak_hold::PeakHold;
