//! Stateful audio-sample processors and a CVSD codec.
//!
//! The processors operate on scalar `f32` samples. They do not store a sample
//! rate, so time-related constructor parameters are measured in samples.

mod cvsd_codec;
mod limiter;
mod peak_hold;

pub use cvsd_codec::CvsdCodec;
pub use limiter::Limiter;
pub use peak_hold::PeakHold;
