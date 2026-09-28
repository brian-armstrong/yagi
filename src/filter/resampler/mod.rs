//! Filters that change the sample rate of a stream.
//!
//! Every type here keeps history between calls, so a stream can be processed
//! in pieces of any size. Rates are the output rate divided by the input rate.
//!
//! [`RationalResampler`] and [`HalfBandResampler`] produce a fixed number of
//! outputs for each block of input. [`MultiStageHalfBandResampler`] chains
//! half-band stages for power-of-two changes. [`ArbitraryResampler`] and
//! [`MultiStageResampler`] accept any ratio. They round their phase step, so
//! the number of outputs for a given input length depends on their current
//! phase. Call their `num_output` method to size an output buffer before
//! each call.

mod arbitrary_resampler;
mod half_band_resampler;
mod multi_stage_half_band_resampler;
mod multi_stage_resampler;
mod rational_resampler;

pub use arbitrary_resampler::ArbitraryResampler;
pub use half_band_resampler::HalfBandResampler;
pub use multi_stage_half_band_resampler::{MultiStageHalfBandResampler, ResampType};
pub use multi_stage_resampler::MultiStageResampler;
pub use rational_resampler::RationalResampler;
