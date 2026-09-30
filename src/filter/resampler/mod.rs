//! Filters that change the sample rate of a stream.
//!
//! Every type here keeps history between calls, so a stream can be processed
//! in pieces. Some methods take input only in whole steps. Half-band
//! decimation and analysis work on pairs of samples, a decimating
//! [`MultistageHalfBandResampler`] on a multiple of its total factor, and
//! [`RationalResampler`] on whole blocks. Each method states what it needs.
//! Rates are the output rate divided by the input rate.
//!
//! [`RationalResampler`] and [`HalfBandResampler`] produce a fixed number of
//! outputs for each block of input. [`MultistageHalfBandResampler`] chains
//! half-band stages for power-of-two changes. [`ArbitraryResampler`] and
//! [`MultistageResampler`] accept any ratio. They round their phase step, so
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
pub use multi_stage_half_band_resampler::{MultistageHalfBandResampler, ResampType};
pub use multi_stage_resampler::MultistageResampler;
pub use rational_resampler::RationalResampler;
