//! Multichannel filter banks and OFDM frame processing.
//!
//! The channelizers convert blocks of time samples to or from frequency
//! channels. The OFDM generator and synchronizer share an
//! [`OfdmFrameConfig`] that defines the subcarrier allocation, cyclic-prefix
//! length and taper length.
//!
//! Channelizer channel arrays, OFDM allocation slices and recovered OFDM
//! symbols all use natural FFT-bin order. Bin zero is DC, followed by the
//! positive frequencies, then the negative frequencies. For `M` channels,
//! index `k` is centered at `k/M` cycles per time-domain sample, modulo one
//! cycle per sample, so index `M - 1` is at `-1/M`.
//!
//! The channelizers do not share an analysis gain. A tone that reads near
//! `M` on one analyzer can read near one on another, so each constructor
//! states the gain of its built-in prototype.

mod ofdm_frame;
mod ofdm_frame_generator;
mod ofdm_frame_synchronizer;
mod oversampled_polyphase_channelizer;
mod polyphase_channelizer;
mod rational_polyphase_channelizer;

pub use ofdm_frame::{
    ofdmframe_init_default_sctype, ofdmframe_init_sctype_range, ofdmframe_sctype_from_string, ofdmframe_sctype_string,
    OfdmFrameConfig, SubcarrierCounts, SubcarrierType,
};
pub use ofdm_frame_generator::OfdmFrameGenerator;
pub use ofdm_frame_synchronizer::{
    EqGainMethod, OfdmFrameSynchronizer, OfdmFrameSynchronizerBlockOutput, OfdmFrameSynchronizerOutput,
    OfdmFrameSynchronizerSymbol,
};
pub use oversampled_polyphase_channelizer::OversampledPolyphaseChannelizer;
pub use polyphase_channelizer::{ChannelizerType, PolyphaseChannelizer};
pub use rational_polyphase_channelizer::RationalPolyphaseChannelizer;
