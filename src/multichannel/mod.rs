// Multichannel module
// Current state:
// - firpfbch: FIR polyphase filterbank channelizer (analyzer/synthesizer)
// - firpfbch2: FIR polyphase filterbank channelizer with 2x output rate
// - firpfbchr: FIR polyphase filterbank channelizer with rational rate
// - ofdmframe: subcarrier allocation and PLCP sequences shared by the
//   generator and synchronizer
// - ofdmframegen: OFDM frame generator
// - ofdmframesync: OFDM frame synchronizer

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
