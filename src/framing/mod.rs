//! Packet coding, frame synchronization, and synthetic test signals.
//!
//! On the transmit side, [`PacketModem`] turns a byte payload into coded
//! constellation symbols and [`PilotGenerator`] interleaves known pilots with
//! them. On the receive side, [`FrameDetector`] finds a known preamble and
//! estimates timing, gain, and carrier offsets. [`QuadratureSynchronizer`] applies
//! those estimates to recover symbols, [`PilotSynchronizer`] corrects residual
//! carrier and gain errors from the pilots, and [`PacketModem`] decodes the
//! result. [`SymbolTracker`] recovers a continuous symbol stream without a
//! preamble. [`SymbolStream`], [`ArbitraryRateSymbolStream`], and
//! [`MultiSignalSource`] generate test signals.
//!
//! Center frequencies and bandwidths are normalized to the sample rate and
//! given in cycles per sample. Carrier offset estimates and corrections are in
//! radians, per sample or per symbol as each item states.
//!
//! A successful decode means only that decoding ran. The returned boolean
//! reports the integrity check, and the payload is written either way.
//!
//! The module has no complete packet transmitter or receiver. Combine these
//! objects for custom framing, or use the OFDM objects in
//! [`crate::multichannel`].

mod arbitrary_rate_symbol_stream;
mod frame_detector;
mod multi_signal_source;
mod packet_modem;
mod packet_symbolizer;
mod pilot_generator;
mod pilot_synchronizer;
mod quadrature_synchronizer;
mod signal_source;
mod symbol_stream;
mod symbol_tracker;

pub use arbitrary_rate_symbol_stream::ArbitraryRateSymbolStream;
pub use frame_detector::FrameDetector;
pub use multi_signal_source::MultiSignalSource;
pub use packet_modem::PacketModem;
pub use packet_symbolizer::PacketSymbolizer;
pub use pilot_generator::{pilot_count, pilot_frame_len, PilotGenerator};
pub use pilot_synchronizer::PilotSynchronizer;
pub use quadrature_synchronizer::{
    QuadratureSynchronizer, QuadratureSynchronizerBlockOutput, QuadratureSynchronizerOutput,
};
pub use signal_source::{SignalSource, SignalSourceCallback, SignalSourceConfig, SignalSourceType, SourceId};
pub use symbol_stream::SymbolStream;
pub use symbol_tracker::SymbolTracker;
