//! Modulation and demodulation.
//!
//! [`Modem`] maps integer symbols to complex constellation points and back,
//! one point per symbol. The other types turn a message or symbols into
//! complex baseband samples and back. [`AmplitudeModem`] and the frequency
//! modulator pair carry a real message. The FSK, CPFSK and GMSK pairs carry
//! each symbol over several samples.
//!
//! Frequencies are in cycles per sample unless an item says otherwise. The
//! symbol demodulators expect input aligned to symbol boundaries, with no
//! frequency offset. Most objects keep state between calls, so use separate
//! objects to modulate and demodulate, and reset them between independent
//! streams.

mod amplitude_modem;
mod cpfsk_demodulator;
mod cpfsk_modulator;
mod frequency_demodulator;
mod frequency_modulator;
mod fsk_demodulator;
mod fsk_modulator;
mod gmsk_demodulator;
mod gmsk_modulator;
mod modem;

pub use amplitude_modem::{AmplitudeModem, AmpmodemType};
pub use cpfsk_demodulator::CpfskDemodulator;
pub use cpfsk_modulator::{CpfskFilterType, CpfskModulator};
pub use frequency_demodulator::FrequencyDemodulator;
pub use frequency_modulator::FrequencyModulator;
pub use fsk_demodulator::FskDemodulator;
pub use fsk_modulator::FskModulator;
pub use gmsk_demodulator::GmskDemodulator;
pub use gmsk_modulator::GmskModulator;
pub use modem::{
    gray_decode, gray_encode, pack_soft_bits, unpack_soft_bits, Modem, ModulationScheme, NUM_MODULATION_SCHEMES,
};
