//! Binary sequences and linear-feedback shift-register generators.
//!
//! [`BinarySequence`] stores a fixed number
//! of bits for correlation and binary code operations.
//! [`MaximalLengthSequence`] generates
//! LFSR sequences from a built-in or caller-provided polynomial.

mod binary_sequence;
mod maximal_length_sequence;

pub use binary_sequence::BinarySequence;
pub use maximal_length_sequence::MaximalLengthSequence;
