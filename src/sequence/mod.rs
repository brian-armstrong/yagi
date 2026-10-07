//! Binary sequences, linear-feedback shift-register generators and complex
//! chip tables.
//!
//! [`BinarySequence`] stores a fixed number of bits for correlation and
//! binary code operations. [`MaximalLengthSequence`] generates LFSR sequences
//! from a built-in or caller-provided polynomial. [`TableSequence`] steps
//! through a table of complex chips, such as a spreading code, and spreads and
//! despreads symbols with it.

mod binary_sequence;
mod maximal_length_sequence;
mod table_sequence;

pub use binary_sequence::BinarySequence;
pub use maximal_length_sequence::MaximalLengthSequence;
pub use table_sequence::TableSequence;
