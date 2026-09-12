// Sequence module
// Current state:
// - Autotests matching and passing
// - msequence and bsequence ready to use

#[path = "binary_sequence.rs"]
pub mod bsequence;
#[path = "maximal_length_sequence.rs"]
pub mod msequence;

pub use bsequence::*;
pub use msequence::*;
