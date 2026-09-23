//! Mu-law companding and uniform sign-magnitude quantization.
//!
//! Use [`compress_mulaw`] before [`quantize_adc`] to give low-amplitude signals
//! more code resolution. Recover samples with [`quantize_dac`] and
//! [`expand_mulaw`]. The uniform quantizer represents real values near
//! `[-1, 1]` with a sign bit and magnitude bits.

pub mod compand;
pub mod quantize;

pub use compand::{compress_cf_mulaw, compress_mulaw, expand_cf_mulaw, expand_mulaw};
pub use quantize::{quantize_adc, quantize_dac};
