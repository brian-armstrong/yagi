// Quantization module
// Current state:
// - compand ready to use (+autotests), mu-law only (liquid's A-law is an
//   empty stub upstream)
// - quantize ready to use (+autotests), liquid's quantizer.inline.c
// - liquid's structured quantizer (quantizer.proto.c) is deliberately not
//   ported: its execute_adc/execute_dac are stubs that write 0 and ignore the
//   requested range, so there is no upstream behavior to reproduce
//
// the two halves compose: compand to warp the amplitude, then quantize.

pub mod compand;
pub mod quantize;

pub use compand::*;
pub use quantize::*;
