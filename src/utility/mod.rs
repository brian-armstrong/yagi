//! Bit packing, bit counting, and byte-array shift operations.
//!
//! These low-level helpers operate directly on caller-provided byte slices.
//! Bit shifts are limited to fewer than eight positions. Byte shifts accept
//! any amount and fill vacated elements with zero.
//! Bit packing uses most-significant-bit-first order within each byte.

mod bits;
mod bshift_array;
mod shift_array;

pub use bits::{
    bdotprod, count_bit_errors, count_bit_errors_array, count_leading_zeros, count_ones, count_ones_mod2, msb_index,
    pack_array, pack_bytes, print_bitstring, repack_bytes, reverse_16, reverse_24, reverse_32, reverse_8, reverse_byte,
    unpack_array, unpack_bytes,
};
pub use bshift_array::{lbcircshift, lbshift, rbcircshift, rbshift};
pub use shift_array::{lcircshift, lshift, rcircshift, rshift};

#[cfg(test)]
pub(crate) mod test_helpers;
