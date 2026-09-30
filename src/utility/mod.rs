//! Bit packing, bit counting, and byte-array shift operations.
//!
//! Packing and shift operations work on caller-provided byte slices, and
//! shifts modify the slice in place. A byte slice is read as one bit string,
//! starting from the most significant bit of the first byte. Bitwise shifts
//! accept amounts in `0..8`, and bytewise shifts accept any amount.
//!
//! [`pack_bytes`] puts a partial final byte in the low bits, while
//! [`unpack_bytes`] always reads from the high bit, so the two are inverses
//! only for whole bytes. Use [`repack_bytes`] for a partial byte aligned to
//! the high bit.

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
