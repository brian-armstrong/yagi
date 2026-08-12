// Utility module
// Current state:
// - bits ready to use (+autotests)
// - other parts of utility TBD

pub mod bits;
pub mod bshift_array;
pub mod shift_array;

pub use bits::*;
pub use bshift_array::*;
pub use shift_array::*;

#[cfg(test)]
pub(crate) mod test_helpers;