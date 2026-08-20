// Dotprod module
// Current state:
// - Dotprod ready to use (+autotests)
// - sumsq missing

mod ccc;
mod crc;
mod rrr;

#[cfg(feature = "simd")]
mod reduce;

pub trait DotProd<Rhs> {
    type Output;

    /// Computes the dot product of two equal-length slices.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths.
    fn dotprod(&self, other: &[Rhs]) -> Self::Output;
}
