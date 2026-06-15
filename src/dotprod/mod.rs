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

    fn dotprod(&self, other: &[Rhs]) -> Self::Output;
}
