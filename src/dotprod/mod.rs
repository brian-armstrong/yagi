// Dotprod module
// Current state:
// - Dotprod ready to use (+autotests)
// - sumsq ready to use (+autotests)

mod ccc;
mod crc;
mod dotproduct;
mod rrr;
mod sumsq;

pub use sumsq::{sumsqcf, sumsqf};

#[cfg(feature = "simd")]
mod reduce;

pub use dotproduct::DotProduct;

/// A dot product kernel resolved ahead of time by [`DotProd::plan`].
pub type DotProdKernel<Elem, Rhs, Out> = unsafe fn(&[Elem], &[Rhs]) -> Out;

pub trait DotProd<Rhs> {
    type Output;

    /// Computes the dot product of two equal-length slices.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths.
    fn dotprod(&self, other: &[Rhs]) -> Self::Output;

    /// Resolve the kernel for a dot product of `len` elements so that a
    /// caller holding fixed-length coefficients can calculate dispatch only
    /// once instead of on every execution.
    /// 
    /// The returned pointer is valid for the life of the process but is tied
    /// to this machine's detected features.
    fn plan(len: usize) -> unsafe fn(&Self, &[Rhs]) -> Self::Output {
        // default impl
        let _ = len;
        |x, h| x.dotprod(h)
    }
}
