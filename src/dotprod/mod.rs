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
#[cfg(feature = "simd")]
mod ccc_block;
#[cfg(feature = "simd")]
mod crc_block;
#[cfg(feature = "simd")]
mod rrr_block;

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

    /// Optionally resolve the kernel for a dot product of a block of elements.
    #[doc(hidden)]
    fn plan_block(
        h: &[Rhs],
    ) -> Option<DotProdBlockPlan<Self, Rhs, Self::Output>>
    where
        Rhs: Copy,
    {
        let _ = h;
        None
    }
}

/// A sliding dot product kernel resolved ahead of time by
/// [`DotProd::plan_block`]. Returns number of outputs written. Some
/// block kernels may not execute over the full input. Use singular
/// kernel for remaining inputs.
/// 
/// `Inputs` is a slice of inputs, while `Coeff` and `Out` are singular
pub type DotProdBlockKernel<Inputs, Coeff, Out> =
    unsafe fn(&Inputs, &[Coeff], &mut [Out]) -> usize;

/// Prepared coefficients and executor for sliding block execution.
#[doc(hidden)]
pub struct DotProdBlockPlan<Elem: ?Sized, Rhs, Out> {
    // A block kernel may layout its coefficients in a different order
    pub(crate) h: Vec<Rhs>,
    pub(crate) input_width: usize,
    pub(crate) output_width: usize,
    pub(crate) executor: DotProdBlockKernel<Elem, Rhs, Out>,
}

impl<Elem: ?Sized, Rhs: Clone, Out> Clone for DotProdBlockPlan<Elem, Rhs, Out> {
    fn clone(&self) -> Self {
        Self {
            h: self.h.clone(),
            input_width: self.input_width,
            output_width: self.output_width,
            executor: self.executor,
        }
    }
}

impl<Elem: ?Sized, Rhs: std::fmt::Debug, Out> std::fmt::Debug
    for DotProdBlockPlan<Elem, Rhs, Out>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DotProdBlockPlan")
            .field("h", &self.h)
            .field("input_width", &self.input_width)
            .field("output_width", &self.output_width)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "simd")]
impl<Elem: ?Sized, Rhs, Out> DotProdBlockPlan<Elem, Rhs, Out> {
    pub(super) fn new(
        h: Vec<Rhs>,
        input_width: usize,
        output_width: usize,
        executor: DotProdBlockKernel<Elem, Rhs, Out>,
    ) -> Self {
        Self { h, input_width, output_width, executor }
    }
}
