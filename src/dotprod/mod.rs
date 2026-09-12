// Dotprod module
// Current state:
// - Dotprod ready to use (+autotests)
// - sumsq ready to use (+autotests)

mod complex_complex;
mod complex_real;
mod dot_product;
mod real_real;
mod sumsq;

pub use sumsq::{sumsqcf, sumsqf};

#[cfg(feature = "simd")]
mod complex_complex_block;
#[cfg(feature = "simd")]
mod complex_real_block;
#[cfg(feature = "simd")]
mod real_real_block;
#[cfg(feature = "simd")]
mod reduce;

pub use dot_product::DotProduct;
pub(crate) use dot_product::DotProductPlan;

/// A dot product kernel resolved ahead of time by [`DotProd::plan`].
pub type DotProdKernel<Inputs, Rhs, Out> = unsafe fn(&Inputs, &[Rhs]) -> Out;

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
    fn plan(len: usize) -> DotProdKernel<Self, Rhs, Self::Output> {
        // default impl
        let _ = len;
        |x, h| x.dotprod(h)
    }

    /// Optionally resolve the kernel and coefficient layout for a block of
    /// dot products with `len` coefficients.
    #[doc(hidden)]
    fn plan_block(len: usize) -> Option<DotProdBlockPlan<Self, Rhs, Self::Output>> {
        let _ = len;
        None
    }
}

/// A sliding dot product kernel resolved ahead of time by
/// [`DotProd::plan_block`]. Returns number of outputs written. Some
/// block kernels may not execute over the full input. Use singular
/// kernel for remaining inputs.
///
/// `Inputs` is a slice of inputs, while `Rhs` and `Out` are singular.
pub type DotProdBlockKernel<Inputs, Rhs, Out> = unsafe fn(&Inputs, &[Rhs], &mut [Out]) -> usize;

type DotProdRepack<Rhs> = fn(&[Rhs], usize, &mut [Rhs]);

/// Coefficient packing and executor plan for sliding block execution.
#[doc(hidden)]
pub struct DotProdBlockPlan<Inputs: ?Sized, Rhs, Out> {
    source_len: usize,
    packed_len: usize,
    pub(crate) input_width: usize,
    pub(crate) output_width: usize,
    pub(crate) executor: DotProdBlockKernel<Inputs, Rhs, Out>,
    repack: DotProdRepack<Rhs>,
}

impl<Inputs: ?Sized, Rhs, Out> DotProdBlockPlan<Inputs, Rhs, Out> {
    pub(crate) fn repack(&self, h: &[Rhs], packed: &mut [Rhs]) {
        assert_eq!(h.len(), self.source_len, "Block plan and coefficient lengths must be equal");
        assert_eq!(packed.len(), self.packed_len, "Invalid packed coefficient length");
        (self.repack)(h, self.input_width, packed);
    }

    pub(crate) fn packed_len(&self) -> usize {
        self.packed_len
    }
}

#[cfg(feature = "simd")]
impl<Inputs: ?Sized, Rhs, Out> DotProdBlockPlan<Inputs, Rhs, Out> {
    pub(super) fn new(
        source_len: usize,
        packed_len: usize,
        input_width: usize,
        output_width: usize,
        executor: DotProdBlockKernel<Inputs, Rhs, Out>,
        repack: DotProdRepack<Rhs>,
    ) -> Self {
        Self { source_len, packed_len, input_width, output_width, executor, repack }
    }
}

impl<Inputs: ?Sized, Rhs, Out> Clone for DotProdBlockPlan<Inputs, Rhs, Out> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Inputs: ?Sized, Rhs, Out> Copy for DotProdBlockPlan<Inputs, Rhs, Out> {}

impl<Inputs: ?Sized, Rhs, Out> std::fmt::Debug for DotProdBlockPlan<Inputs, Rhs, Out> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DotProdBlockPlan")
            .field("source_len", &self.source_len)
            .field("packed_len", &self.packed_len)
            .field("input_width", &self.input_width)
            .field("output_width", &self.output_width)
            .finish_non_exhaustive()
    }
}
