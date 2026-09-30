//! Dot products and sum-of-squares operations for real and complex samples.
//!
//! [`DotProd`] computes `sum(x[i] * h[i])` over equal-length inputs. Complex
//! products are bilinear: neither operand is conjugated. [`DotProduct`] owns
//! coefficients for repeated single-window or sliding operations.
//! Built-in kernels accumulate in `f32`. Rounding can vary with input length,
//! the entry point ([`DotProd::dotprod`], [`DotProduct::execute`], or
//! [`DotProduct::execute_block`]), and the SIMD features available on the CPU.
//! Compare results with a numerical tolerance rather than exact equality.
//! Large products or sums can overflow to infinity.

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

/// A dot-product kernel resolved by [`DotProd::plan`].
///
/// Most callers should use [`DotProd::dotprod`] or [`DotProduct`] instead.
///
/// # Safety
///
/// Call this only with inputs and coefficients whose lengths exactly equal the
/// length passed to [`DotProd::plan`]. A planned kernel can use unchecked
/// indexing or instructions selected for the current CPU.
pub type DotProdKernel<Inputs, Rhs, Out> = unsafe fn(&Inputs, &[Rhs]) -> Out;

/// Computes a dot product between two sample sequences.
///
/// The result is `sum(self[i] * other[i])`. Implementations cover `f32` and
/// `Complex32` slices, with real and complex coefficient combinations. Complex
/// multiplication does not conjugate either input. Implementations also accept
/// supported sample types in a [`std::collections::VecDeque`] as the first
/// operand.
///
/// The built-in implementations can accumulate in a different order from
/// [`DotProduct::execute`] and can select different SIMD kernels on different
/// CPUs. Results for the same inputs need not match bit for bit.
pub trait DotProd<Rhs> {
    type Output;

    /// Computes the dot product of two equal-length sequences.
    ///
    /// The built-in implementations return zero for two empty sequences.
    ///
    /// # Panics
    ///
    /// The built-in implementations panic if the operands have different
    /// lengths.
    fn dotprod(&self, other: &[Rhs]) -> Self::Output;

    /// Resolves a kernel for dot products with `len` elements.
    ///
    /// This avoids repeated runtime dispatch when the length is fixed. The
    /// returned function remains valid for the life of the process. SIMD
    /// implementations may select it using CPU features detected on this
    /// machine.
    ///
    /// # Safety
    ///
    /// The returned kernel must receive a first operand and a coefficient
    /// slice of exactly `len` elements each.
    fn plan(len: usize) -> DotProdKernel<Self, Rhs, Self::Output> {
        // default impl
        let _ = len;
        |x, h| x.dotprod(h)
    }

    /// Resolves a packed sliding-dot-product strategy when one is available.
    ///
    /// This is an internal extension point used by [`DotProduct`].
    #[doc(hidden)]
    fn plan_block(len: usize) -> Option<DotProdBlockPlan<Self, Rhs, Self::Output>> {
        let _ = len;
        None
    }
}

/// A planned kernel for a sliding sequence of dot products.
///
/// It returns the number of outputs written. A block kernel can leave a tail
/// for a singular kernel to process.
///
/// `Inputs` is a sample slice. `Rhs` and `Out` are individual coefficients and
/// results.
///
/// # Safety
///
/// The input, packed coefficients, and output must satisfy the layout and
/// length requirements of the [`DotProdBlockPlan`] that supplied this kernel.
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
