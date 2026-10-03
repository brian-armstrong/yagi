//! Numerical optimization: search objects and sample objectives.
//!
//! Each search object owns a caller-supplied utility function and minimizes
//! or maximizes it according to an [`OptimDirection`]. The objects keep their
//! current candidate between calls and advance through single steps or
//! bounded runs. None of them checks for convergence, so stopping at an
//! iteration limit says nothing about whether an optimum was found.
//!
//! [`GradientSearch`] and [`NewtonSearch`] search a vector of `f32`
//! parameters, [`QuadSectionSearch`] a single `f32`, and [`GeneticSearch`] a
//! [`Chromosome`] of bit-encoded traits. [`rosenbrock`], [`invgauss`],
//! [`multimodal`] and [`spiral`] are objectives for trying them out.

mod chromosome;
mod common;
mod genetic_search;
mod gradient_search;
mod newton_search;
mod quad_section_search;
mod utilities;

pub use chromosome::Chromosome;
pub use genetic_search::GeneticSearch;
pub use gradient_search::GradientSearch;
pub use newton_search::NewtonSearch;
pub use quad_section_search::{OptimDirection, QuadSectionSearch};
pub use utilities::{invgauss, multimodal, rosenbrock, spiral};
