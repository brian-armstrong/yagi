//! Adaptive FIR equalizers for real and complex samples.
//!
//! [`LeastMeanSquaresEqualizer`] adapts its weights with a normalized
//! least-mean-squares rule. [`RecursiveLeastSquaresEqualizer`] uses a
//! recursive least-squares update, which converges in fewer samples at a
//! higher cost per sample. Both keep a history of received samples. The usual
//! cycle is to push a sample, compute its estimate, then update the weights
//! from a known or decided desired value.
//!
//! The two equalizers do not order their weights the same way, and their
//! getters and setters do not all pair up. FIR order puts the weight for the
//! newest sample first. History order puts the weight for the oldest sample
//! first.
//!
//! - LMS constructors and `set_coefficients` take FIR order. LMS `weights()`
//!   returns FIR order and round-trips through `set_coefficients`. LMS
//!   `coefficients()` returns the internal form, which is reversed and
//!   conjugated, and does not.
//! - RLS `from_coefficients` and `set_coefficients` take history order. RLS
//!   `weights()` and `train` use FIR order. Reverse the output of `weights()`
//!   before passing it back to either setter.

mod least_mean_squares;
mod recursive_least_squares;

pub use least_mean_squares::LeastMeanSquaresEqualizer;
pub use recursive_least_squares::RecursiveLeastSquaresEqualizer;
