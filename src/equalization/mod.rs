//! Adaptive FIR equalizers for real and complex samples.
//!
//! [`LeastMeanSquaresEqualizer`] adapts its weights with a normalized
//! least-mean-squares rule. [`RecursiveLeastSquaresEqualizer`] uses a
//! recursive least-squares update, which converges in fewer samples at a
//! higher cost per sample. Both keep a history of received samples. The usual
//! cycle is to push a sample, compute its estimate, then update the weights
//! from a known or decided desired value.
//!
//! Weights are in FIR order, with the weight for the newest sample first.

mod least_mean_squares;
mod recursive_least_squares;

pub use least_mean_squares::LeastMeanSquaresEqualizer;
pub use recursive_least_squares::RecursiveLeastSquaresEqualizer;
