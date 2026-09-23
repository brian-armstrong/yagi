//! Adaptive finite-impulse-response equalizers for real and complex samples.
//!
//! [`LeastMeanSquaresEqualizer`] updates weights with a normalized
//! least-mean-squares rule. [`RecursiveLeastSquaresEqualizer`] uses a recursive
//! least-squares update. Both retain an input history. Push a sample, compute
//! its estimate, then update with a known or estimated desired value.

mod least_mean_squares;
mod recursive_least_squares;

pub use least_mean_squares::LeastMeanSquaresEqualizer;
pub use recursive_least_squares::RecursiveLeastSquaresEqualizer;
