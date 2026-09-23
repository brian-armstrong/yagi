//! Shared error types for fallible yagi operations.

use std::error;
use std::fmt;

/// Result type returned by fallible yagi operations.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Error returned by a yagi operation.
///
/// The variant classifies the broad cause. The contained message supplies the
/// operation-specific detail. Callers should normally handle the error at the
/// operation boundary instead of relying on a particular message.
pub enum Error {
    /// An internal invariant was not met.
    ///
    /// This usually indicates inconsistent internal state rather than invalid
    /// caller input.
    Internal(String),
    /// A constructor or configuration request is invalid.
    Config(String),
    /// An input value is invalid for the requested operation.
    Value(String),
    /// An input is outside the supported range or has an invalid length.
    Range(String),
    /// The operation is not available in the current mode or state.
    Mode(String),
    /// An iterative numerical algorithm did not converge.
    NoConvergence(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Internal(s) => write!(f, "{}", s),
            Error::Config(s) => write!(f, "{}", s),
            Error::Value(s) => write!(f, "{}", s),
            Error::Range(s) => write!(f, "{}", s),
            Error::Mode(s) => write!(f, "{}", s),
            Error::NoConvergence(s) => write!(f, "{}", s),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Error::Internal(..) => None,
            Error::Config(..) => None,
            Error::Value(..) => None,
            Error::Range(..) => None,
            Error::Mode(..) => None,
            Error::NoConvergence(..) => None,
        }
    }
}
