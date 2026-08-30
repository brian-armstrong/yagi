use std::error;
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Internal(String),
    Config(String),
    Value(String),
    Range(String),
    Mode(String),
    NoConvergence(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Internal(s) =>
                write!(f, "{}", s),
            Error::Config(s) =>
                write!(f, "{}", s),
            Error::Value(s) =>
                write!(f, "{}", s),
            Error::Range(s) =>
                write!(f, "{}", s),
            Error::Mode(s) =>
                write!(f, "{}", s),
            Error::NoConvergence(s) =>
                write!(f, "{}", s),
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