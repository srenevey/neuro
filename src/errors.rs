//! Errors that may be returned by methods in the crate.
use std::fmt;

use crate::data;

#[derive(Debug)]
pub enum Error {
    DataSetError(data::DataSetError),
    HDF5Error(hdf5::Error),
    InvalidInputShape,
    InvalidOutputShape,
    NoLayer,
    UnknownLayer,
    UnknownOptimizer,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::DataSetError(ref err) => write!(f, "DataSetError: {}", err),
            Error::HDF5Error(ref err) => write!(f, "HDF5Error: {}", err),
            Error::InvalidInputShape => write!(f, "The input shape of the network must be a slice with 1, 2, or 3 elements."),
            Error::InvalidOutputShape => write!(f, "The output shape of the network is invalid."),
            Error::NoLayer => write!(f, "The network doesn't contain any layer."),
            Error::UnknownLayer => write!(f, "The type of layer is unknown."),
            Error::UnknownOptimizer => write!(f, "The type of optimizer is unknown."),
        }
    }
}

impl std::convert::From<data::DataSetError> for Error {
    fn from(error: data::DataSetError) -> Error {
        Error::DataSetError(error)
    }
}

impl std::convert::From<hdf5::Error> for Error {
    fn from(error: hdf5::Error) -> Error {
        Error::HDF5Error(error)
    }
}
