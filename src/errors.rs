//! Errors that may be returned by methods in the crate.
use crate::data;
use std::fmt;

#[derive(Debug)]
pub enum NeuroError {
    DataSetError(data::DataSetError),
    NoLayer,
}

impl fmt::Display for NeuroError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            NeuroError::DataSetError(ref err) => write!(f, "DataSetError: {}", err),
            NeuroError::NoLayer => write!(f, "The network doesn't contain any layer."),
        }
    }
}

impl std::convert::From<data::DataSetError> for NeuroError {
    fn from(error: data::DataSetError) -> NeuroError {
        NeuroError::DataSetError(error)
    }
}