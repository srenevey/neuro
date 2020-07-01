//! Simple interfaces to import data.
use std::fmt;
use std::io;

use crate::tensor::*;

pub(crate) use self::batch_iterator::BatchIterator;
pub use self::image_data::ImageDataSet;
pub use self::image_data::ImageDataSetBuilder;
pub use self::image_data::ImageOps;
pub use self::tabular_data::TabularDataSet;

mod batch_iterator;
mod image_data;
mod tabular_data;

/// Errors that may be raised by data sets methods.
#[derive(Debug)]
pub enum DataSetError {
    Io(io::Error),
    Csv(csv::Error),
    DimensionMismatch,
    PathDoesNotExist,
    TrainPathDoesNotExist,
    ValidPathDoesNotExist,
    ImageFormatNotSupported,
    InvalidImagePath,
    InvalidValidationFraction,
    DifferentNumbersOfChannels,
}

/// Types of data.
#[derive(Debug)]
enum IO {
    Input,
    Output,
}

impl fmt::Display for DataSetError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DataSetError::Io(ref err) => write!(f, "IO error: {}", err),
            DataSetError::Csv(ref err) => write!(f, "CSV error: {}", err),
            DataSetError::DimensionMismatch => write!(f, "The number of input and output samples differ."),
            DataSetError::PathDoesNotExist => write!(f, "The path does not exist."),
            DataSetError::TrainPathDoesNotExist => write!(f, "The root directory does not contain a 'train' subfolder."),
            DataSetError::ValidPathDoesNotExist => write!(f, "The root directory does not contain a 'valid' subfolder."),
            DataSetError::ImageFormatNotSupported => write!(f, "The image format is not supported."),
            DataSetError::InvalidImagePath => write!(f, "The path could not be opened as an image."),
            DataSetError::InvalidValidationFraction => write!(f, "The validation fraction is incorrect. It must be between 0 and 1."),
            DataSetError::DifferentNumbersOfChannels => write!(f, "The directory contains images with different numbers of channels."),
        }
    }
}

impl std::convert::From<io::Error> for DataSetError {
    fn from(error: io::Error) -> DataSetError {
        DataSetError::Io(error)
    }
}

/// Defines the type of scaling that has been performed.
#[derive(Debug)]
pub enum Scaling {
    Normalized,
    Standardized,
}


/// Trait that must be implemented for any type of dataset supported by neuro.
pub trait DataSet {
    /// Returns the dimension of the samples.
    fn input_shape(&self) -> Dim;

    /// Returns the dimension of the labels.
    fn output_shape(&self) -> Dim;

    /// Returns the number of samples in the training set.
    fn num_train_samples(&self) -> u64;

    /// Returns the number of samples in the validation set.
    fn num_valid_samples(&self) -> u64;

    /// Returns the classes in the data set.
    fn classes(&self) -> Option<Vec<String>> { None }

    /// Returns a reference to the training samples.
    fn x_train(&self) -> &Tensor;

    /// Returns a reference to the training labels.
    fn y_train(&self) -> &Tensor;

    /// Returns a reference to the validation samples.
    fn x_valid(&self) -> Option<&Tensor>;

    /// Returns a reference to the validation labels.
    fn y_valid(&self) -> Option<&Tensor>;

    /// Returns a reference to the test samples.
    fn x_test(&self) -> Option<&Tensor>;

    /// Returns a reference to the test labels.
    fn y_test(&self) -> Option<&Tensor>;

    /// Returns a reference to the type of scaling that has been applied to the input features and the values used for the scaling.
    ///
    /// If scaling has been applied, the returned tuple contains the following:
    /// * Normalization: (Scaling::Normalized, minimum value, maximum value)
    /// * Standardization: (Scaling::Standardized, mean, standard deviation)
    ///
    fn x_train_stats(&self) -> &Option<(Scaling, Tensor, Tensor)>;

    /// Returns a reference to the type of scaling that has been applied to the output labels and the values used for the scaling.
    ///
    /// If scaling has been applied, the returned tuple contains the following:
    /// * Normalization: (Scaling::Normalized, minimum value, maximum value)
    /// * Standardization: (Scaling::Standardized, mean, standard deviation)
    ///
    fn y_train_stats(&self) -> &Option<(Scaling, Tensor, Tensor)>;
}