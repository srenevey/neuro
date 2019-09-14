pub use self::image_data::ImageDataSet;
pub use self::tabular_data::TabularDataSet;

pub mod batch_iterator;
pub mod tabular_data;
pub mod image_data;

use arrayfire::*;
use std::io;
use std::fmt;


#[derive(Debug)]
pub enum DataSetError {
    Io(io::Error),
    Csv(csv::Error),
    DimensionMismatch,
    PathDoesNotExist,
    TrainPathDoesNotExist,
    ValidPathDoesNotExist,
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
        }
    }
}

impl std::convert::From<io::Error> for DataSetError {
    fn from(error: io::Error) -> DataSetError {
        DataSetError::Io(error)
    }
}


enum Scaling {
    Normalized,
    Standarized,
}

pub trait DataSet {
    fn num_features(&self) -> u64;
    fn num_outputs(&self) -> u64;
    fn num_train_samples(&self) -> u64;
    fn shuffle(&mut self);
    fn x_train(&self) -> &Vec<Array<f64>>;
    fn y_train(&self) -> &Vec<Array<f64>>;
    fn x_valid(&self) -> &Array<f64>;
    fn y_valid(&self) -> &Array<f64>;
}