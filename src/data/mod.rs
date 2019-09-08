pub mod batch_iterator;
pub mod tabular_data;

use arrayfire::*;
use std::io;
use std::fmt;


#[derive(Debug)]
pub enum DataSetError {
    Io(io::Error),
    Csv(csv::Error),
    DimensionMismatch,
}

impl fmt::Display for DataSetError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DataSetError::Io(ref err) => write!(f, "IO error: {}", err),
            DataSetError::Csv(ref err) => write!(f, "CSV error: {}", err),
            DataSetError::DimensionMismatch => write!(f, "The number of input and output samples differ.")
        }
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