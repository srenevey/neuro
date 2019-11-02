//! Define the different types of layers

// Public re-exports
pub use self::batch_normalization::BatchNormalization;
pub use self::conv2d::Conv2D;
pub use self::conv2d::Padding;
pub use self::dense::Dense;
pub use self::dropout::Dropout;
pub use self::initializers::Initializer;
pub use self::max_pooling::MaxPooling2D;

pub mod batch_normalization;
pub mod conv2d;
pub mod dense;
pub mod dropout;
pub mod initializers;
pub mod max_pooling;

use crate::regularizers::*;

use std::fs;
use std::io;
use std::io::BufWriter;
use arrayfire::*;
use crate::Tensor;

pub trait Layer: std::fmt::Display {
    fn initialize_parameters(&mut self, input_shape: Dim4);
    fn compute_activation(&self, input: &Tensor) -> Tensor;
    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor;
    fn compute_dactivation_mut(&mut self, input: &Tensor) -> Tensor;
    fn output_shape(&self) -> Dim4;
    fn parameters(&self) -> Option<Vec<&Tensor>> { None }

    /// Returns the parameters and their derivatives.
    fn parameters_mut(&mut self) -> Option<(Vec<&mut Tensor>, Vec<&Tensor>)> { None }
    fn dparameters(&self) -> Option<Vec<&Tensor>> { None }
    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()>;
    fn set_regularizer(&mut self, regularizer: Option<Regularizer>) {}
    fn print(&self) {}
}

/*
pub trait Layer: std::fmt::Display {
    fn initialize_parameters(&mut self, input_shape: Dim4);
    fn compute_activation(&self, input: &Array<f64>) -> Array<f64>;
    fn compute_activation_mut(&mut self, input: &Array<f64>) -> Array<f64>;
    fn compute_dactivation_mut(&mut self, input: &Array<f64>) -> Array<f64>;
    fn output_shape(&self) -> Dim4;
    fn parameters(&self) -> Option<Vec<&Array<f64>>>;
    fn dparameters(&self) -> Option<Vec<&Array<f64>>>;
    fn set_parameters(&mut self, parameters: Vec<Array<f64>>);
    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()>;
    fn set_regularizer(&mut self, regularizer: Option<Regularizer>) {}
    fn print(&self) {}
}
*/

