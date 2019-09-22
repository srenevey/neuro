//! Define the different types of layers

// Public re-exports
pub use self::batch_normalization::BatchNormalization;
pub use self::conv2d::Conv2D;
pub use self::conv2d::ConvMode;
pub use self::dense::Dense;
pub use self::max_pooling::MaxPooling2D;

pub mod dense;
pub mod batch_normalization;
pub mod conv2d;
pub mod max_pooling;
pub mod initializers;

use arrayfire::*;

pub trait Layer {
    fn initialize_parameters(&mut self, input_shape: Dim4);
    fn compute_activation(&self, prev_activation: &Array<f64>) -> Array<f64>;
    fn compute_activation_mut(&mut self, prev_activation: &Array<f64>) -> Array<f64>;
    fn output_shape(&self) -> Dim4;
    fn compute_dactivation_mut(&mut self, dz: &Array<f64>) -> Array<f64>;
    fn parameters(&self) -> Vec<&Array<f64>>;
    fn dparameters(&self) -> Vec<&Array<f64>>;
    fn set_parameters(&mut self, parameters: Vec<Array<f64>>);
}