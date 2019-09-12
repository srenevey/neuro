pub mod dense;
pub mod batch_normalization;
pub mod conv2d;

use arrayfire::*;

pub enum Initializer {
    Zeros,
    Ones,
    Constant(f64),
    RandomUniform,
    RandomNormal,
    XavierNormal,
    XavierUniform,
    HeNormal,
    HeUniform,
}


pub trait Layer {
    fn initialize_parameters(&mut self, fan_in: u64);
    fn compute_activation(&self, prev_activation: &Array<f64>) -> Array<f64>;
    fn compute_activation_mut(&mut self, prev_activation: &Array<f64>) -> Array<f64>;
    fn fan_out(&self) -> u64;
    fn compute_da_prev_mut(&mut self, dz: &Array<f64>) -> Array<f64>;
    fn parameters(&self) -> Vec<&Array<f64>>;
    fn dparameters(&self) -> Vec<&Array<f64>>;
    fn set_parameters(&mut self, parameters: Vec<Array<f64>>);
}