//! Initializers

use arrayfire::*;
use crate::tensor::*;
use crate::tensor::PrimitiveType;

/// Used to generate the initial values for the parameters of the model.
#[derive(Debug, Copy, Clone)]
pub enum Initializer {
    /// Given constant value.
    Constant(f64),
    /// Normal distribution scaled using Glorot scale factor.
    GlorotNormal,
    /// Uniform distribution scaled using Glorot scale factor.
    GlorotUniform,
    /// Normal distribution scaled using He scale factor.
    HeNormal,
    /// Uniform distribution scaled using He scale factor.
    HeUniform,
    /// Normal distribution scaled using Lecun scale factor.
    LecunNormal,
    /// Uniform distribution scaled using Lecun scale factor.
    LecunUniform,
    /// Normal distribution with mean 0 and standard deviation 0.01.
    Normal,
    /// Normal distribution with mean 0 and given standard deviation.
    NormalScaled(f64),
    /// Ones.
    Ones,
    /// Uniform distribution within -0.01 and 0.01.
    Uniform,
    /// Uniform distribution within the given bounds.
    UniformBounded(f64, f64),
    /// Zeros.
    Zeros,
}


impl Initializer {

    /// Creates a tensor with random values generated from the distribution specified by the initializer.
    ///
    /// # Arguments
    /// * `dims`: dimensions of the tensor created
    /// * `fan_in`: number of input units
    /// * `fan_out`: number of output units
    ///
    pub(crate) fn new(self,
                      dims: Dim4,
                      fan_in: u64,
                      fan_out: u64
    ) -> Tensor {
        match self {
            Initializer::Constant(x) => constant(x as PrimitiveType, dims),
            Initializer::GlorotNormal => {
                let standard_deviation = (2. / (fan_out + fan_in) as PrimitiveType).sqrt();
                Tensor::scaled_normal(0 as PrimitiveType, standard_deviation, dims)
            },
            Initializer::GlorotUniform => {
                let limit = (6. / (fan_out + fan_in) as PrimitiveType).sqrt();
                Tensor::scaled_uniform(-limit, limit, dims)
            },
            Initializer::HeNormal => {
                let standard_deviation = (2. / fan_in as PrimitiveType).sqrt();
                Tensor::scaled_normal(0 as PrimitiveType, standard_deviation, dims)
            },
            Initializer::HeUniform => {
                let limit = (6. / fan_in as PrimitiveType).sqrt();
                Tensor::scaled_uniform(-limit, limit, dims)
            },
            Initializer::LecunNormal => {
                let standard_deviation = (1. / fan_in as PrimitiveType).sqrt();
                Tensor::scaled_normal(0 as PrimitiveType, standard_deviation, dims)
            },
            Initializer::LecunUniform => {
                let limit = (3. / fan_in as PrimitiveType).sqrt();
                Tensor::scaled_uniform(-limit, limit, dims)
            },
            Initializer::Normal => Tensor::scaled_normal(0 as PrimitiveType, 0.01, dims),
            Initializer::NormalScaled(standard_deviation) => Tensor::scaled_normal(0 as PrimitiveType, standard_deviation as PrimitiveType, dims),
            Initializer::Ones => Tensor::ones(dims),
            Initializer::Uniform => Tensor::scaled_uniform(-0.01, 0.01, dims),
            Initializer::UniformBounded(lb, ub) => Tensor::scaled_uniform(lb as PrimitiveType, ub as PrimitiveType, dims),
            Initializer::Zeros => Tensor::zeros(dims),
        }
    }


    pub(crate) fn id(&self) -> u64 {
        match self {
            Initializer::Constant(_) => 0,
            Initializer::GlorotNormal => 1,
            Initializer::GlorotUniform => 2,
            Initializer::HeNormal => 3,
            Initializer::HeUniform => 4,
            Initializer::LecunNormal => 5,
            Initializer::LecunUniform => 6,
            Initializer::Normal => 7,
            Initializer::NormalScaled(_) => 8,
            Initializer::Ones => 9,
            Initializer::Uniform => 10,
            Initializer::UniformBounded(_,_) => 11,
            Initializer::Zeros => 12,
        }
    }

}