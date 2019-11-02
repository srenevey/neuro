use arrayfire::*;
use crate::tensor::*;
use crate::tensor::PrimitiveType;

#[derive(Debug, Copy, Clone)]
pub enum Initializer {
    /// Initialize the parameters with zeros
    Zeros,
    /// Initialize the parameters with ones
    Ones,
    /// Initialize the parameters with a specified constant
    Constant(PrimitiveType),
    /// Initialize the parameters with values drawn from a random normal distribution
    RandomNormal,
    /// Initialize the parameters with values drawn from a random uniform distribution
    RandomUniform,
    /// Initialize the parameters with values drawn from a random normal distribution scaled using Glorot scale factor
    GlorotNormal,
    /// Initialize the parameters with values drawn from a random uniform distribution scaled using Glorot scale factor
    GlorotUniform,
    /// Initialize the parameters with values drawn from a random normal distribution scaled using He scale factor
    HeNormal,
    /// Initialize the parameters with values drawn from a random uniform distribution scaled using He scale factor
    HeUniform,
    /// Initialize the parameters with values drawn from a random normal distribution scaled using Lecun scale factor
    LecunNormal,
    /// Initialize the parameters with values drawn from a random uniform distribution scaled using Lecun scale factor
    LecunUniform,
}

impl Initializer {
    pub(crate) fn new(self, dims: Dim4, fan_in: u64, fan_out: u64) -> Tensor {
        match self {
            Initializer::Zeros  => Tensor::zeros(dims),
            Initializer::Ones   => Tensor::ones(dims),
            Initializer::Constant(x) => constant(x, dims),
            Initializer::RandomUniform  => randu(dims),
            Initializer::RandomNormal   => randn(dims),
            Initializer::GlorotNormal => {
                let standard_deviation = (2. / (fan_out + fan_in) as PrimitiveType).sqrt();
                Tensor::scaled_normal(0 as PrimitiveType, standard_deviation, dims)
            },
            Initializer::GlorotUniform => {
                let limit = (6. / (fan_out + fan_in) as PrimitiveType).sqrt();
                Tensor::scaled_uniform(-limit, limit, dims)
            },
            Initializer::HeNormal     => {
                let standard_deviation = (2. / fan_in as PrimitiveType).sqrt();
                Tensor::scaled_normal(0 as PrimitiveType, standard_deviation, dims)
            },
            Initializer::HeUniform     => {
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
            }
        }
    }

    pub(crate) fn id(&self) -> u64 {
        match self {
            Initializer::Zeros              => 0,
            Initializer::Ones               => 1,
            Initializer::Constant(x) => 2,
            Initializer::RandomUniform      => 3,
            Initializer::RandomNormal       => 4,
            Initializer::GlorotNormal       => 5,
            Initializer::GlorotUniform      => 6,
            Initializer::HeNormal           => 7,
            Initializer::HeUniform          => 8,
            Initializer::LecunNormal        => 9,
            Initializer::LecunUniform       => 10,
        }
    }
}