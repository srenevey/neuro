//! Dropout layer
use arrayfire::*;
use rand::prelude::*;
use std::fmt;

use crate::errors::Error;
use crate::io::{write_scalar, read_scalar};
use crate::layers::Layer;
use crate::tensor::*;

/// Defines a dropout layer.
pub struct Dropout {
    drop_rate: f64,
    output_shape: Dim,
    grad: Tensor,
    random_engine: RandomEngine,
    scaling_factor: PrimitiveType
}

impl Dropout {

    pub(crate) const NAME: &'static str = "Dropout";

    /// Creates a dropout layer.
    ///
    /// # Arguments
    ///
    /// * `drop_rate` - The probability that a unit will be dropped.
    ///
    /// # Panics
    ///
    /// The method panics if `rate` is smaller than 0 or greater than 1.
    pub fn new(drop_rate: f64) -> Box<Dropout> {

        if drop_rate < 0. || drop_rate > 1. {
            panic!("The drop rate is invalid.");
        }

        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        let random_engine = RandomEngine::new(RandomEngineType::PHILOX_4X32_10, Some(seed));

        let scaling_factor = 1. / (1. - drop_rate) as PrimitiveType;

        Box::new(Dropout {
            drop_rate,
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            grad: Tensor::new_empty_tensor(),
            random_engine,
            scaling_factor,
        })
    }

    /// Generates a binomial mask to let some values pass through the layer.
    fn generate_binomial_mask(&self, dims: Dim4) -> Tensor {
        let random_values = random_uniform::<f64>(dims, &self.random_engine);
        let cond = gt(&random_values, &self.drop_rate, true);
        cond.cast()
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<Self> {
        let _ = hdf5::silence_errors();
        let drop_rate = group.dataset("drop_rate").and_then(|ds| Ok(read_scalar::<f64>(&ds))).expect("Could not retrieve the drop rate.");
        let output_shape = group.dataset("output_shape").and_then(|value| value.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape.");

        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        let random_engine = RandomEngine::new(RandomEngineType::PHILOX_4X32_10, Some(seed));

        let scaling_factor = 1. / (1. - drop_rate) as PrimitiveType;

        Box::new(Self {
            drop_rate,
            output_shape: Dim::new(&(output_shape[0])),
            grad: Tensor::new_empty_tensor(),
            random_engine,
            scaling_factor,
        })
    }
}

impl Layer for Dropout {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn initialize_parameters(&mut self, input_shape: Dim4) {
        self.output_shape = input_shape;
    }

    fn compute_activation(&self, prev_activation: &Tensor) -> Tensor {
        prev_activation.copy()
    }

    fn compute_activation_mut(&mut self, prev_activation: &Tensor) -> Tensor {
        let mask = self.generate_binomial_mask(prev_activation.dims());
        let output = prev_activation * &mask;
        self.grad = mask;

        // Inverted dropout
        &output * self.scaling_factor
    }

    fn compute_dactivation_mut(&mut self, dz: &Tensor) -> Tensor {
        &self.grad * dz * self.scaling_factor
    }

    fn output_shape(&self) -> Dim4 {
        self.output_shape
    }


    fn save(&self, group: &hdf5::Group, layer_number: usize) -> Result<(), Error> {
        let group_name = layer_number.to_string() + &String::from("_") + Self::NAME;
        let dropout = group.create_group(&group_name)?;

        let drop_rate = dropout.new_dataset::<f64>().create("drop_rate", 1)?;
        write_scalar(&drop_rate, &self.drop_rate);

        let output_shape = dropout.new_dataset::<[u64; 4]>().create("output_shape", 1)?;
        output_shape.write(&[*self.output_shape.get()])?;

        Ok(())
    }
}

impl fmt::Display for Dropout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} \t 0  \t\t [{}, {}, {}]", Self::NAME, self.output_shape[0], self.output_shape[1], self.output_shape[2])
    }
}