//! Dropout layer
use crate::layers::Layer;
use crate::tensor::*;

use std::fmt;
use std::fs;
use std::io;
use std::io::BufWriter;

use arrayfire::*;
use rand::prelude::*;


/// Defines a dropout layer.
pub struct Dropout {
    drop_rate: f64,
    output_shape: Dim4,
    grad: Tensor,
    random_engine: RandomEngine,
    scaling_factor: PrimitiveType
}

impl Dropout {
    /// Creates a dropout layer.
    ///
    /// # Arguments
    /// * `rate`: probability that a unit will be dropped. Must be between 0 and 1.
    ///
    /// # Panics
    /// The method panics if `rate` is smaller than 0 or greater than 1.
    ///
    pub fn new(rate: f64) -> Box<Dropout> {

        if rate < 0. || rate > 1. {
            panic!("The drop rate is invalid.");
        }

        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        let random_engine = RandomEngine::new(RandomEngineType::PHILOX_4X32_10, Some(seed));

        let scaling_factor = 1. / (1. - rate) as PrimitiveType;

        Box::new(Dropout {
            drop_rate: rate,
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
}

impl Layer for Dropout {
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

    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        Ok(())
    }
}

impl fmt::Display for Dropout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dropout \t 0")
    }
}