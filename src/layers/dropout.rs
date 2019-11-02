//! # Dropout
//!
//! This layer performs dropout operation.

use crate::layers::Layer;
use crate::Tensor;
use crate::tensor::*;
use arrayfire::*;
use rand::prelude::*;
use std::io;
use std::io::BufWriter;
use std::fs;
use std::fmt;

pub struct Dropout {
    drop_rate: f64,
    output_shape: Dim4,
    grad: Tensor,
    //random_engine: RandomEngine,
}

impl Dropout {
    /// Create a dropout layer.
    ///
    /// # Arguments
    /// * `rate`: probability that a unit will be dropped
    ///
    pub fn new(rate: f64) -> Box<Dropout> {
        /*
        let mut rng = rand::thread_rng();
        let seed: u64 = rng.gen();
        let random_engine = RandomEngine::new(RandomEngineType::PHILOX_4X32_10, None);
        */

        Box::new(Dropout {
            drop_rate: rate,
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            grad: Tensor::new_empty_tensor(),
            //random_engine,
        })
    }

    fn generate_binomial_mask(&self, dims: Dim4) -> Tensor {
        let height = dims.get()[0];
        let width = dims.get()[1];
        let num_channels = dims.get()[2];
        let mb_size = dims.get()[3];

        let mut rng = rand::thread_rng();
        let num_inputs = height * width;
        let num_ones = ((1. - self.drop_rate) * height as f64 * width as f64).floor() as u64;
        let num_zeros = num_inputs - num_ones;

        let num_2d_slices = num_channels * mb_size;
        let mut values: Vec<PrimitiveType> = Vec::new();

        for _ in 0..num_2d_slices {
            let mut tmp = Vec::with_capacity(num_inputs as usize);
            for _ in 0..num_ones {
                tmp.push(1.0);
            }
            for _ in 0..num_zeros {
                tmp.push(0.0);
            }
            tmp.shuffle(&mut rng);
            values.extend(&tmp);
        }
        Tensor::new(&values[..], dims)

        // Alternatively
        //let random_values = random_uniform::<f64>(dims, &self.random_engine);
        //let cond = gt(&random_values, &self.rate, true);
        //let ones = constant(1.0f64, dims);
        //selectr(&ones, &cond, 0.0);
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
        let output = mul(prev_activation, &mask, true);
        self.grad = mask;

        // Inverted dropout
        let scale = 1. / (1. - self.drop_rate) as PrimitiveType;
        mul(&output, &scale, true)
    }

    fn compute_dactivation_mut(&mut self, _dz: &Tensor) -> Tensor {
        let scale = 1. / (1. - self.drop_rate) as PrimitiveType;
        mul(&self.grad, &scale, true)
    }

    fn output_shape(&self) -> Dim4 {
        self.output_shape
    }

    fn parameters(&self) -> Option<Vec<&Tensor>> {
        None
    }

    fn parameters_mut(&mut self) -> Option<(Vec<&mut Tensor>, Vec<&Tensor>)> {
        None
    }

    fn dparameters(&self) -> Option<Vec<&Tensor>> {
        None
    }

    fn set_parameters(&mut self, parameters: Vec<Tensor>) {

    }

    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        Ok(())
    }
}

impl fmt::Display for Dropout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        Ok(())
    }
}