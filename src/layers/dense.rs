//! Dense layer
use crate::activations::*;
use crate::layers::*;
use crate::initializers::Initializer;
use crate::regularizers::*;
use crate::tensor::*;
use crate::tensor::Reduction;

use std::fmt;
use std::fs;
use std::io;
use std::io::Write;

use byteorder::{BigEndian, WriteBytesExt};
use arrayfire::*;


/// Defines a dense (or fully connected) layer.
///
/// This layer can be used as a conventional dense layer in feedforward neural networks or as a fully connected
/// layer in convolutional neural networks. If the inputs of the layer are not one-dimensional, they are
/// reshaped before being used to compute the activations.
///
pub struct Dense
{
    units: u64,
    activation: Activation,
    weights: Tensor,
    dweights: Tensor,
    biases: Tensor,
    dbiases: Tensor,
    input_shape: Dim4,
    linear_activation: Option<Tensor>,
    previous_input: Option<Tensor>,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
    regularizer: Option<Regularizer>,
}


impl Dense
{
    /// Creates a dense layer with given number of units and activation function.
    ///
    /// By default, the weights are initialized with a HeUniform initializer and the biases with a Zeros initializer.
    ///
    /// # Arguments
    /// * `units`: number of units in the layer
    /// * `activation`: activation function of the layer
    ///
    pub fn new(units: u64, activation: Activation) -> Box<Dense> {
        Box::new(Dense {
            units,
            activation,
            weights: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            input_shape: Dim4::new(&[0, 0, 0, 0]),
            linear_activation: None,
            previous_input: None,
            weights_initializer: Initializer::HeUniform,
            biases_initializer: Initializer::Zeros,
            regularizer: None,
        })
    }

    /// Creates a dense layer with the given parameters.
    ///
    /// # Arguments
    /// * `units`: number of units in the layer
    /// * `activation`: activation function of the layer
    /// * `weights_initializer`: initializer used to initialize the weights of the layer
    /// * `biases_initializer`: initializer used to initialize the biases of the layer
    ///
    pub fn with_param(units: u64,
                      activation: Activation,
                      weights_initializer: Initializer,
                      biases_initializer: Initializer
    ) -> Box<Dense> {
        Box::new(Dense {
            units,
            activation,
            weights: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            input_shape: Dim4::new(&[0, 0, 0, 0]),
            linear_activation: None,
            previous_input: None,
            weights_initializer,
            biases_initializer,
            regularizer: None,
        })
    }
}

impl Layer for Dense
{
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let fan_in = input_shape.get()[0] * input_shape.get()[1] * input_shape.get()[2];
        let fan_out = self.units;
        //println!("fan_in: {}, fan_out: {}", fan_in, fan_out);
        self.weights = self.weights_initializer.new(Dim4::new(&[fan_out, fan_in, 1, 1]), fan_in, fan_out);
        self.biases = self.biases_initializer.new(Dim4::new(&[fan_out, 1, 1, 1]), fan_in, fan_out);
        self.input_shape = input_shape;
    }

    fn compute_activation(&self, input: &Tensor) -> Tensor {
        let flat_input = Tensor::flatten(input);
        let linear_activation = add(&matmul(&self.weights, &flat_input, MatProp::NONE, MatProp::NONE), &self.biases, true);
        let nonlinear_activation = self.activation.eval(&linear_activation);
        nonlinear_activation.eval();
        nonlinear_activation
    }


    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor {
        let flat_input = Tensor::flatten(input);
        let linear_activation = add(&matmul(&self.weights, &flat_input, MatProp::NONE, MatProp::NONE), &self.biases, true);
        linear_activation.eval();
        let nonlinear_activation = self.activation.eval(&linear_activation);
        nonlinear_activation.eval();

        // Save input and linear activation for efficient backprop
        self.previous_input = Some(flat_input);
        self.linear_activation = Some(linear_activation);

        // Return the non linear activation
        nonlinear_activation
    }


    fn compute_dactivation_mut(&mut self, da: &Tensor) -> Tensor {
        match &self.linear_activation {
            Some(linear_activation) => {
                let dz = mul(da, &self.activation.grad(linear_activation), true);
                match &mut self.previous_input {
                    Some(previous_input) => {
                        self.dweights = matmul(&dz, previous_input, MatProp::NONE, MatProp::TRANS).reduce(Reduction::MeanBatches);

                        match self.regularizer {
                            Some(regularizer) => self.dweights += regularizer.grad(&self.weights),
                            None => {}
                        }
                        self.dbiases = dz.reduce(Reduction::MeanBatches);
                    },
                    None => panic!("The previous activations have not been computed!"),
                }
                matmul(&self.weights, &dz, MatProp::TRANS, MatProp::NONE).reshape(Dim4::new(&[self.input_shape[0], self.input_shape[1], self.input_shape[2], da.batch_size()]))
            },
            None => panic!("The linear activations z have not been computed!"),
        }
    }

    fn output_shape(&self) -> Dim4 {
        Dim4::new(&[self.units, 1, 1, 1])
    }


    fn parameters(&self) -> Option<Vec<&Tensor>> {
        Some(vec![&self.weights, &self.biases])
    }


    fn parameters_mut(&mut self) -> Option<(Vec<&mut Tensor>, Vec<&Tensor>)> {
        Some((vec![&mut self.weights, &mut self.biases], vec![&self.dweights, &self.dbiases]))
    }


    // Currently not used!
    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        writer.write(b"dense\n")?;
        writer.write(&self.units.to_be_bytes())?;
        writer.write(b"\n");
        writer.write(&self.activation.id().to_be_bytes())?;
        writer.write(b"\n");

        // Weights dimensions
        let dims: [[u8; 8]; 4] = [self.weights.dims().get()[0].to_be_bytes(), self.weights.dims().get()[1].to_be_bytes(), self.weights.dims().get()[2].to_be_bytes(), self.weights.dims().get()[3].to_be_bytes()];
        let flat: Vec<u8> = dims.concat();
        writer.write(flat.as_slice());
        writer.write(b"\n");

        // Weights
        let num_weights = (self.weights.dims().get()[0] * self.weights.dims().get()[1] * self.weights.dims().get()[2]) as usize;
        let mut buf: Vec<f64> = vec![0.; num_weights];
        self.weights.host(buf.as_mut_slice());
        for weight in buf.iter() {
            writer.write_f64::<BigEndian>(*weight);
        }
        writer.write(b"\n");

        // Biases dimensions
        writer.write(&self.biases.dims().get()[0].to_be_bytes());
        writer.write(&self.biases.dims().get()[1].to_be_bytes());
        writer.write(&self.biases.dims().get()[2].to_be_bytes());
        writer.write(&self.biases.dims().get()[3].to_be_bytes());
        writer.write(b"\n");

        // Biases
        let num_biases = (self.biases.dims().get()[0] * self.biases.dims().get()[1] * self.biases.dims().get()[2]) as usize;
        let mut buf: Vec<f64> = vec![0.; num_biases];
        self.biases.host(buf.as_mut_slice());
        for bias in buf.iter() {
            writer.write_f64::<BigEndian>(*bias);
        }
        writer.write(b"\n");

        // Input dimensions
        let dims: [[u8; 8]; 4] = [self.input_shape.get()[0].to_be_bytes(), self.input_shape.get()[1].to_be_bytes(), self.input_shape.get()[2].to_be_bytes(), self.input_shape.get()[3].to_be_bytes()];
        let flat: Vec<u8> = dims.concat();
        writer.write(flat.as_slice());
        writer.write(b"\n");

        // Weights and biases initializers
        writer.write(&self.weights_initializer.id().to_be_bytes());
        writer.write(b"\n");
        writer.write(&self.biases_initializer.id().to_be_bytes());
        writer.write(b"\n");

        Ok(())
    }


    fn set_regularizer(&mut self, regularizer: Option<Regularizer>) {
        self.regularizer = regularizer;
    }

    fn print(&self) {
        println!("Number of parameters: {}", self.weights.dims()[0] * self.weights.dims()[1] * self.weights.dims()[2] * self.weights.dims()[3] + self.biases.dims()[0]);
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Dense \t\t {}", self.weights.dims()[0] * self.weights.dims()[1] * self.weights.dims()[2] * self.weights.dims()[3] + self.biases.dims()[0])
    }
}