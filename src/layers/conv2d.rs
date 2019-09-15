use arrayfire::*;
use crate::activations::*;
use crate::layers::initializers::*;
use super::Layer;

pub enum ConvMode {
    Valid,
    Same,
}


pub struct Conv2D {
    activation: Activation,
    mode: arrayfire::ConvMode,
    kernel_size: (u64, u64),
    filters: u64,
    output_shape: Dim4,
    weights: Array<f64>,
    biases: Array<f64>,
    dweights: Array<f64>,
    dbiases: Array<f64>,
    z: Option<Array<f64>>,
    a_prev: Option<Array<f64>>,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
}

impl Conv2D {
    pub fn new(filters: u64, kernel_size: (u64, u64), conv_mode: ConvMode, activation: Activation) -> Box<Conv2D> {

        let mode = match conv_mode {
            ConvMode::Valid => arrayfire::ConvMode::EXPAND,
            ConvMode::Same => arrayfire::ConvMode::DEFAULT,
        };

        Box::new(Conv2D {
            activation,
            mode,
            kernel_size,
            filters,
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            weights: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            biases: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dweights: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dbiases: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            z: None,
            a_prev: None,
            weights_initializer: Initializer::HeUniform,
            biases_initializer: Initializer::Zeros,
        })
    }

    pub fn with_param(filters: u64, kernel_size: (u64, u64), conv_mode: ConvMode, activation: Activation, weights_initializer: Initializer, biases_initializer: Initializer) -> Box<Conv2D> {
        let mode = match conv_mode {
            ConvMode::Valid => arrayfire::ConvMode::EXPAND,
            ConvMode::Same => arrayfire::ConvMode::DEFAULT,
        };

        Box::new(Conv2D {
            activation,
            mode,
            kernel_size,
            filters,
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            weights: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            biases: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dweights: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dbiases: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            z: None,
            a_prev: None,
            weights_initializer,
            biases_initializer,
        })
    }
}

impl Layer for Conv2D {
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let fan_in = (input_shape.get()[0] * input_shape.get()[1] * input_shape.get()[2]) as f64;
        let fan_out = (input_shape.get()[0] as f64 - self.kernel_size.0 as f64 + 1.) * (input_shape.get()[1] as f64 - self.kernel_size.1 as f64 + 1.) * input_shape.get()[2] as f64;
        self.output_shape = Dim4::new(&[input_shape.get()[0] - self.kernel_size.0 + 1, input_shape.get()[1] - self.kernel_size.1 + 1, input_shape.get()[2], 1]);
        self.weights = self.weights_initializer.new(Dim4::new(&[self.kernel_size.0, self.kernel_size.1, input_shape.get()[2], self.filters]), fan_in, fan_out);
        self.biases = self.biases_initializer.new(Dim4::new(&[1, 1, 1, self.filters]), fan_in, fan_out);
    }

    fn compute_activation(&self, prev_activation: &Array<f64>) -> Array<f64> {
        self.activation.eval(&add(&convolve2(prev_activation, &self.weights, self.mode, ConvDomain::SPATIAL), &self.biases, true))
    }

    fn compute_activation_mut(&mut self, prev_activation: &Array<f64>) -> Array<f64> {
        self.a_prev = Some(prev_activation.clone());
        self.z = Some(add(&convolve2(prev_activation, &self.weights, self.mode, ConvDomain::SPATIAL), &self.biases, true));

        match &self.z {
            Some(z) => self.activation.eval(z),
            None => panic!("The linear activations z have not been computed!")
        }
    }

    fn output_shape(&self) -> Dim4 {
        self.output_shape
    }

    fn compute_dactivation_mut(&mut self, dz: &Array<f64>) -> Array<f64> {
        unimplemented!()
    }

    fn parameters(&self) -> Vec<&Array<f64>> {
        vec![&self.weights, &self.biases]
    }

    fn dparameters(&self) -> Vec<&Array<f64>> {
        vec![&self.dweights, &self.dbiases]
    }

    fn set_parameters(&mut self, parameters: Vec<Array<f64>>) {
        self.weights = parameters[0].copy();
        self.biases = parameters[1].copy();
    }
}