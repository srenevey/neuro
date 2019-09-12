use arrayfire::*;
use crate::activations::*;
use super::*;


pub struct Conv2D {
    activation: Activation,
    mode: ConvMode,
    filter: Array<f64>,
    biases: Array<f64>,
}

impl Conv2D {
    pub fn new(activation: Activation, filter: u64, mode: ConvMode) -> Conv2D {
        Conv2D {
            activation,
            mode,
            filter: Array::new_empty(Dim4::new(&[filter, filter, 1, 1])),
            biases: Array::new_empty(Dim4::new(&[0, 0, 1, 1])),
        }
    }
}

impl Layer for Conv2D {
    fn initialize_parameters(&mut self, fan_in: u64) {
        // Change number of channels of filter
        unimplemented!()
    }

    fn compute_activation(&self, prev_activation: &Array<f64>) -> Array<f64> {
        unimplemented!()
    }

    fn compute_activation_mut(&mut self, prev_activation: &Array<f64>) -> Array<f64> {
        self.a_prev = Some(prev_activation.clone());
        self.z = Some(add(&convolve2(prev_activation, &self.filter, self.mode, ConvDomain::SPATIAL), &self.biases, true));

        match &self.z {
            Some(z) => self.activation.eval(z),
            None => panic!("The linear activations z have not been computed!")
        }
    }

    fn fan_out(&self) -> u64 {
        unimplemented!()
    }

    fn compute_da_prev_mut(&mut self, dz: &Array<f64>) -> Array<f64> {
        unimplemented!()
    }

    fn parameters(&self) -> Vec<&Array<f64>> {
        unimplemented!()
    }

    fn dparameters(&self) -> Vec<&Array<f64>> {
        unimplemented!()
    }

    fn set_parameters(&mut self, parameters: Vec<Array<f64>>) {
        unimplemented!()
    }
}