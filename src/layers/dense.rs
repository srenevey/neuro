use arrayfire::*;
use crate::activations::*;
use super::*;
use super::initializers::Initializer;

pub struct Dense
{
    units: u64,
    activation: Activation,
    weights: Array<f64>,
    dweights: Array<f64>,
    biases: Array<f64>,
    dbiases: Array<f64>,
    z: Option<Array<f64>>,
    a_prev: Option<Array<f64>>,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
}


impl Dense
{
    pub fn new(units: u64, activation: Activation) -> Box<Dense> {
        Box::new(Dense {
            units,
            activation,
            weights: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dweights: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            biases: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dbiases: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            z: None,
            a_prev: None,
            weights_initializer: Initializer::HeUniform,
            biases_initializer: Initializer::Zeros,
        })
    }

    pub fn with_param(units: u64, activation: Activation, weights_initializer: Initializer, biases_initializer: Initializer) -> Box<Dense> {
        Box::new(Dense {
            units,
            activation,
            weights: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dweights: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            biases: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dbiases: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            z: None,
            a_prev: None,
            weights_initializer,
            biases_initializer,
        })
    }


}

impl Layer for Dense
{
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let fan_in = (input_shape.get()[0] * input_shape.get()[1]) as f64;
        let fan_out = self.units as f64;
        self.weights = self.weights_initializer.new(Dim4::new(&[self.units, input_shape.get()[0], 1, 1]), fan_in, fan_out);
        self.biases = self.biases_initializer.new(Dim4::new(&[self.units, 1, 1, 1]), fan_in, fan_out);

        /*
        match self.weights_initializer {
            Initializer::Zeros  => { self.weights = constant(0.0f64, Dim4::new(&[self.units, fan_in.dims()[0], 1, 1])); },
            Initializer::Ones   => { self.weights = constant(1.0f64, Dim4::new(&[self.units, fan_in, 1, 1])); },
            Initializer::Constant(x) => { self.weights = constant(x, Dim4::new(&[self.units, fan_in, 1, 1])); },
            Initializer::RandomUniform  => { self.weights = randu::<f64>(Dim4::new(&[self.units, fan_in, 1, 1])); },
            Initializer::RandomNormal   => { self.weights = mul(&0.01, &randn::<f64>(Dim4::new(&[self.units, fan_in, 1, 1])), false); },
            Initializer::XavierNormal => { self.weights = mul(&(2. / (self.units + fan_in) as f64).sqrt(), &randn::<f64>(Dim4::new(&[self.units, fan_in, 1, 1])), false); },
            Initializer::XavierUniform => {
                let lim = (6. / (self.units + fan_in) as f64).sqrt();
                self.weights = constant(-lim, Dim4::new(&[self.units, fan_in, 1, 1])) + constant(2. * lim, Dim4::new(&[self.units, fan_in, 1, 1])) * randu::<f64>(Dim4::new(&[self.units, fan_in, 1, 1]));
            },
            Initializer::HeNormal     => { self.weights = mul(&(2. / fan_in as f64).sqrt(), &randn::<f64>(Dim4::new(&[self.units, fan_in, 1, 1])), false); }
            Initializer::HeUniform     => {
                let lim = (6. / fan_in as f64).sqrt();
                self.weights = constant(-lim, Dim4::new(&[self.units, fan_in, 1, 1])) + constant(2. * lim, Dim4::new(&[self.units, fan_in, 1, 1])) * randu::<f64>(Dim4::new(&[self.units, fan_in, 1, 1]));
            }
        }

        match self.biases_initializer {
            Initializer::Zeros  => { self.biases = constant(0.0f64, Dim4::new(&[self.units, 1, 1, 1])); },
            Initializer::Ones   => { self.biases = constant(1.0f64, Dim4::new(&[self.units, 1, 1, 1]));},
            Initializer::Constant(x) => { self.biases = constant(x, Dim4::new(&[self.units, 1, 1, 1]));},
            Initializer::RandomUniform  => { self.biases = randu::<f64>(Dim4::new(&[self.units, 1, 1, 1])); },
            Initializer::RandomNormal   => { self.biases = randn::<f64>(Dim4::new(&[self.units, 1, 1, 1])); },
            Initializer::XavierNormal => {},
            Initializer::XavierUniform => {},
            Initializer::HeNormal     => {},
            Initializer::HeUniform => {}
        }
        */
    }

    fn compute_activation(&self, prev_activation: &Array<f64>) -> Array<f64> {
        self.activation.eval(&add(&matmul(&self.weights, prev_activation, MatProp::NONE, MatProp::NONE), &self.biases, true))
    }

    fn compute_activation_mut(&mut self, prev_activation: &Array<f64>) -> Array<f64> {
        self.a_prev = Some(prev_activation.clone());
        self.z = Some(add(&matmul(&self.weights, prev_activation, MatProp::NONE, MatProp::NONE), &self.biases, true));

        match &self.z {
            Some(z) => self.activation.eval(z),
            None => panic!("The linear activations z have not been computed!")
        }
    }


    fn output_shape(&self) -> Dim4 {
        Dim4::new(&[self.units, 1, 1, 1])
    }


    fn compute_dactivation_mut(&mut self, da: &Array<f64>) -> Array<f64> {
        match &self.z {
            Some(z) => {
                let dz = mul(da, &self.activation.grad(z), true);
                match &mut self.a_prev {
                    Some(a_prev) => {
                        let num_samples = a_prev.dims().get()[3];
                        self.dweights = sum(&matmul(&dz, a_prev, MatProp::NONE, MatProp::TRANS), 3) / num_samples as f64;
                        self.dbiases = sum(&dz, 3) / num_samples as f64;
                    },
                    None => panic!("The previous activations have not been computed!"),
                }
                matmul(&self.weights, &dz, MatProp::TRANS, MatProp::NONE)
            },
            None => panic!("The linear activations z have not been computed!"),
        }
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