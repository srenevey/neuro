use arrayfire::*;
use crate::tensor::*;
use crate::tensor::PrimitiveType;

/// Activation functions
///
#[derive(Debug)]
pub enum Activation {
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
}

impl Activation {
    pub fn eval(&self, z: &Tensor) -> Tensor {
        match self {
            Activation::Sigmoid => sigmoid(z),
            Activation::Softmax => {
                // Input value is shifted for numerical stability
                let z_shifted = sub(z, &max(z, 0), true);
                div(&exp(&z_shifted), &sum(&exp(&z_shifted), 0), true)
            },
            Activation::Tanh => tanh(z),
            Activation::ReLU => {
                maxof(&Tensor::zeros(z.dims()), z, true)
            },
            Activation::LeakyReLU => {
                maxof(&Tensor::zeros(z.dims()), &mul(&constant(0.01 as PrimitiveType, z.dims()), z, true), true)
            },
            Activation::Linear => { z.copy() }
        }
    }

    pub(crate) fn grad(&self, z: &Tensor) -> Tensor {
        match self {
            Activation::Sigmoid => sigmoid(z) * (Tensor::ones(z.dims()) - sigmoid(z)),
            Activation::Softmax => Tensor::ones(z.dims()), // TODO: implement
            Activation::Tanh => Tensor::ones(z.dims()) - mul(&tanh(z), &tanh(z), true),
            Activation::ReLU => {
                let cond = ge(z, &(0 as PrimitiveType), true);
                selectr(&Tensor::ones(z.dims()), &cond, 0.0)
            },
            Activation::LeakyReLU => {
                let cond = ge(z, &(0 as PrimitiveType), true);
                selectr(&Tensor::ones(z.dims()), &cond, 0.01)
            },
            Activation::Linear => Tensor::ones(z.dims()),
        }
    }

    pub(crate) fn id(&self) -> u64 {
        match self {
            Activation::Sigmoid     => 0,
            Activation::Softmax     => 1,
            Activation::Tanh        => 2,
            Activation::ReLU        => 3,
            Activation::LeakyReLU   => 4,
            Activation::Linear      => 5,
        }
    }
}



#[cfg(test)]
mod tests {
    use crate::activations::Activation;
    use crate::assert_approx_eq;
    use arrayfire::*;
    use crate::tensor::*;

    #[test]
    fn sigmoid_eval() {
        let activation = Activation::Sigmoid;
        let values: [PrimitiveType; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [0.000034, 0.231475, 0.310026, 0.475021, 0.5, 0.537430, 0.750260, 0.890903, 0.999945];
        assert_approx_eq!(output, expected_output);
    }


    #[test]
    fn sigmoid_grad() {
        let activation = Activation::Sigmoid;
        let values: [PrimitiveType; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [0.000034, 0.177894, 0.213910, 0.249376, 0.25, 0.248599, 0.187370, 0.097195, 0.000055];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn tanh_eval() {
        let activation = Activation::Tanh;
        let values: [PrimitiveType; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [-1.0, -0.833655, -0.664037, -0.099668, 0.0, 0.148885, 0.800499, 0.970452, 1.0];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn tanh_grad() {
        let activation = Activation::Tanh;
        let values: [PrimitiveType; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [0.000000, 0.305020, 0.559055, 0.990066, 1.0, 0.977833, 0.359201, 0.058223, 0.000000];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn linear_eval() {
        let activation = Activation::Linear;
        let values: [PrimitiveType; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        assert_approx_eq!(output, values);
    }

    #[test]
    fn linear_grad() {
        let activation = Activation::Linear;
        let values: [PrimitiveType; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [1.0; 9];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn relu_eval() {
        let activation = Activation::ReLU;
        let values: [PrimitiveType; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [10.3, 0.0, 0.8, 0.1, 0.0, 0.0, 1.1, 0.0, 0.0];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn relu_grad() {
        let activation = Activation::ReLU;
        let values: [PrimitiveType; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn leaky_relu_eval() {
        let activation = Activation::LeakyReLU;
        let values: [PrimitiveType; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [0.103, 0.0, 0.008, 0.001, 0.0, 0.0, 0.011, 0.0, 0.0];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn leaky_relu_grad() {
        let activation = Activation::LeakyReLU;
        let values: [PrimitiveType; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [1.0, 0.01, 1.0, 1.0, 1.0, 0.01, 1.0, 0.01, 0.01];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn softmax_eval() {
        let activation = Activation::Softmax;
        let values: [PrimitiveType; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let input = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&input);
        let mut output: [PrimitiveType; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [PrimitiveType; 9] = [0.999915025297827, 0.000010129232797, 0.000074845469376, 0.372628471150606, 0.337168183722601, 0.290203345126792, 0.960817236817529, 0.039165028193086, 0.000017734989384];
        assert_approx_eq!(output, expected_output);
    }
}
