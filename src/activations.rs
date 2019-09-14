use arrayfire::*;

pub enum Activation {
    Sigmoid,
    Softmax,
    Tanh,
    ReLU,
    LeakyReLU,
    Linear,
}

impl Activation {
    pub fn eval(&self, z: &Array<f64>) -> Array<f64> {
        match self {
            Activation::Sigmoid     => { sigmoid(z) },
            Activation::Softmax     => { div(&exp(z), &sum(&exp(z), 0), true) },
            Activation::Tanh        => { tanh(z) },
            Activation::ReLU        => { maxof(&constant(0.0f64, z.dims()), z, true) },
            Activation::LeakyReLU   => { maxof(&constant(0.0f64, z.dims()), &mul(&0.01f64, z, true), true) },
            Activation::Linear      => { z.clone() }
        }
    }

    pub fn grad(&self, z: &Array<f64>) -> Array<f64> {
        match self {
            Activation::Sigmoid     => { sigmoid(z) * (constant(1.0f64, z.dims()) - sigmoid(z)) },
            Activation::Softmax     => { constant(0.0f64, z.dims()) }
            Activation::Tanh        => { constant(1.0f64, z.dims()) - tanh(z) * tanh(z) },
            Activation::ReLU        => {
                let cond = ge(z, &0.0f64, true);
                selectr(&constant(1.0f64, z.dims()), &cond, 0.0f64)
            },
            Activation::LeakyReLU   => {
                let cond = ge(z, &0.0f64, true);
                selectr(&constant(1.0f64, z.dims()), &cond, 0.01f64)
            },
            Activation::Linear      => { constant(1.0f64, z.dims()) }
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::activations::Activation;
    use crate::assert_approx_eq;
    use arrayfire::*;

    #[test]
    fn sigmoid_eval() {
        let activation = Activation::Sigmoid;
        let values: [f64; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [0.000034, 0.231475, 0.310026, 0.475021, 0.5, 0.537430, 0.750260, 0.890903, 0.999945];
        assert_approx_eq!(output, expected_output);
    }


    #[test]
    fn sigmoid_grad() {
        let activation = Activation::Sigmoid;
        let values: [f64; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [0.000034, 0.177894, 0.213910, 0.249376, 0.25, 0.248599, 0.187370, 0.097195, 0.000055];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn tanh_eval() {
        let activation = Activation::Tanh;
        let values: [f64; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [-1.0, -0.833655, -0.664037, -0.099668, 0.0, 0.148885, 0.800499, 0.970452, 1.0];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn tanh_grad() {
        let activation = Activation::Tanh;
        let values: [f64; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [0.000000, 0.305020, 0.559055, 0.990066, 1.0, 0.977833, 0.359201, 0.058223, 0.000000];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn linear_eval() {
        let activation = Activation::Linear;
        let values: [f64; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        assert_approx_eq!(output, values);
    }

    #[test]
    fn linear_grad() {
        let activation = Activation::Linear;
        let values: [f64; 9] = [-10.3, -1.2, -0.8, -0.1, 0., 0.15, 1.1, 2.1, 9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [1.0; 9];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn relu_eval() {
        let activation = Activation::ReLU;
        let values: [f64; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [10.3, 0.0, 0.8, 0.1, 0.0, 0.0, 1.1, 0.0, 0.0];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn relu_grad() {
        let activation = Activation::ReLU;
        let values: [f64; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn leaky_relu_eval() {
        let activation = Activation::LeakyReLU;
        let values: [f64; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.eval(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [0.103, 0.0, 0.008, 0.001, 0.0, 0.0, 0.011, 0.0, 0.0];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn leaky_relu_grad() {
        let activation = Activation::LeakyReLU;
        let values: [f64; 9] = [10.3, -1.2, 0.8, 0.1, 0., -0.15, 1.1, -2.1, -9.8];
        let z = Array::new(&values, Dim4::new(&[3, 3, 1, 1]));
        let eval = activation.grad(&z);
        let mut output: [f64; 9] = [0.; 9];
        eval.host(&mut output);
        let expected_output: [f64; 9] = [1.0, 0.01, 1.0, 1.0, 1.0, 0.01, 1.0, 0.01, 0.01];
        assert_approx_eq!(output, expected_output);
    }
}