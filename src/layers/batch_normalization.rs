use arrayfire::*;
use super::*;

pub struct BatchNormalization {
    mb_mean: Array<f64>,
    mb_variance: Array<f64>,
    mean: Array<f64>,
    variance: Array<f64>,
    normalized_input: Array<f64>,
    gamma: Array<f64>,
    dgamma: Array<f64>,
    beta: Array<f64>,
    dbeta: Array<f64>,
    momentum: f64,
    eps: f64,
}

impl BatchNormalization {
    pub fn new() -> Box<BatchNormalization> {
        Box::new(BatchNormalization {
            mb_mean: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            mb_variance: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            mean: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            variance: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            normalized_input: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            gamma: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dgamma: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            beta: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dbeta: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            momentum: 0.99,
            eps: 1e-5f64,
        })
    }

    pub fn with_param(momentum: f64, eps: f64) -> Box<BatchNormalization> {
        Box::new(BatchNormalization {
            mb_mean: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            mb_variance: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            mean: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            variance: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            normalized_input: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            gamma: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dgamma: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            beta: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dbeta: Array::<f64>::new_empty(Dim4::new(&[0, 0, 0, 0])),
            momentum,
            eps,
        })
    }
}

impl Layer for BatchNormalization {
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        self.gamma = constant(1.0f64, input_shape);
        self.beta = constant(0.0f64, input_shape);
        self.mean = constant(0.0f64, input_shape);
        self.variance = constant(0.0f64, input_shape);
    }

    fn compute_activation(&self, prev_activation: &Array<f64>) -> Array<f64> {
        add(&mul(&self.gamma, &div(&sub(prev_activation, &self.mean, true), &sqrt(&sub(&self.variance, &self.eps, true)), true), true), &self.beta, true)
    }

    fn compute_activation_mut(&mut self, prev_activation: &Array<f64>) -> Array<f64> {
        // Compute mini-batch mean and variance
        self.mb_mean = mean(prev_activation, 3);
        self.mb_variance = var(prev_activation, false, 3);
        self.mb_mean.eval();
        self.mb_variance.eval();

        // Update the training set mean and variance using running averages
        self.mean = mul(&self.momentum, &self.mean, false) + &self.mb_mean * (1.0 - self.momentum);
        self.variance = mul(&self.momentum, &self.variance, false) + &self.mb_variance * (1.0 - self.momentum);
        self.mean.eval();
        self.variance.eval();

        // Cache the normalized input for backprop
        self.normalized_input = div(&sub(prev_activation, &self.mb_mean, true), &sqrt(&add(&self.mb_variance, &self.eps, true)), true);
        self.normalized_input.eval();

        add(&mul(&self.gamma, &self.normalized_input, true), &self.beta, true)
    }

    fn output_shape(&self) -> Dim4 {
        self.gamma.dims()
    }


    fn compute_dactivation_mut(&mut self, dz: &Array<f64>) -> Array<f64> {
        self.dgamma = sum(&mul(dz, &self.normalized_input, true), 3);
        self.dbeta = sum(&dz.copy(), 3);

        // Compute the derivative of the loss wrt the variance
        // c1 corresponds to: input - mb_mean
        let c1 = mul(&self.normalized_input, &sqrt(&add(&self.mb_variance, &self.eps, true)), true);
        // c2 corresponds to: sqrt(variance - eps)
        let c2 = sqrt(&add(&self.mb_variance, &self.eps, true));
        let fac = mul(&div(&self.gamma, &(-2.0f64), true), &pow(&c2, &(-3.0f64), true), true);
        let dmb_variance = mul(&sum(&mul(dz, &c1, true), 3), &fac, true);

        // Compute the derivative of the loss wrt the mean
        let term1 = mul(&sum(&dz, 3), &sub(&0.0f64, &div(&self.gamma, &c2, true), true), true);
        let term2 = mul(&dmb_variance, &mean(&mul(&(-2.0f64), &c1, true), 3), true);
        let dmb_mean = add(&term1, &term2, true);

        // Compute the derivative of the loss wrt the normalized input
        let dnormalized_input = mul(dz, &self.gamma, true);

        // Compute and return the derivative of the loss wrt the input
        let term1 = mul(&dnormalized_input, &div(&1.0f64, &sqrt(&self.variance), true), true);
        let m = self.normalized_input.dims()[3] as f64;
        let term2 = mul(&dmb_variance, &mul(&(2.0f64 / m), &c1, true), true);
        let term3 = div(&dmb_mean, &m, true);
        let final_term = add(&term1, &add(&term2, &term3, true), true);
        //af_print!("final", final_term);
        final_term
    }

    fn parameters(&self) -> Vec<&Array<f64>> {
        vec![&self.gamma, &self.beta]
    }

    fn dparameters(&self) -> Vec<&Array<f64>> {
        vec![&self.dgamma, &self.dbeta]
    }

    fn set_parameters(&mut self, parameters: Vec<Array<f64>>) {
        self.gamma = parameters[0].copy();
        self.beta = parameters[1].copy();
    }
}