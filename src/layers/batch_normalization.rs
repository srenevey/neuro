use arrayfire::*;
use super::*;
use std::fmt;
use crate::tensor::*;

pub struct BatchNormalization {
    mb_mean: Tensor,
    mb_variance: Tensor,
    mean: Tensor,
    variance: Tensor,
    normalized_input: Tensor,
    gamma: Tensor,
    dgamma: Tensor,
    beta: Tensor,
    dbeta: Tensor,
    momentum: PrimitiveType,
    eps: PrimitiveType,
}

impl BatchNormalization {
    pub fn new() -> Box<BatchNormalization> {
        Box::new(BatchNormalization {
            mb_mean: Tensor::new_empty_tensor(),
            mb_variance: Tensor::new_empty_tensor(),
            mean: Tensor::new_empty_tensor(),
            variance: Tensor::new_empty_tensor(),
            normalized_input: Tensor::new_empty_tensor(),
            gamma: Tensor::new_empty_tensor(),
            dgamma: Tensor::new_empty_tensor(),
            beta: Tensor::new_empty_tensor(),
            dbeta: Tensor::new_empty_tensor(),
            momentum: 0.99,
            eps: 1e-5,
        })
    }

    pub fn with_param(momentum: PrimitiveType, eps: PrimitiveType) -> Box<BatchNormalization> {
        Box::new(BatchNormalization {
            mb_mean: Tensor::new_empty_tensor(),
            mb_variance: Tensor::new_empty_tensor(),
            mean: Tensor::new_empty_tensor(),
            variance: Tensor::new_empty_tensor(),
            normalized_input: Tensor::new_empty_tensor(),
            gamma: Tensor::new_empty_tensor(),
            dgamma: Tensor::new_empty_tensor(),
            beta: Tensor::new_empty_tensor(),
            dbeta: Tensor::new_empty_tensor(),
            momentum,
            eps,
        })
    }
}

impl Layer for BatchNormalization {
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        self.gamma = Tensor::ones(input_shape);
        self.beta = Tensor::zeros(input_shape);
        self.mean = Tensor::zeros(input_shape);
        self.variance = Tensor::ones(input_shape);
    }

    fn compute_activation(&self, input: &Tensor) -> Tensor {
        add(&mul(&self.gamma, &div(&sub(input, &self.mean, true), &sqrt(&sub(&self.variance, &self.eps, true)), true), true), &self.beta, true)
    }

    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor {
        // Compute mini-batch mean and variance
        self.mb_mean = input.reduce(Reduction::MeanBatches);
        self.mb_variance = var(input, false, 3);
        self.mb_mean.eval();
        self.mb_variance.eval();

        // Update the training set mean and variance using running averages
        self.mean = mul(&self.momentum, &self.mean, false) + &self.mb_mean * (1.0 - self.momentum);
        self.variance = mul(&self.momentum, &self.variance, false) + &self.mb_variance * (1.0 - self.momentum);
        self.mean.eval();
        self.variance.eval();

        // Cache the normalized input for backprop
        self.normalized_input = div(&sub(input, &self.mb_mean, true), &sqrt(&add(&self.mb_variance, &self.eps, true)), true);
        self.normalized_input.eval();

        add(&mul(&self.gamma, &self.normalized_input, true), &self.beta, true)
    }

    fn compute_dactivation_mut(&mut self, dz: &Tensor) -> Tensor {
        self.dgamma = sum(&mul(dz, &self.normalized_input, true), 3);
        self.dbeta = sum(&dz.copy(), 3);

        // Compute the derivative of the loss wrt the variance
        // c1 corresponds to: input - mb_mean
        let c1 = mul(&self.normalized_input, &sqrt(&add(&self.mb_variance, &self.eps, true)), true);
        // c2 corresponds to: sqrt(variance - eps)
        let c2 = sqrt(&add(&self.mb_variance, &self.eps, true));
        let fac = mul(&div(&self.gamma, &(-2.0 as PrimitiveType), true), &pow(&c2, &(-3.0 as PrimitiveType), true), true);
        let dmb_variance = mul(&sum(&mul(dz, &c1, true), 3), &fac, true);

        // Compute the derivative of the loss wrt the mean
        let term1 = mul(&sum(&dz, 3), &sub(&(0.0 as PrimitiveType), &div(&self.gamma, &c2, true), true), true);
        let term2 = mul(&dmb_variance, &mean(&mul(&(-2.0 as PrimitiveType), &c1, true), 3), true);
        let dmb_mean = add(&term1, &term2, true);

        // Compute the derivative of the loss wrt the normalized input
        let dnormalized_input = mul(dz, &self.gamma, true);

        // Compute and return the derivative of the loss wrt the input
        let term1 = mul(&dnormalized_input, &div(&(1.0 as PrimitiveType), &sqrt(&self.variance), true), true);
        let m = self.normalized_input.dims()[3] as PrimitiveType;
        let term2 = mul(&dmb_variance, &mul(&(2.0 as PrimitiveType / m), &c1, true), true);
        let term3 = div(&dmb_mean, &m, true);
        let final_term = add(&term1, &add(&term2, &term3, true), true);
        //af_print!("final", final_term);
        final_term
    }

    fn output_shape(&self) -> Dim4 {
        self.gamma.dims()
    }


    fn parameters(&self) -> Option<Vec<&Tensor>> {
        Some(vec![&self.gamma, &self.beta])
    }


    fn parameters_mut(&mut self) -> Option<(Vec<&mut Tensor>, Vec<&Tensor>)> {
        Some((vec![&mut self.gamma, &mut self.beta], vec![&self.dgamma, &self.dbeta]))
    }

    fn dparameters(&self) -> Option<Vec<&Tensor>> {
        Some(vec![&self.dgamma, &self.dbeta])
    }


    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        Ok(())
    }

}

impl fmt::Display for BatchNormalization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "BatchNorm \t 2")
    }
}