//! Regularization methods.
use crate::tensor::*;

use arrayfire::*;

/// Defines the regularization methods.
#[derive(Debug, Copy, Clone)]
pub enum Regularizer {
    /// Absolute value regularization.
    L1(PrimitiveType),

    /// Squared norm regularization.
    ///
    /// # Example
    /// ```
    /// let regularizer = Regularizer::L2(0.01);
    /// ```
    L2(PrimitiveType),
}

impl Regularizer
{
    pub(crate) fn eval(&self, weights: Vec<&Tensor>) -> PrimitiveType {
        let batch_size = weights[0].dims().get()[0] as PrimitiveType;
        match &self {
            Regularizer::L1(lambda) => {
                //let mut total_sum = Tensor::new(&[0.], Dim4::new(&[1, 1, 1, 1]));
                let mut total_sum = 0.;
                for weight in weights {
                    //total_sum += sum(&sum(&sum(&sum(&abs(weight), 3), 2), 1), 0);
                    total_sum += sum_all(&abs(weight)).0 as PrimitiveType;
                }
                total_sum * (*lambda) / batch_size
            },
            Regularizer::L2(lambda) => {
                // let mut total_sum = Tensor::new(&[0.], Dim4::new(&[1, 1, 1, 1]));
                let mut total_sum = 0.;
                for weight in weights {
                    let prod = matmul(weight, weight, MatProp::TRANS, MatProp::NONE);
                    //total_sum += sum(&sum(&sum(&sum(&prod, 3), 2), 1), 0);
                    total_sum += sum_all(&prod).0 as PrimitiveType;
                }
                total_sum * (*lambda) / (2.0 * batch_size)
            },
        }

    }

    pub(crate) fn grad(&self, weights: &Tensor) -> Tensor {
        let batch_size = weights.dims().get()[0] as PrimitiveType;
        match &self {
            Regularizer::L1(lambda) => {
                mul(&(*lambda / batch_size), &sign(weights), true)
            },
            Regularizer::L2(lambda) => {
                mul(&(*lambda / batch_size), weights, true)
            },
        }
    }
}