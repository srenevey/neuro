use arrayfire::*;
use crate::tensor::*;

#[derive(Debug, Copy, Clone)]
pub enum Regularizer {
    L1(PrimitiveType),
    L2(PrimitiveType),
}

impl Regularizer
{
    pub(crate) fn eval(&self, weights: Vec<&Tensor>) -> PrimitiveType {
        let batch_size = weights[0].dims().get()[0] as PrimitiveType;
        match &self {
            Regularizer::L1(lambda) => {
                let mut sum: PrimitiveType = 0.;
                for weight in weights {
                    sum += sum_all(&abs(weight)).0 as PrimitiveType;
                }
                lambda / batch_size * sum
            },
            Regularizer::L2(lambda) => {
                let mut sum: PrimitiveType = 0.;
                for weight in weights {
                    sum += sum_all(&matmul(weight, weight, MatProp::TRANS, MatProp::NONE)).0 as PrimitiveType;
                }
                lambda / (2.0 * batch_size) * sum
            },
        }

    }

    pub(crate) fn grad(&self, weights: &Tensor) -> Tensor {
        let batch_size = weights.dims().get()[0] as PrimitiveType;
        match &self {
            Regularizer::L1(lambda) => {
                mul(&(lambda / batch_size), &sign(weights), false)
            },
            Regularizer::L2(lambda) => {
                mul(&(lambda / batch_size), weights, false)
            },
        }
    }
}