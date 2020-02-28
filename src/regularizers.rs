//! Regularization methods.
use arrayfire::*;
use std::str::FromStr;
use std::fmt;

use crate::tensor::*;

/// Defines the regularization methods.
#[derive(Debug, Copy, Clone)]
pub enum Regularizer {
    /// Absolute value regularization.
    L1(PrimitiveType),

    /// Squared norm regularization.
    ///
    /// # Example
    /// ```
    /// # use neuro::regularizers::Regularizer;
    /// let regularizer = Regularizer::L2(0.01);
    /// ```
    L2(PrimitiveType),
}


#[derive(hdf5::H5Type, Clone, Debug)]
#[repr(C)]
pub(crate) struct H5Regularizer {
    name: hdf5::types::VarLenUnicode,
    lambda: PrimitiveType,
}

impl From<&H5Regularizer> for Regularizer {
    fn from(h5_reg: &H5Regularizer) -> Self {
        match h5_reg.name.as_str() {
            "L1" => Regularizer::L1(h5_reg.lambda),
            "L2" => Regularizer::L2(h5_reg.lambda),
            _ => panic!("Unrecognized regularizer"),
        }
    }
}

impl Regularizer
{
    pub(crate) fn eval(self, weights: Vec<&Tensor>) -> PrimitiveType {
        let batch_size = weights[0].dims().get()[0] as PrimitiveType;
        match &self {
            Regularizer::L1(lambda) => {
                let mut total_sum = 0.;
                for weight in weights {
                    total_sum += sum_all(&abs(weight)).0 as PrimitiveType;
                }
                total_sum * (*lambda) / batch_size
            },
            Regularizer::L2(lambda) => {
                let mut total_sum = 0.;
                for weight in weights {
                    let prod = matmul(weight, weight, MatProp::TRANS, MatProp::NONE);
                    total_sum += sum_all(&prod).0 as PrimitiveType;
                }
                total_sum * (*lambda) / (2.0 * batch_size)
            },
        }

    }

    pub(crate) fn grad(self, weights: &Tensor) -> Tensor {
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

    pub(crate) fn save(self, group: &hdf5::Group) -> hdf5::Result<()> {
        match &self {
            Regularizer::L1(lambda) => {
                let regularizer = group.new_dataset::<H5Regularizer>().create("regularizer", 1)?;
                regularizer.write(&[H5Regularizer { name: hdf5::types::VarLenUnicode::from_str("L1").unwrap() , lambda: *lambda }])?;
            },
            Regularizer::L2(lambda) => {
                let regularizer = group.new_dataset::<H5Regularizer>().create("regularizer", 1)?;
                regularizer.write(&[H5Regularizer { name: hdf5::types::VarLenUnicode::from_str("L2").unwrap() , lambda: *lambda }])?;
            }
        }
        Ok(())
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Option<Regularizer> {
        if let Ok(reg) = group.dataset("regularizer") {
            let h5_regularizer = reg.read_raw::<H5Regularizer>().unwrap();
            Some(Regularizer::from(&h5_regularizer[0]))
        } else { None }
    }
}

impl fmt::Display for Regularizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Regularizer::L1(_) => write!(f, "L1"),
            Regularizer::L2(_) => write!(f, "L2"),
        }
    }
}