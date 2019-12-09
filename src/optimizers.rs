//! Optimizers used to train the neural network.
use crate::layers::Layer;
use crate::tensor::*;

use std::fs;
use std::io;
use std::io::{BufWriter, Write};

use arrayfire::*;
use byteorder::{BigEndian, WriteBytesExt};

/// Defines the trait that needs to be implemented by any optimizer working with neuro.
pub trait Optimizer
{
    fn update_parameters(&mut self, layer: &mut dyn Layer, layer_idx: usize);
    fn update_time_step(&mut self) {}
    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>);
    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()>;
    /*
    fn write_vec_array(vec: &Vec<Tensor>, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        for tensor in vec.iter() {
            let dims: [[u8; 8]; 4] = [tensor.dims().get()[0].to_be_bytes(), tensor.dims().get()[1].to_be_bytes(), tensor.dims().get()[2].to_be_bytes(), tensor.dims().get()[3].to_be_bytes()];
            let flat: Vec<u8> = dims.concat();
            writer.write(flat.as_slice())?;
            writer.write(b"\n")?;

            let num_params = (tensor.dims().get()[0] * tensor.dims().get()[1] * tensor.dims().get()[2]) as usize;
            let mut buf: Vec<f64> = vec![0.; num_params];
            tensor.host(buf.as_mut_slice());
            for value in buf.iter() {
                writer.write_f64::<BigEndian>(*value)?;
            }
            writer.write(b"\n")?;

        }
        Ok(())
    }
    */
}

/// Stochastic Gradient Descent
pub struct SGD {
    learning_rate: PrimitiveType,
    momentum: PrimitiveType,
    vdw: Vec<Tensor>,
    vdb: Vec<Tensor>,
}

impl SGD {
    /// Creates a Stochastic Gradient Descent optimizer.
    ///
    /// # Arguments
    /// * `learning_rate`: learning rate used to update the parameters of the layers
    ///
    pub fn new(learning_rate: PrimitiveType) -> SGD {
        SGD {
            learning_rate,
            momentum: 0.0,
            vdw: Vec::new(),
            vdb: Vec::new(),
        }
    }

    /// Creates a Stochastic Gradient Descent optimizer with momentum estimation.
    ///
    /// # Arguments
    /// * `learning_rate`: learning rate used to update the parameters of the layers
    /// * `momentum`: momentum coefficient
    ///
    pub fn with_param(learning_rate: PrimitiveType, momentum: PrimitiveType) -> SGD {
        SGD {
            learning_rate,
            momentum,
            vdw: Vec::new(),
            vdb: Vec::new(),
        }
    }
}

impl Optimizer for SGD
{
    fn update_parameters(&mut self,
                         layer: &mut dyn Layer,
                         layer_idx: usize
    ) {
        if let Some((mut param, dparam)) = layer.parameters_mut() {
            self.vdw[layer_idx] = &self.vdw[layer_idx] * self.momentum + dparam[0] * (1. - self.momentum);
            self.vdb[layer_idx] = &self.vdb[layer_idx] * self.momentum + dparam[1] * (1. - self.momentum);

            self.vdw[layer_idx].eval();
            self.vdb[layer_idx].eval();

            *param[0] -= &self.vdw[layer_idx] * self.learning_rate;
            *param[1] -= &self.vdb[layer_idx] * self.learning_rate;
        }
    }

    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>) {
        for dim in layers_dims {
            self.vdw.push(Tensor::zeros(dim.0));
            self.vdb.push(Tensor::zeros(dim.1));
        }
    }

    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        /*
        writer.write_u64::<BigEndian>(0)?;
        writer.write(b"\n")?;
        writer.write_f64::<BigEndian>(self.learning_rate)?;
        writer.write(b"\n")?;
        writer.write_f64::<BigEndian>(self.momentum)?;

        Self::write_vec_array(&self.vdw, writer)?;
        Self::write_vec_array(&self.vdb, writer)?;
        */

        Ok(())
    }
}


/// Adaptive moments estimation
pub struct Adam {
    learning_rate: PrimitiveType,
    beta1: PrimitiveType,
    beta2: PrimitiveType,
    eps: PrimitiveType,
    time_step: i32,
    vdw: Vec<Tensor>,
    vdb: Vec<Tensor>,
    sdw: Vec<Tensor>,
    sdb: Vec<Tensor>,
}

impl Adam {
    /// Creates an Adam optimizer.
    ///
    /// The exponential decay rates for the first and second moment estimates are set to 0.9 and 0.999 respectively.
    /// The epsilon value used for numerical stability is 1e-8.
    ///
    /// # Arguments
    /// * `learning_rate`: learning rate used to update the parameters of the layers
    ///
    pub fn new(learning_rate: PrimitiveType) -> Adam {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            time_step: 0,
            vdw: Vec::new(),
            vdb: Vec::new(),
            sdw: Vec::new(),
            sdb: Vec::new(),
        }
    }

    /// Creates an Adam optimizer with the given parameters.
    ///
    /// # Arguments
    /// * `learning_rate`: learning rate used to update the parameters of the layers
    /// * `beta1`: exponential decay rate for the first moment estimate
    /// * `beta2`: exponential decay rate for the second moment estimate
    /// * `eps`: small constant used for numerical stability
    ///
    pub fn with_param(learning_rate: PrimitiveType,
                      beta1: PrimitiveType,
                      beta2: PrimitiveType,
                      eps: PrimitiveType
    ) -> Adam {
        Adam {
            learning_rate,
            beta1,
            beta2,
            eps,
            time_step: 0,
            vdw: Vec::new(),
            vdb: Vec::new(),
            sdw: Vec::new(),
            sdb: Vec::new(),
        }
    }
}

impl Optimizer for Adam
{
    fn update_parameters(&mut self,
                         layer: &mut dyn Layer,
                         layer_idx: usize
    ) {
        if let Some((mut param, dparam)) = layer.parameters_mut() {
            // Update biased first moment estimate
            self.vdw[layer_idx] = &self.vdw[layer_idx] * self.beta1 + dparam[0] * (1. - self.beta1);
            self.vdb[layer_idx] = &self.vdb[layer_idx] * self.beta1 + dparam[1] * (1. - self.beta1);

            // Update biased second moment estimate
            self.sdw[layer_idx] = &self.sdw[layer_idx] * self.beta2 + &(dparam[0] * dparam[0]) * (1. - self.beta2);
            self.sdb[layer_idx] = &self.sdb[layer_idx] * self.beta2 + &(dparam[1] * dparam[1]) * (1. - self.beta2);

            self.vdw[layer_idx].eval();
            self.vdb[layer_idx].eval();
            self.sdw[layer_idx].eval();
            self.sdb[layer_idx].eval();

            // Correct both estimates
            let vdw_corr = &self.vdw[layer_idx] / (1. - self.beta1.powi(self.time_step));
            let vdb_corr = &self.vdb[layer_idx] / (1. - self.beta1.powi(self.time_step));
            let sdw_corr = &self.sdw[layer_idx] / (1. - self.beta2.powi(self.time_step));
            let sdb_corr = &self.sdb[layer_idx] / (1. - self.beta2.powi(self.time_step));

            // Update the layer's parameters
            *param[0] -= &vdw_corr / (&sqrt(&sdw_corr) + self.eps) * self.learning_rate;
            *param[1] -= &vdb_corr / (&sqrt(&sdb_corr) + self.eps) * self.learning_rate;
        }
    }

    fn update_time_step(&mut self) {
        self.time_step += 1;
    }

    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>) {
        for dim in layers_dims {
            self.vdw.push(Tensor::zeros(dim.0));
            self.vdb.push(Tensor::zeros(dim.1));
            self.sdw.push(Tensor::zeros(dim.0));
            self.sdb.push(Tensor::zeros(dim.1));
        }
    }

    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        /*
        writer.write_u64::<BigEndian>(1)?;
        writer.write(b"\n")?;
        writer.write_f64::<BigEndian>(self.learning_rate)?;
        writer.write(b"\n")?;
        writer.write_f64::<BigEndian>(self.beta1)?;
        writer.write(b"\n");
        writer.write_f64::<BigEndian>(self.beta2)?;
        writer.write(b"\n");
        writer.write_f64::<BigEndian>(self.eps)?;
        writer.write(b"\n");
        writer.write_u64::<BigEndian>(self.iteration)?;
        writer.write(b"\n");

        Self::write_vec_array(&self.vdw, writer)?;
        Self::write_vec_array(&self.vdb, writer)?;
        Self::write_vec_array(&self.sdw, writer)?;
        Self::write_vec_array(&self.sdb, writer)?;
*/
        Ok(())
    }
}


/// RMSProp
pub struct RMSProp {
    learning_rate: PrimitiveType,
    decay_rate: PrimitiveType,
    eps: PrimitiveType,
    sdw: Vec<Tensor>,
    sdb: Vec<Tensor>,
}

impl RMSProp {

    /// Creates an RMSProp optimizer.
    ///
    /// The exponential decay rate for the first moment estimate is set to 0.9 and the epsilon value used for
    /// numerical stability to 1e-8.
    ///
    /// # Arguments
    /// * `learning_rate`: learning rate used to update the parameters of the layers
    ///
    pub fn new(learning_rate: PrimitiveType) -> RMSProp {
        RMSProp {
            learning_rate,
            decay_rate: 0.9,
            eps: 1e-8,
            sdw: Vec::new(),
            sdb: Vec::new(),
        }
    }

    /// Creates an RMSProp optimizer with the given parameters.
    ///
    /// # Arguments
    /// * `learning_rate`: learning rate used to update the parameters of the layers
    /// * `decay_rate`: exponential decay rate of the moving average
    /// * `eps`: small constant used for numerical stability
    ///
    pub fn with_param(learning_rate: PrimitiveType,
                      decay_rate: PrimitiveType,
                      eps: PrimitiveType
    ) -> RMSProp {
        RMSProp {
            learning_rate,
            decay_rate,
            eps,
            sdw: Vec::new(),
            sdb: Vec::new(),
        }
    }
}

impl Optimizer for RMSProp
{
    fn update_parameters(&mut self,
                         layer: &mut dyn Layer,
                         layer_idx: usize
    ) {
        if let Some((mut param, dparam)) = layer.parameters_mut() {
            self.sdw[layer_idx] = &self.sdw[layer_idx] * self.decay_rate + &(dparam[0] * dparam[0]) * (1. - self.decay_rate);
            self.sdb[layer_idx] = &self.sdb[layer_idx] * self.decay_rate + &(dparam[1] * dparam[1]) * (1. - self.decay_rate);

            self.sdw[layer_idx].eval();
            self.sdb[layer_idx].eval();

            *param[0] -= dparam[0] / (&sqrt(&self.sdw[layer_idx]) + self.eps) * self.learning_rate;
            *param[1] -= dparam[1] / (&sqrt(&self.sdb[layer_idx]) + self.eps) * self.learning_rate;
        }
    }

    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>) {
        for dim in layers_dims {
            self.sdw.push(Tensor::zeros(dim.0));
            self.sdb.push(Tensor::zeros(dim.1));
        }
    }

    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        /*
        writer.write_u64::<BigEndian>(2)?;
        writer.write(b"\n")?;
        writer.write_f64::<BigEndian>(self.learning_rate)?;
        writer.write(b"\n")?;
        writer.write_f64::<BigEndian>(self.beta2)?;
        writer.write(b"\n");
        writer.write_f64::<BigEndian>(self.eps)?;
        writer.write(b"\n");

        Self::write_vec_array(&self.sdw, writer);
        Self::write_vec_array(&self.sdb, writer);
        */
        Ok(())
    }
}

/// AdaDelta
#[derive(Default)]
pub struct AdaDelta {
    decay_rate: PrimitiveType,
    eps: PrimitiveType,
    grad_acc: [Vec<Tensor>; 2],
    updates_acc: [Vec<Tensor>; 2],
}

impl AdaDelta {

    /// Creates an AdaDelta optimizer.
    ///
    /// The exponential decay rate is set to 0.95 and the epsilon value used for numerical stability to 1e-6.
    ///
    pub fn new() -> AdaDelta {
        AdaDelta {
            decay_rate: 0.95,
            eps: 1e-6,
            grad_acc: Default::default(),
            updates_acc: Default::default(),
        }
    }

    /// Creates an AdaDelta optimizer with the given parameters.
    ///
    /// # Arguments
    /// * `decay_rate`: exponential decay rate used for the gradient and update accumulations
    /// * `eps`: small constant used for numerical stability
    ///
    pub fn with_param(decay_rate: PrimitiveType, eps: PrimitiveType) -> AdaDelta {
        AdaDelta {
            decay_rate,
            eps,
            grad_acc: Default::default(),
            updates_acc: Default::default(),
        }
    }
}


impl Optimizer for AdaDelta
{
    fn update_parameters(&mut self,
                         layer: &mut dyn Layer,
                         layer_idx: usize
    ) {
        if let Some((mut param, dparam)) = layer.parameters_mut() {
            for i in 0..param.len() {
                // Accumulate gradients
                self.grad_acc[i][layer_idx] = &self.grad_acc[i][layer_idx] * self.decay_rate + &(dparam[i] * dparam[i]) * (1. - self.decay_rate);
                // Compute update
                let update = - sqrt(&(&self.updates_acc[i][layer_idx] + self.eps)) / sqrt(&(&self.grad_acc[i][layer_idx] * &self.grad_acc[i][layer_idx] + self.eps)) * dparam[i];
                // Accumulate updates
                self.updates_acc[i][layer_idx] = &self.updates_acc[i][layer_idx] * self.decay_rate + &(&update * &update) * (1. - self.decay_rate);

                self.grad_acc[i][layer_idx].eval();
                self.updates_acc[i][layer_idx].eval();

                // Apply update
                *param[i] += update;
            }
        }
    }

    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>) {
        for dim in layers_dims {
            self.grad_acc[0].push(Tensor::zeros(dim.0));
            self.updates_acc[0].push(Tensor::zeros(dim.0));
            self.grad_acc[1].push(Tensor::zeros(dim.1));
            self.updates_acc[1].push(Tensor::zeros(dim.1));
        }
    }

    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        Ok(())
    }
}