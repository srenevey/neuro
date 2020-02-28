//! Optimizers used to train the neural network.
use arrayfire::*;
use std::str::FromStr;

use crate::errors::Error;
use crate::io::save_vec_tensor;
use crate::layers::Layer;
use crate::tensor::*;


/// Defines the trait that needs to be implemented by any optimizer working with neuro.
pub trait Optimizer
{
    fn name(&self) -> &str;
    fn update_parameters(&mut self, layer: &mut dyn Layer, layer_idx: usize);
    fn update_time_step(&mut self) {}
    fn initialize_parameters(&mut self, layers_dims: Vec<(Dim, Dim)>);
    fn save(&self, file: &hdf5::File) -> Result<(), Error>;
}


/// Stochastic Gradient Descent
pub struct SGD {
    learning_rate: PrimitiveType,
    momentum: PrimitiveType,
    first_moment_est: [Vec<Tensor>; 2],
}

impl SGD {

    pub(crate) const NAME: &'static str = "SGD";

    /// Creates a Stochastic Gradient Descent optimizer.
    pub fn new(learning_rate: PrimitiveType) -> SGD {
        SGD {
            learning_rate,
            momentum: 0.0,
            first_moment_est: Default::default(),
        }
    }

    /// Creates a Stochastic Gradient Descent optimizer with momentum estimation.
    pub fn with_param(learning_rate: PrimitiveType, momentum: PrimitiveType) -> SGD {
        SGD {
            learning_rate,
            momentum,
            first_moment_est: Default::default(),
        }
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<SGD> {
        let learning_rate = group.dataset("learning_rate").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the learning rate.");
        let momentum = group.dataset("momentum").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the momentum.");
        let first_moment_est_0 = group.dataset("first_moment_est_0").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve first_moment_est_0.");
        let first_moment_est_1 = group.dataset("first_moment_est_1").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve first_moment_est_1.");

        Box::new(SGD {
            learning_rate: learning_rate[0],
            momentum: momentum[0],
            first_moment_est: [first_moment_est_0.iter().map(Tensor::from).collect::<Vec<Tensor>>(), first_moment_est_1.iter().map(Tensor::from).collect::<Vec<Tensor>>()],
        })
    }
}


impl Optimizer for SGD
{
    fn name(&self) -> &str {
        Self::NAME
    }

    fn update_parameters(&mut self,
                         layer: &mut dyn Layer,
                         layer_idx: usize
    ) {
        if let Some((mut param, dparam)) = layer.parameters_mut() {
            for i in 0..param.len() {
                self.first_moment_est[i][layer_idx] = &self.first_moment_est[i][layer_idx] * self.momentum + dparam[i] * (1. - self.momentum);
                self.first_moment_est[i][layer_idx].eval();
                *param[i] -= &self.first_moment_est[i][layer_idx] * self.learning_rate;
            }
        }
    }

    fn initialize_parameters(&mut self, layers_dims: Vec<(Dim, Dim)>) {
        for dim in layers_dims {
            self.first_moment_est[0].push(Tensor::zeros(dim.0));
            self.first_moment_est[1].push(Tensor::zeros(dim.1));
        }
    }

    fn save(&self, file: &hdf5::File) -> Result<(), Error> {

        let optimizer = file.create_group("optimizer")?;

        let opt_type = optimizer.new_dataset::<hdf5::types::VarLenUnicode>().create("type", 1)?;
        opt_type.write(&[hdf5::types::VarLenUnicode::from_str(Self::NAME).unwrap()])?;

        let learning_rate = optimizer.new_dataset::<PrimitiveType>().create("learning_rate", 1)?;
        learning_rate.write(&[self.learning_rate])?;

        let momentum = optimizer.new_dataset::<PrimitiveType>().create("momentum", 1)?;
        momentum.write(&[self.momentum])?;

        save_vec_tensor(&optimizer, &self.first_moment_est[0], "first_moment_est_0")?;
        save_vec_tensor(&optimizer, &self.first_moment_est[1], "first_moment_est_1")?;

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
    first_moment_est: [Vec<Tensor>; 2],
    second_moment_est: [Vec<Tensor>; 2],
}

impl Adam {

    pub(crate) const NAME: &'static str = "Adam";

    /// Creates an Adam optimizer.
    ///
    /// The exponential decay rates for the first and second moment estimates are set to 0.9 and 0.999 respectively.
    /// The epsilon value used for numerical stability is 1e-8.
    ///
    pub fn new(learning_rate: PrimitiveType) -> Adam {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            time_step: 0,
            first_moment_est: Default::default(),
            second_moment_est: Default::default(),
        }
    }

    /// Creates an Adam optimizer with the given parameters.
    ///
    /// # Arguments
    /// * `learning_rate` - learning rate used to update the parameters of the layers.
    /// * `beta1` - exponential decay rate for the first moment estimate.
    /// * `beta2` - exponential decay rate for the second moment estimate.
    /// * `eps` - small constant used for numerical stability.
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
            first_moment_est: Default::default(),
            second_moment_est: Default::default(),
        }
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<Adam> {
        let learning_rate = group.dataset("learning_rate").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the learning rate.");
        let beta1 = group.dataset("beta1").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve beta1.");
        let beta2 = group.dataset("beta2").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve beta2.");
        let eps = group.dataset("eps").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the epsilon value.");
        let time_step = group.dataset("time_step").and_then(|ds| ds.read_raw::<i32>()).expect("Could not retrieve the time step.");
        let first_moment_est_0 = group.dataset("first_moment_est_0").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve first_moment_est_0.");
        let first_moment_est_1 = group.dataset("first_moment_est_1").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve first_moment_est_1.");
        let second_moment_est_0 = group.dataset("second_moment_est_0").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve second_moment_est_0.");
        let second_moment_est_1 = group.dataset("second_moment_est_1").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve second_moment_est_1.");

        Box::new(Adam {
            learning_rate: learning_rate[0],
            beta1: beta1[0],
            beta2: beta2[0],
            eps: eps[0],
            time_step: time_step[0],
            first_moment_est: [first_moment_est_0.iter().map(Tensor::from).collect::<Vec<Tensor>>(), first_moment_est_1.iter().map(Tensor::from).collect::<Vec<Tensor>>()],
            second_moment_est: [second_moment_est_0.iter().map(Tensor::from).collect::<Vec<Tensor>>(), second_moment_est_1.iter().map(Tensor::from).collect::<Vec<Tensor>>()],
        })
    }
}

impl Optimizer for Adam
{
    fn name(&self) -> &str {
        Self::NAME
    }

    fn update_parameters(&mut self,
                         layer: &mut dyn Layer,
                         layer_idx: usize
    ) {
        if let Some((mut param, dparam)) = layer.parameters_mut() {

            for i in 0..param.len() {
                // Update the biased first and second moment estimates
                self.first_moment_est[i][layer_idx] = &self.first_moment_est[i][layer_idx] * self.beta1 + dparam[i] * (1. - self.beta1);
                self.second_moment_est[i][layer_idx] = &self.second_moment_est[i][layer_idx] * self.beta2 + &(dparam[i] * dparam[i]) * (1. - self.beta2);

                self.first_moment_est[i][layer_idx].eval();
                self.second_moment_est[i][layer_idx].eval();

                // Correct both estimates
                let first_moment_est_corr = &self.first_moment_est[i][layer_idx] / (1. - self.beta1.powi(self.time_step));
                let second_moment_est_corr = &self.second_moment_est[i][layer_idx] / (1. - self.beta2.powi(self.time_step));

                // Update the parameter
                *param[i] -= &first_moment_est_corr / (&sqrt(&second_moment_est_corr) + self.eps) * self.learning_rate;
            }
        }
    }

    fn update_time_step(&mut self) {
        self.time_step += 1;
    }

    fn initialize_parameters(&mut self, layers_dims: Vec<(Dim, Dim)>) {

        for dim in layers_dims {
            self.first_moment_est[0].push(Tensor::zeros(dim.0));
            self.second_moment_est[0].push(Tensor::zeros(dim.0));
            self.first_moment_est[1].push(Tensor::zeros(dim.1));
            self.second_moment_est[1].push(Tensor::zeros(dim.1));
        }
    }

    fn save(&self, file: &hdf5::File) -> Result<(), Error> {

        let optimizer = file.create_group("optimizer")?;

        let opt_type = optimizer.new_dataset::<hdf5::types::VarLenUnicode>().create("type", 1)?;
        opt_type.write(&[hdf5::types::VarLenUnicode::from_str(Self::NAME).unwrap()])?;

        let learning_rate = optimizer.new_dataset::<PrimitiveType>().create("learning_rate", 1)?;
        learning_rate.write(&[self.learning_rate])?;

        let beta1 = optimizer.new_dataset::<PrimitiveType>().create("beta1", 1)?;
        beta1.write(&[self.beta1])?;

        let beta2 = optimizer.new_dataset::<PrimitiveType>().create("beta2", 1)?;
        beta2.write(&[self.beta2])?;

        let eps = optimizer.new_dataset::<PrimitiveType>().create("eps", 1)?;
        eps.write(&[self.eps])?;

        let time_step = optimizer.new_dataset::<PrimitiveType>().create("time_step", 1)?;
        time_step.write(&[self.time_step])?;

        save_vec_tensor(&optimizer, &self.first_moment_est[0], "first_moment_est_0")?;
        save_vec_tensor(&optimizer, &self.first_moment_est[1], "first_moment_est_1")?;
        save_vec_tensor(&optimizer, &self.second_moment_est[0], "second_moment_est_0")?;
        save_vec_tensor(&optimizer, &self.second_moment_est[1], "second_moment_est_1")?;

        Ok(())
    }
}


/// RMSProp
pub struct RMSProp {
    learning_rate: PrimitiveType,
    decay_rate: PrimitiveType,
    eps: PrimitiveType,
    first_moment_est: [Vec<Tensor>; 2],
}

impl RMSProp {

    pub(crate) const NAME: &'static str = "RMSProp";

    /// Creates an RMSProp optimizer.
    ///
    /// The exponential decay rate for the first moment estimate is set to 0.9 and the epsilon value used for
    /// numerical stability to 1e-8.
    ///
    pub fn new(learning_rate: PrimitiveType) -> RMSProp {
        RMSProp {
            learning_rate,
            decay_rate: 0.9,
            eps: 1e-8,
            first_moment_est: Default::default(),
        }
    }

    /// Creates an RMSProp optimizer with the given parameters.
    pub fn with_param(learning_rate: PrimitiveType,
                      decay_rate: PrimitiveType,
                      eps: PrimitiveType
    ) -> RMSProp {
        RMSProp {
            learning_rate,
            decay_rate,
            eps,
            first_moment_est: Default::default(),
        }
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<RMSProp> {
        let learning_rate = group.dataset("learning_rate").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the learning rate.");
        let decay_rate = group.dataset("decay_rate").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the decay rate.");
        let eps = group.dataset("eps").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the epsilon value.");
        let first_moment_est_0 = group.dataset("first_moment_est_0").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve first_moment_est_0.");
        let first_moment_est_1 = group.dataset("first_moment_est_1").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve first_moment_est_1.");
        Box::new(RMSProp {
            learning_rate: learning_rate[0],
            decay_rate: decay_rate[0],
            eps: eps[0],
            first_moment_est: [first_moment_est_0.iter().map(Tensor::from).collect::<Vec<Tensor>>(), first_moment_est_1.iter().map(Tensor::from).collect::<Vec<Tensor>>()],
        })
    }
}

impl Optimizer for RMSProp
{
    fn name(&self) -> &str {
        Self::NAME
    }

    fn update_parameters(&mut self,
                         layer: &mut dyn Layer,
                         layer_idx: usize
    ) {
        if let Some((mut param, dparam)) = layer.parameters_mut() {
            for i in 0..param.len() {
                self.first_moment_est[i][layer_idx] = &self.first_moment_est[i][layer_idx] * self.decay_rate + &(dparam[i] * dparam[i]) * (1. - self.decay_rate);
                self.first_moment_est[i][layer_idx].eval();
                *param[i] -= dparam[i] / (&sqrt(&self.first_moment_est[i][layer_idx]) + self.eps) * self.learning_rate;
            }
        }
    }

    fn initialize_parameters(&mut self, layers_dims: Vec<(Dim, Dim)>) {
        for dim in layers_dims {
            self.first_moment_est[0].push(Tensor::zeros(dim.0));
            self.first_moment_est[1].push(Tensor::zeros(dim.1));
        }
    }

    fn save(&self, file: &hdf5::File) -> Result<(), Error> {
        let optimizer = file.create_group("optimizer")?;

        let opt_type = optimizer.new_dataset::<hdf5::types::VarLenUnicode>().create("type", 1)?;
        opt_type.write(&[hdf5::types::VarLenUnicode::from_str(Self::NAME).unwrap()])?;

        let learning_rate = optimizer.new_dataset::<PrimitiveType>().create("learning_rate", 1)?;
        learning_rate.write(&[self.learning_rate])?;

        let decay_rate = optimizer.new_dataset::<PrimitiveType>().create("decay_rate", 1)?;
        decay_rate.write(&[self.decay_rate])?;

        let eps = optimizer.new_dataset::<PrimitiveType>().create("eps", 1)?;
        eps.write(&[self.eps])?;

        save_vec_tensor(&optimizer, &self.first_moment_est[0], "first_moment_est_0")?;
        save_vec_tensor(&optimizer, &self.first_moment_est[1], "first_moment_est_1")?;
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

    pub(crate) const NAME: &'static str = "AdaDelta";

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

    /// Creates an AdaDelta optimizer with the parameters.
    pub fn with_param(decay_rate: PrimitiveType, eps: PrimitiveType) -> AdaDelta {
        AdaDelta {
            decay_rate,
            eps,
            grad_acc: Default::default(),
            updates_acc: Default::default(),
        }
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<AdaDelta> {
        let decay_rate = group.dataset("decay_rate").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the decay rate.");
        let eps = group.dataset("eps").and_then(|ds| ds.read_raw::<PrimitiveType>()).expect("Could not retrieve the epsilon value.");
        let gradacc0 = group.dataset("grad_acc_0").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve grad_acc_0.");
        let gradacc1 = group.dataset("grad_acc_1").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve grad_acc_1.");
        let updatesacc0 = group.dataset("updates_acc_0").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve updates_acc_0.");
        let updatesacc1 = group.dataset("updates_acc_1").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve updates_acc_1.");
        Box::new(AdaDelta {
            decay_rate: decay_rate[0],
            eps: eps[0],
            grad_acc: [gradacc0.iter().map(Tensor::from).collect::<Vec<Tensor>>(), gradacc1.iter().map(Tensor::from).collect::<Vec<Tensor>>()],
            updates_acc: [updatesacc0.iter().map(Tensor::from).collect::<Vec<Tensor>>(), updatesacc1.iter().map(Tensor::from).collect::<Vec<Tensor>>()],
        })
    }
}


impl Optimizer for AdaDelta
{
    fn name(&self) -> &str {
        Self::NAME
    }

    fn update_parameters(&mut self,
                         layer: &mut dyn Layer,
                         layer_idx: usize
    ) {
        if let Some((mut param, dparam)) = layer.parameters_mut() {
            for i in 0..param.len() {
                // Accumulate gradients
                self.grad_acc[i][layer_idx] = &self.grad_acc[i][layer_idx] * self.decay_rate + &(dparam[i] * dparam[i]) * (1. - self.decay_rate);
                // Compute update
                let update = - sqrt(&(&self.updates_acc[i][layer_idx] + self.eps)) / sqrt(&(&self.grad_acc[i][layer_idx] + self.eps)) * dparam[i];
                // Accumulate updates
                self.updates_acc[i][layer_idx] = &self.updates_acc[i][layer_idx] * self.decay_rate + &(&update * &update) * (1. - self.decay_rate);

                self.grad_acc[i][layer_idx].eval();
                self.updates_acc[i][layer_idx].eval();

                // Apply update
                *param[i] += update;
            }
        }
    }

    fn initialize_parameters(&mut self, layers_dims: Vec<(Dim, Dim)>) {
        for dim in layers_dims {
            self.grad_acc[0].push(Tensor::zeros(dim.0));
            self.updates_acc[0].push(Tensor::zeros(dim.0));
            self.grad_acc[1].push(Tensor::zeros(dim.1));
            self.updates_acc[1].push(Tensor::zeros(dim.1));
        }
    }

    fn save(&self, file: &hdf5::File) -> Result<(), Error> {
        let optimizer = file.create_group("optimizer")?;

        let opt_type = optimizer.new_dataset::<hdf5::types::VarLenUnicode>().create("type", 1)?;
        opt_type.write(&[hdf5::types::VarLenUnicode::from_str(Self::NAME).unwrap()])?;

        let decay_rate = optimizer.new_dataset::<PrimitiveType>().create("decay_rate", 1)?;
        decay_rate.write(&[self.decay_rate])?;

        let eps = optimizer.new_dataset::<PrimitiveType>().create("eps", 1)?;
        eps.write(&[self.eps])?;

        save_vec_tensor(&optimizer, &self.grad_acc[0], "grad_acc_0")?;
        save_vec_tensor(&optimizer, &self.grad_acc[1], "grad_acc_1")?;
        save_vec_tensor(&optimizer, &self.updates_acc[0], "updates_acc_0")?;
        save_vec_tensor(&optimizer, &self.updates_acc[1], "updates_acc_1")?;

        Ok(())
    }
}