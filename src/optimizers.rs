use arrayfire::*;
use crate::layers::Layer;

pub trait Optimizer
{
    fn update_parameters(&mut self, layer: &mut Box<dyn Layer>, layer_idx: usize);
    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>);
}


/// Stochastic Gradient Descent
pub struct SGD {
    learning_rate: f64,
    momentum: f64,
    vdw: Vec<Array<f64>>,
    vdb: Vec<Array<f64>>,
}

impl SGD {
    pub fn new(learning_rate: f64) -> SGD {
        SGD {
            learning_rate,
            momentum: 0.0,
            vdw: Vec::new(),
            vdb: Vec::new(),
        }
    }

    pub fn with_param(learning_rate: f64, momentum: f64) -> SGD {
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
    fn update_parameters(&mut self, layer: &mut Box<dyn Layer>, layer_idx: usize) {
        self.vdw[layer_idx] = mul(&self.momentum, &self.vdw[layer_idx], false) + mul(&(1. - self.momentum), layer.dparameters()[0], false);
        self.vdb[layer_idx] = mul(&self.momentum, &self.vdb[layer_idx], false) + mul(&(1. - self.momentum), layer.dparameters()[1], false);

        self.vdw[layer_idx].eval();
        self.vdb[layer_idx].eval();

        let updated_weights = layer.parameters()[0] - mul(&self.learning_rate, &self.vdw[layer_idx], false);
        updated_weights.eval();
        let updated_biases = layer.parameters()[1] -  mul(&self.learning_rate, &self.vdb[layer_idx], false);
        updated_biases.eval();
        layer.set_parameters(vec![updated_weights, updated_biases]);
        //layer.set_weights(updated_weights);
        //layer.set_biases(updated_biases);
    }

    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>) {
        for dim in layers_dims {
            self.vdw.push(constant(0.0f64, dim.0));
            self.vdb.push(constant(0.0f64, dim.1));
        }
    }
}


/// Adam - Adaptive moment estimation
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    iteration: u64,
    v: Vec<Vec<Array<f64>>>,
    s: Vec<Vec<Array<f64>>>,
    vdw: Vec<Array<f64>>,
    vdb: Vec<Array<f64>>,
    sdw: Vec<Array<f64>>,
    sdb: Vec<Array<f64>>,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Adam {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            iteration: 0,
            v: Vec::new(),
            s: Vec::new(),
            vdw: Vec::new(),
            vdb: Vec::new(),
            sdw: Vec::new(),
            sdb: Vec::new(),
        }
    }

    pub fn with_param(learning_rate: f64, beta1: f64, beta2: f64, eps: f64) -> Adam {
        Adam {
            learning_rate,
            beta1,
            beta2,
            eps,
            iteration: 0,
            v: Vec::new(),
            s: Vec::new(),
            vdw: Vec::new(),
            vdb: Vec::new(),
            sdw: Vec::new(),
            sdb: Vec::new(),
        }
    }
}

impl Optimizer for Adam
{
    fn update_parameters(&mut self, layer: &mut Box<dyn Layer>, layer_idx: usize) {
        self.iteration += 1;
        //self.vdw[layer_idx] = add(&mul(&self.beta1, &self.vdw[layer_idx], false), &mul(&(1. - self.beta1), layer.dweights(), false), false);
        //self.vdb[layer_idx] = add(&mul(&self.beta1, &self.vdb[layer_idx], false), &mul(&(1. - self.beta1), layer.dbiases(), false), false);
        //self.sdw[layer_idx] = add(&mul(&self.beta2, &self.sdw[layer_idx], false), &mul(&(1. - self.beta2), &(layer.dweights() * layer.dweights()), false), false);
        //self.sdb[layer_idx] = add(&mul(&self.beta2, &self.sdb[layer_idx], false), &mul(&(1. - self.beta2), &(layer.dbiases() * layer.dbiases()), false), false);

        self.vdw[layer_idx] = add(&mul(&self.beta1, &self.vdw[layer_idx], false), &mul(&(1. - self.beta1), layer.dparameters()[0], false), false);
        self.vdb[layer_idx] = add(&mul(&self.beta1, &self.vdb[layer_idx], false), &mul(&(1. - self.beta1), layer.dparameters()[1], false), false);
        self.sdw[layer_idx] = add(&mul(&self.beta2, &self.sdw[layer_idx], false), &mul(&(1. - self.beta2), &(layer.dparameters()[0] * layer.dparameters()[0]), false), false);
        self.sdb[layer_idx] = add(&mul(&self.beta2, &self.sdb[layer_idx], false), &mul(&(1. - self.beta2), &(layer.dparameters()[1] * layer.dparameters()[1]), false), false);


        self.vdw[layer_idx].eval();
        self.vdb[layer_idx].eval();
        self.sdw[layer_idx].eval();
        self.sdb[layer_idx].eval();

        let vdw_corr = &self.vdw[layer_idx] / (1. - self.beta1.powi(self.iteration as i32));
        let vdb_corr = &self.vdb[layer_idx] / (1. - self.beta1.powi(self.iteration as i32));
        let sdw_corr = &self.sdw[layer_idx] / (1. - self.beta2.powi(self.iteration as i32));
        let sdb_corr = &self.sdb[layer_idx] / (1. - self.beta2.powi(self.iteration as i32));

        let updated_weights = layer.parameters()[0] - mul(&self.learning_rate, &div(&vdw_corr, &add(&sqrt(&sdw_corr), &self.eps, false), false), false);
        let updated_biases = layer.parameters()[1] - mul(&self.learning_rate, &div(&vdb_corr, &add(&sqrt(&sdb_corr), &self.eps, false), false), false);
        updated_weights.eval();
        updated_biases.eval();

        layer.set_parameters(vec![updated_weights, updated_biases]);
        //layer.set_weights(updated_weights);
        //layer.set_biases(updated_biases);
    }

    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>) {
        for dim in layers_dims {
            self.vdw.push(constant(0.0f64, dim.0));
            self.vdb.push(constant(0.0f64, dim.1));
            self.sdw.push(constant(0.0f64, dim.0));
            self.sdb.push(constant(0.0f64, dim.1));
        }
    }
}


/// Root Mean Square prop
pub struct RMSprop {
    learning_rate: f64,
    beta2: f64,
    eps: f64,
    sdw: Vec<Array<f64>>,
    sdb: Vec<Array<f64>>,
}

impl RMSprop {
    pub fn new(learning_rate: f64) -> RMSprop {
        RMSprop {
            learning_rate,
            beta2: 0.9,
            eps: 1e-8,
            sdw: Vec::new(),
            sdb: Vec::new(),
        }
    }

    pub fn with_param(learning_rate: f64, beta2: f64, eps: f64) -> RMSprop {
        RMSprop {
            learning_rate,
            beta2,
            eps,
            sdw: Vec::new(),
            sdb: Vec::new(),
        }
    }
}

impl Optimizer for RMSprop
{
    fn update_parameters(&mut self, layer: &mut Box<dyn Layer>, layer_idx: usize) {
        self.sdw[layer_idx] = add(&mul(&self.beta2, &self.sdw[layer_idx], false), &mul(&(1. - self.beta2), &(layer.dparameters()[0] * layer.dparameters()[0]), false), false);
        self.sdb[layer_idx] = add(&mul(&self.beta2, &self.sdb[layer_idx], false), &mul(&(1. - self.beta2), &(layer.dparameters()[1] * layer.dparameters()[1]), false), false);

        self.sdw[layer_idx].eval();
        self.sdb[layer_idx].eval();

        let updated_weights = layer.parameters()[0] - mul(&self.learning_rate, &div(layer.dparameters()[0], &add(&sqrt(&self.sdw[layer_idx]), &self.eps, false), false), false);
        let updated_biases = layer.parameters()[1] - mul(&self.learning_rate, &div(layer.dparameters()[1], &add(&sqrt(&self.sdb[layer_idx]), &self.eps, false), false), false);
        updated_weights.eval();
        updated_biases.eval();

        layer.set_parameters(vec![updated_weights, updated_biases]);

        //layer.set_weights(updated_weights);
        //layer.set_biases(updated_biases);
    }

    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>) {
        for dim in layers_dims {
            self.sdw.push(constant(0.0f64, dim.0));
            self.sdb.push(constant(0.0f64, dim.1));
        }
    }
}
