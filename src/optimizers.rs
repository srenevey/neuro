use arrayfire::*;
use crate::layers::Layer;
use crate::tensor::*;
use std::fs;
use std::io::BufWriter;
use std::io;
use std::io::Write;
use byteorder::{BigEndian, WriteBytesExt};


pub trait Optimizer
{
    fn update_parameters(&mut self, layer: &mut Box<dyn Layer>, layer_idx: usize);
    fn update_time_step(&mut self) {}
    fn initialize_opt_params(&mut self, layers_dims: Vec<(Dim4, Dim4)>);
    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()>;
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
}

/// Stochastic Gradient Descent
///
/// The stochastic gradient descent performs gradient descent with momentum. For every parameter W in the layer, the value
///
/// <div align="center"><img src="https://latex.codecogs.com/svg.latex?V_{dW}&space;=&space;\beta&space;V_{dW}&space;&plus;&space;(1&space;-&space;\beta)dW" title="V_{dW} = \beta V_{dW} + (1 - \beta)dW" /></div>
///
/// is computed where <img src="https://latex.codecogs.com/svg.latex?\beta" title="\beta" /> is the momentum. The parameter is then updated according to
///
/// <div align="center"><img src="https://latex.codecogs.com/svg.latex?W&space;=&space;W&space;-&space;\alpha&space;V_{dW}" title="W = W - \alpha V_{dW}" /></div>
///
/// with <img src="https://latex.codecogs.com/svg.latex?\alpha" title="\alpha" /> the learning rate.
///
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

    /// Creates a Stochastic Gradient Descent optimizer with Nesterov momentum estimation.
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
    fn update_parameters(&mut self, layer: &mut Box<dyn Layer>, layer_idx: usize) {
        match layer.parameters_mut() {
            Some((mut param, dparam)) => {

                self.vdw[layer_idx] = &self.vdw[layer_idx] * self.momentum + dparam[0] * (1. - self.momentum);
                self.vdb[layer_idx] = &self.vdb[layer_idx] * self.momentum + dparam[1] * (1. - self.momentum);

                self.vdw[layer_idx].eval();
                self.vdb[layer_idx].eval();

                *param[0] -= &self.vdw[layer_idx] * self.learning_rate;
                *param[1] -= &self.vdb[layer_idx] * self.learning_rate;
            },
            None => {}
        }
    }

    /*
    fn update_parameters(&mut self, layer: &mut Box<dyn Layer>, layer_idx: usize) {
        match layer.parameters() {
            Some(param) => {
                match layer.dparameters() {
                    Some(dparam) => {
                        self.vdw[layer_idx] = mul(&self.momentum, &self.vdw[layer_idx], true) + mul(&(1. - self.momentum), dparam[0], true);
                        self.vdb[layer_idx] = mul(&self.momentum, &self.vdb[layer_idx], true) + mul(&(1. - self.momentum), dparam[1], true);

                        self.vdw[layer_idx].eval();
                        self.vdb[layer_idx].eval();

                        let updated_weights = param[0] - mul(&self.learning_rate, &self.vdw[layer_idx], true);
                        let updated_biases = param[1] -  mul(&self.learning_rate, &self.vdb[layer_idx], true);
                        layer.set_parameters(vec![updated_weights, updated_biases]);
                    },
                    None => panic!("The layer has some parameters but the gradient with respect to these parameters return None.")
                }
            },
            None => {}
        }
    }
    */

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


/// Adam - Adaptive moments estimation
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
                         layer: &mut Box<dyn Layer>,
                         layer_idx: usize
    ) {
        match layer.parameters_mut() {
            Some((mut param, dparam)) => {

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
            },
            None => {}
        }
    }
    /*
    fn update_parameters(&mut self, layer: &mut Box<dyn Layer>, layer_idx: usize) {
        match layer.parameters() {
            Some(param) => {
                match layer.dparameters() {
                    Some(dparam) => {
                        self.iteration += 1;
                        //self.vdw[layer_idx] = add(&mul(&self.beta1, &self.vdw[layer_idx], false), &mul(&(1. - self.beta1), layer.dweights(), false), false);
                        //self.vdb[layer_idx] = add(&mul(&self.beta1, &self.vdb[layer_idx], false), &mul(&(1. - self.beta1), layer.dbiases(), false), false);
                        //self.sdw[layer_idx] = add(&mul(&self.beta2, &self.sdw[layer_idx], false), &mul(&(1. - self.beta2), &(layer.dweights() * layer.dweights()), false), false);
                        //self.sdb[layer_idx] = add(&mul(&self.beta2, &self.sdb[layer_idx], false), &mul(&(1. - self.beta2), &(layer.dbiases() * layer.dbiases()), false), false);

                        self.vdw[layer_idx] = add(&mul(&self.beta1, &self.vdw[layer_idx], false), &mul(&(1. - self.beta1), dparam[0], false), false);
                        self.vdb[layer_idx] = add(&mul(&self.beta1, &self.vdb[layer_idx], false), &mul(&(1. - self.beta1), dparam[1], false), false);
                        self.sdw[layer_idx] = add(&mul(&self.beta2, &self.sdw[layer_idx], false), &mul(&(1. - self.beta2), &(dparam[0] * dparam[0]), false), false);
                        self.sdb[layer_idx] = add(&mul(&self.beta2, &self.sdb[layer_idx], false), &mul(&(1. - self.beta2), &(dparam[1] * dparam[1]), false), false);


                        self.vdw[layer_idx].eval();
                        self.vdb[layer_idx].eval();
                        self.sdw[layer_idx].eval();
                        self.sdb[layer_idx].eval();

                        let vdw_corr = &self.vdw[layer_idx] / (1. - self.beta1.powi(self.iteration as i32));
                        let vdb_corr = &self.vdb[layer_idx] / (1. - self.beta1.powi(self.iteration as i32));
                        let sdw_corr = &self.sdw[layer_idx] / (1. - self.beta2.powi(self.iteration as i32));
                        let sdb_corr = &self.sdb[layer_idx] / (1. - self.beta2.powi(self.iteration as i32));

                        let updated_weights = param[0] - mul(&self.learning_rate, &div(&vdw_corr, &add(&sqrt(&sdw_corr), &self.eps, false), false), false);
                        let updated_biases = param[1] - mul(&self.learning_rate, &div(&vdb_corr, &add(&sqrt(&sdb_corr), &self.eps, false), false), false);

                        layer.set_parameters(vec![updated_weights, updated_biases]);
                    },
                    None => panic!("The layer has some parameters but the gradient with respect to these parameters return None.")
                }
            },
            None => {}
        }
    }
    */

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


/// Root Mean Square Prop
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
                         layer: &mut Box<dyn Layer>,
                         layer_idx: usize
    ) {
        match layer.parameters_mut() {
            Some((mut param, dparam)) => {
                self.sdw[layer_idx] = &self.sdw[layer_idx] * self.decay_rate + &(dparam[0] * dparam[0]) * (1. - self.decay_rate);
                self.sdb[layer_idx] = &self.sdb[layer_idx] * self.decay_rate + &(dparam[1] * dparam[1]) * (1. - self.decay_rate);

                self.sdw[layer_idx].eval();
                self.sdb[layer_idx].eval();

                *param[0] -= dparam[0] / (&sqrt(&self.sdw[layer_idx]) + self.eps) * self.learning_rate;
                *param[1] -= dparam[1] / (&sqrt(&self.sdb[layer_idx]) + self.eps) * self.learning_rate;
            },
            None => {}
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
