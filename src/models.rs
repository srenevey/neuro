use crate::data::{DataSet, Scaling};
use crate::layers::*;
use crate::losses::*;
use crate::metrics::*;
use crate::optimizers::*;
use crate::regularizers::*;
use crate::Tensor;

use std::fmt;
use std::fs;
use std::io;
use std::io::{BufWriter, BufReader, Read, Write};
use std::path::Path;

use arrayfire::*;
use crate::tensor::PrimitiveType;


enum NetworkError {
    NoLayers
}

pub struct Network<'a, D, O, L>
where D: DataSet, O: Optimizer, L: Loss
{
    data: &'a D,
    layers: Vec<Box<dyn Layer>>,
    loss_function: L,
    metrics: Option<Vec<Metrics>>,
    optimizer: O,
    regularizer: Option<Regularizer>,
}


impl<'a, D, O, L> Network<'a, D, O, L>
where D: DataSet, O: Optimizer, L: Loss
{
    /// Creates an empty neural network.
    ///
    /// # Arguments
    /// * `data`: dataset used to train the neural network
    /// * `loss_function`: loss function minimized in the optimization process
    /// * `optimizer`: algorithm used to optimize the parameters of the network
    /// * `metrics`: optional vector of metrics used to evaluate the model
    ///
    pub fn new(data: &'a D,
               loss_function: L,
               optimizer: O,
               metrics: Option<Vec<Metrics>>,
               regularizer: Option<Regularizer>
    ) -> Network<D, O, L> {
        Network {
            data,
            layers: Vec::new(),
            loss_function,
            metrics,
            optimizer,
            regularizer,
        }
    }

    /// Adds a layer to the network.
    ///
    /// # Arguments
    /// * `layer`: layer to be added
    ///
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        let input_shape = match self.layers.last() {
            Some(layer) => layer.output_shape(),
            None => self.data.input_shape(),
        };
        self.layers.push(layer);
        self.layers.last_mut().unwrap().initialize_parameters(input_shape);
        self.layers.last_mut().unwrap().set_regularizer(self.regularizer);
        device_gc();
    }


    /// Computes the output of the network for a given input.
    ///
    /// # Arguments
    /// * `input`: array containing the samples to be evaluated
    ///
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers.iter().fold(
            input.copy(),
            |a_prev, layer| layer.compute_activation(&a_prev)
        )
    }


    /// Computes a forward pass of the network.
    ///
    /// The intermediate linear activations computed during the forward pass are stored in each layer for efficient back propagation.
    ///
    /// # Arguments
    /// * `input`: array containing the samples fed into the model
    ///
    fn forward_mut(&mut self, input: &Tensor) -> Tensor {
        self.layers.iter_mut().fold(
            input.copy(),
            |a_prev, layer| layer.compute_activation_mut(&a_prev)
        )
        /*
        let mut a_prev = x.copy();
        for layer in self.layers.iter_mut() {
            a_prev = layer.compute_activation_mut(&a_prev);
        }
        a_prev
        */
    }

    /// Computes a backward pass of the network.
    ///
    /// # Arguments
    /// * `y_pred`: output of the network
    /// * `y_true`: true labels
    ///
    fn backward(&mut self,
                y_pred: &Tensor,
                y_true: &Tensor
    ) {
        self.layers.iter_mut().rev().fold(
            self.loss_function.grad(y_pred, y_true),
            |da_prev, layer| layer.compute_dactivation_mut(&da_prev)
        );

        /*
        let mut da_prev = self.loss_function.grad(y_pred, y_true);
        for layer in self.layers.iter_mut().rev() {
            da_prev = layer.compute_da_prev_mut(&da_prev);
        }
        */
    }

    /// Fits the neural network with the training data.
    ///
    /// # Arguments
    /// * `batch_size`: size of the mini-batches used for training
    /// * `epochs`: number of epochs to train for
    /// * `print_loss`: number of epochs between two computations of the validation loss
    ///
    pub fn fit(&mut self,
               batch_size: u64,
               epochs: u64,
               print_loss: Option<u64>
    ) {
        let (name, platform, _, _) = device_info();
        println!("Running on {} using {}.", name, platform);

        self.initialize_optimizer();
        //self.data.initialize_batches(batch_size);
        let (batch_size, num_batches) = self.number_batches(batch_size);

        for epoch in 1..=epochs {

            //self.data.shuffle();
            let mut train_loss = 0.;


            // If the losses are to be printed, compute the validation set loss before the weights are updated
            let mut valid_loss = 0.;
            let mut metrics_values = Vec::new();
            match print_loss {
                Some(print_iter) => {
                    if epoch % print_iter == 0 {

                        // Compute the loss for the validation set
                        let (loss, valid_pred) = self.evaluate();
                        valid_loss = loss;

                        // Evaluate the metrics
                        metrics_values = self.compute_metrics(&valid_pred, &self.data.y_valid());
                    }
                },
                None => {},
            }

            // Iterate over the batches
            for i in 0..num_batches {
                let (mini_batch_x, mini_batch_y) = self.data.batch(i, batch_size);

                //println!("{:?}", mini_batch_x.dims());

                // Compute activation of the last layer
                let final_activation = self.forward_mut(&mini_batch_x);
                //println!("done with forward prop");
                //println!("final activation dims: {} x {} x {} x {}", final_activation.dims().get()[0], final_activation.dims().get()[1], final_activation.dims().get()[2], final_activation.dims().get()[3]);

                //af_print!("", &final_activation);

                //let metrics_values = self.compute_metrics(&final_activation, &mini_batch_y);
                //println!("batch: {}, batch metrics: {:?}", i, metrics_values);

                let loss = self.compute_loss(&final_activation, &mini_batch_y);
                train_loss += loss;
                //println!("done computing the loss");

                // Compute the backward pass and update the parameters
                self.backward(&final_activation, &mini_batch_y);
                //println!("done with backward prop");
                self.update_parameters();
                //println!("done updating the parameters");

                //device_gc();
                //println!("batch: {}", i);
                //mem_info!("Memory used by Arrayfire");
            }

            train_loss /= num_batches as PrimitiveType;

            // Print the losses and the metrics
            match print_loss {
                Some(print_iter) => {
                    if epoch % print_iter == 0 {

                        /*
                        // Compute the loss for the validation set
                        let (valid_loss, valid_pred) = self.evaluate();

                        // Evaluate the metrics
                        let metrics_values = self.compute_metrics(&valid_pred, &self.data.y_valid());
                        */
                        println!("epoch: {}, train_loss: {}, valid_loss: {}, metrics: {:?}", epoch, train_loss, valid_loss, metrics_values);
                    }
                },
                None => {},
            }

            //mem_info!("Memory used by Arrayfire");
            //device_gc();

        }
    }

    /// Performs a sanity checks on the batch size and computes the number of batches.
    ///
    /// # Arguments
    /// * `batch_size`: desired batch size
    ///
    /// # Returns
    /// Tuple containing the batch size and number of batches.
    ///
    fn number_batches(&self, batch_size: u64) -> (u64, u64) {
        let num_train_samples = self.data.num_train_samples();
        if batch_size < num_train_samples {
            let num_batches = (num_train_samples as f64 / batch_size as f64).ceil() as u64;
            (batch_size, num_batches)
        } else {
            (num_train_samples, 1)
        }
    }

    /// Initializes the parameters of the optimizer.
    fn initialize_optimizer(&mut self) {
        let mut dims = Vec::<(Dim4, Dim4)>::new();
        for layer in self.layers.iter() {
            match layer.parameters() {
                Some(param) => dims.push((param[0].dims(), param[1].dims())),
                None => dims.push((Dim4::new(&[0, 0, 0, 0]), Dim4::new(&[0, 0, 0, 0])))
            }

        }
        self.optimizer.initialize_opt_params(dims);
    }


    /// Computes the loss for the predicted output.
    ///
    /// # Arguments
    /// * `y_pred`: predicted output of the model
    /// * `y_true`: true labels
    ///
    fn compute_loss(&self,
                    y_pred: &Tensor,
                    y_true: &Tensor
    ) -> PrimitiveType {
        let regularization: PrimitiveType = match &self.regularizer {
            Some(regularizer) => {
                let mut weights: Vec<&Tensor> = Vec::new();
                for layer in self.layers.iter() {
                    match layer.parameters() {
                        Some(params) => weights.push(params[0]),
                        None => {}
                    }
                }
                regularizer.eval(weights)
            },
            None => 0.0
        };

        self.loss_function.eval(y_pred, y_true) + regularization
    }


    /// Evaluate the metrics.
    ///
    /// # Arguments
    /// * `y_pred`: labels predicted by the model
    /// * `y_true`: true labels
    ///
    fn compute_metrics(&self,
                       y_pred: &Tensor,
                       y_true: &Tensor
    ) -> Vec<PrimitiveType> {
        let mut metrics_values: Vec<PrimitiveType> = Vec::new();
        match &self.metrics {
            Some(m) => {
                for metrics in m {
                    let metrics_value = metrics.eval(y_pred, y_true);
                    metrics_values.push(metrics_value);
                }
            },
            None => {},
        }
        metrics_values
    }

    /// Updates the parameters of the model.
    fn update_parameters(&mut self) {
        self.optimizer.update_time_step();
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            self.optimizer.update_parameters(layer, idx);
        }
    }


    /// Evaluates the model with the validation data.
    ///
    /// Returns the loss computed on the validation set and the predictions.
    fn evaluate(&self) -> (PrimitiveType, Tensor) {
        // Compute prediction
        let y_pred = self.forward(&self.data.x_valid());

        // Compute the loss for the validation set
        let loss = self.compute_loss(&y_pred, &self.data.y_valid());
        (loss, y_pred)
    }


    /// Computes the output of the network for the given input.
    ///
    /// # Arguments
    /// * `input`: array containing the inputs to be evaluated. Several inputs can be passed at once by stacking them on the fourth dimension.
    ///
    pub fn predict(&self, input: &Tensor) -> Tensor {

        // Normalize the input if the network has been trained on normalized samples
        match self.data.x_train_stats() {
            Some(stats) => {},
            None => {}
        }

        // Compute the output of the network
        let mut y_pred = self.forward(&input);

        // Unscale the output values
        match self.data.y_train_stats() {
            Some(stats) => {
                match stats.0 {
                    Scaling::Normalized => {
                        let y_min = &stats.1;
                        let y_max = &stats.2;
                        let range = y_max - y_min;
                        y_pred = add(&mul(&range, &y_pred, true), y_max, true);
                    },
                    Scaling::Standarized => {
                        let mean = &stats.1;
                        let std = &stats.2;
                        y_pred = add(&mul(std, &y_pred, true), mean, true);
                    }
                }
            },
            None => {},
        }

        y_pred
    }

    /// Save the model
    ///
    /// # Arguments
    /// * `filename`: name of the file where the model is saved
    ///
    pub fn save(&self, filename: &str) -> io::Result<()> {
        let f = fs::File::create(filename)?;
        {
            let mut writer = BufWriter::new(f);
            writer.write(&self.loss_function.id().to_be_bytes())?;
            writer.write(b"\n")?;
            self.optimizer.save(&mut writer)?;

            // Save layers
            for layer in self.layers.iter() {
                layer.save(&mut writer)?;
            }
        }
        Ok(())
    }
}

impl<'a, D, O, L> fmt::Display for Network<'a, D, O, L>
where D: DataSet, O: Optimizer, L: Loss
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Layer \t\t Parameters\n");
        write!(f, "------------------------\n");
        for layer in self.layers.iter() {
            println!("{}", layer);
        }
        Ok(())
    }
}