//! Base module to create neural networks.
use crate::data::{DataSet, BatchIterator, Scaling};
use crate::layers::*;
use crate::losses::*;
use crate::metrics::*;
use crate::optimizers::*;
use crate::regularizers::*;
use crate::Tensor;
use crate::tensor::*;

use std::fmt;
use std::fs;
use std::io;
use std::io::{BufWriter, BufReader, Read, Write};
use std::path::Path;

use arrayfire::*;

enum NetworkError {
    NoLayers
}

/// Structure representing a neural network.
pub struct Network<'a, D, O, L>
where D: DataSet, O: Optimizer, L: Loss
{
    data: &'a D,
    layers: Vec<Box<dyn Layer>>,
    loss_function: L,
    metrics: Option<Vec<Metrics>>,
    optimizer: O,
    regularizer: Option<Regularizer>,
    input_shape: Dim4,
    output_shape: Dim4,
    classes: Option<Vec<String>>,
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
    /// * `metrics`: (optional) vector of Metrics used to evaluate the model
    /// * `regularizer`: (optional) method used to regularize the model
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
            input_shape: data.input_shape(),
            output_shape: data.output_shape(),
            classes: data.classes(),
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
    fn forward_mut(&mut self, input: &mut Tensor) {
        /*
        self.layers.iter_mut().fold(
            input,
            |a_prev, layer| layer.compute_activation_mut(&a_prev)
        )
        */

        //let mut a_prev = x.copy();
        for layer in self.layers.iter_mut() {
            *input = layer.compute_activation_mut(input);
        }
        //input.copy()
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
    /// The training data are shuffled at the beginning of each epoch, before batches are created.
    ///
    /// # Arguments
    /// * `batch_size`: size of the mini-batches used for training
    /// * `epochs`: number of epochs to train for
    /// * `print_loss`: number of epochs between two computations and printing of the validation loss
    ///
    pub fn fit(&mut self,
               batch_size: u64,
               epochs: u64,
               print_loss: Option<u64>
    ) {
        device_gc();

        let (name, platform, _, _) = device_info();
        println!("Running on {} using {}.", name, platform);

        self.initialize_optimizer();


        for epoch in 1..=epochs {

            let (x_train_shuffled, y_train_shuffled) = Tensor::shuffle(self.data.x_train(), self.data.y_train());
            let batches = BatchIterator::new((&x_train_shuffled, &y_train_shuffled), batch_size);


            //mem_info!("Memory used by Arrayfire");
            //device_gc();

            // Iterate over the batches
            for (mut mini_batch_x, mini_batch_y) in batches {


                // Compute a pass on the network
                self.forward_mut(&mut mini_batch_x);
                self.backward(&mini_batch_x, &mini_batch_y);

                // Update the parameters of the model
                self.update_parameters();
                //println!("done updating the parameters");

                //device_gc();
                //println!("batch: {} done", i);
                //mem_info!("Memory used by Arrayfire");
            }

            // Compute and print the losses and the metrics
            match print_loss {
                Some(print_iter) => {
                    if epoch % print_iter == 0 {

                        // Compute the loss on the training set
                        let train_loss = self.compute_train_loss(batch_size);
                        //println!("Train loss computed");

                        // Compute the loss on the validation set
                        let (valid_loss, valid_pred) = self.evaluate(batch_size);
                        //println!("Validation loss computed");

                        // Evaluate the metrics on the validation set
                        let metrics_values = self.compute_metrics(&valid_pred, &self.data.y_valid(), batch_size);
                        //println!("Metrics computed");

                        println!("epoch: {}, train_loss: {}, valid_loss: {}, metrics: {:?}", epoch, train_loss, valid_loss, metrics_values);
                    }
                },
                None => {},
            }
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
        let regularization = match &self.regularizer {
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
            None => 0.0,
        };
        self.loss_function.eval(y_pred, y_true) + regularization
    }


    /// Computes the loss on the training set.
    ///
    /// # Arguments
    /// * `batch_size`: size of the mini-batches used to propagate the training data
    ///
    fn compute_train_loss(&self, batch_size: u64) -> PrimitiveType {
        let mut loss = 0.;

        // Create batch iterator
        let batches = BatchIterator::new((self.data.x_train(), self.data.y_train()), batch_size);
        let num_batches = batches.num_batches() as PrimitiveType;
        for (mini_batch_x, mini_batch_y) in batches {
            let y_pred_batch = self.forward(&mini_batch_x);
            loss += self.compute_loss(&y_pred_batch, &mini_batch_y);
        }
        loss / num_batches
    }


    /// Evaluates the model with the validation data.
    ///
    /// # Returns
    /// Returns the loss computed on the validation set and the predictions in a tuple.
    ///
    fn evaluate(&self, batch_size: u64) -> (PrimitiveType, Tensor) {
        let mut loss = 0.;
        let mut y_pred = Array::new_empty(self.output_shape);

        // Create batch iterator
        let batches = BatchIterator::new((self.data.x_valid(), self.data.y_valid()), batch_size);
        let num_batches = batches.num_batches() as PrimitiveType;
        let mut count = 0;
        for (mini_batch_x, mini_batch_y) in batches {
            let y_pred_batch = self.forward(&mini_batch_x);
            loss += self.compute_loss(&y_pred_batch, &mini_batch_y);

            if count == 0 {
                y_pred = y_pred_batch;
            } else {
                y_pred = join(3, &y_pred, &y_pred_batch);
            }
            count += 1;
        }
        (loss / num_batches, y_pred)
    }


    /// Evaluates the metrics.
    ///
    /// # Arguments
    /// * `y_pred`: labels predicted by the model
    /// * `y_true`: true labels
    /// * `batch_size`: y_pred and y_true are split in chunks of batch_size to reduce the memory footprint
    ///
    /// # Returns
    /// Vector containing the values for each metric.
    ///
    fn compute_metrics(&self,
                       y_pred: &Tensor,
                       y_true: &Tensor,
                       batch_size: u64
    ) -> Vec<PrimitiveType> {
        let num_metrics = match &self.metrics {
            Some(m) => m.len(),
            None => 0
        };

        let mut metrics_values: Vec<PrimitiveType> = vec![0.; num_metrics];

        match &self.metrics {
            Some(m) => {
                let batches = BatchIterator::new((y_pred, y_true), batch_size);
                let num_batches = batches.num_batches() as PrimitiveType;

                for (y_pred_batch, y_true_batch) in batches {
                    for (i, metrics) in m.iter().enumerate() {
                        let metrics_value = metrics.eval(&y_pred_batch, &y_true_batch);
                        metrics_values[i] += metrics_value;
                    }
                }
                // Divide by number of batches
                for metric in metrics_values.iter_mut() {
                    *metric /= num_batches;
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



    /// Computes the output of the network for the given input.
    ///
    /// # Arguments
    /// * `input`: tensor of inputs. Multiple samples can be evaluated at once by stacking them along the fourth dimension of the tensor.
    ///
    /// # Returns
    /// Tensor of the predicted output
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

    /// Predicts the class for the input.
    ///
    /// # Arguments
    /// * `input`: tensor of inputs. Multiple samples can be evaluated at once by stacking them along the fourth dimension of the tensor.
    ///
    /// # Returns
    /// Vector of tuples containing the predicted class and the probability for each sample.
    ///
    pub fn predict_class(&self, input: &Tensor) -> Vec<(String, PrimitiveType)> {
        let batch_size = input.dims().get()[3] as usize;
        let mut predictions: Vec<(String, PrimitiveType)> = Vec::with_capacity(batch_size);

        // Compute the output of the network and retrieve value and index of maximum value
        let y_pred = self.predict(input);
        let (probabilities_tensor, class_idxs_tensor) = imax(&y_pred, 0);

        // Retrieve values from GPU
        let mut probabilities: Vec<PrimitiveType> = vec![0 as PrimitiveType; batch_size as usize];
        let mut class_idxs: Vec<u32> = vec![0; batch_size as usize];
        probabilities_tensor.host(&mut probabilities);
        class_idxs_tensor.host(&mut class_idxs);

        // Build output structure
        match &self.classes {
            Some(classes) => {
                for i in 0..batch_size {
                    // Handle multiclass and binary classification
                    if classes.len() > 2 {
                        predictions.push((classes[class_idxs[i] as usize].clone(), probabilities[i]));
                    } else {
                        let idx = probabilities[i].round() as usize;
                        if probabilities[i] < 0.5 { probabilities[i] = 1. - probabilities[i]; }
                        predictions.push((classes[idx].clone(), probabilities[i]));
                    }
                }
            },
            None => panic!("The network is not aware of any classes."),
        }
        predictions
    }

    /*
    /// Saves the model.
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
    */
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