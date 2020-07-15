//! Base module to create neural networks.
use arrayfire::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::fmt;
use std::str::FromStr;
use rand::prelude::*;

use crate::data::{DataSet, BatchIterator};
use crate::errors::Error;
use crate::io::*;
use crate::layers::*;
use crate::losses::*;
use crate::metrics::*;
use crate::optimizers::*;
use crate::regularizers::*;
use crate::tensor::*;

enum Mode {
    Test,
    Train,
    Valid,
}


/// Structure representing a neural network.
pub struct Network
{
    layers: Vec<Box<dyn Layer>>,
    loss_function: Box<dyn Loss>,
    optimizer: Box<dyn Optimizer>,
    regularizer: Option<Regularizer>,
    input_shape: Dim,
    output_shape: Dim,
    classes: Option<Vec<String>>,
}

impl Network
{
    /// Creates an empty neural network.
    ///
    /// The input shape must be in the form [height, width, channel, 1]. Mini-batches are created along the fourth dimension.
    pub fn new(input_shape: Dim,
               loss_function: Box<dyn Loss>,
               optimizer: Box<dyn Optimizer>,
               regularizer: Option<Regularizer>
    ) -> Result<Network, Error> {

        // Generate a random seed used by ArrayFire
        let mut rng = thread_rng();
        set_seed(rng.gen());

        Ok(Network {
            layers: Vec::new(),
            loss_function,
            optimizer,
            regularizer,
            input_shape,
            output_shape: Dim::new(&[0, 0, 0, 0]),
            classes: None,
        })
    }

    /// Adds a layer to the network.
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        let input_shape = match self.layers.last() {
            Some(layer) => layer.output_shape(),
            None => self.input_shape,
        };
        self.layers.push(layer);

        if let Some(layer) = self.layers.last_mut() {
            layer.initialize_parameters(input_shape);
            layer.set_regularizer(self.regularizer);

            self.output_shape = layer.output_shape();
        }
    }


    /// Computes the output of the network for a given input.
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers.iter().fold(
            input.copy(),
            |a_prev, layer| layer.compute_activation(&a_prev)
        )
    }


    /// Computes a forward pass of the network.
    ///
    /// The intermediate linear activations computed during the forward pass are stored in each layer for efficient back propagation.
    fn forward_mut(&mut self, input: &mut Tensor) {
        for layer in self.layers.iter_mut() {
            *input = layer.compute_activation_mut(input);
        }
    }

    /// Computes a backward pass of the network.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - The output of the network.
    /// * `y_true` - The true labels.
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
    /// The training data are shuffled at the beginning of each epoch, before batches are created. The progress is printed
    /// at every `print_loss` epoch.
    pub fn fit<T: DataSet>(&mut self,
               data: &T,
               batch_size: u64,
               epochs: u64,
               print_loss: Option<u64>,
               metrics: Option<Vec<Metrics>>,
    ) {
        let device = get_device();
        let (name, platform, _, _) = device_info();
        println!("Running on {} using {}.", name, platform);

        self.initialize_optimizer();

        // If it's a classification problem, store the classes.
        self.classes = data.classes();

        // Initialize progress bar
        let num_bins = match print_loss {
            Some(p) => {
                let num_batches_train = 2 * p * (data.num_train_samples() as f64 / batch_size as f64).ceil() as u64;
                let num_batches_valid = (data.num_valid_samples() as f64 / batch_size as f64).ceil() as u64;
                num_batches_train + num_batches_valid
            },
            None => epochs
        };
        let mut progress_bar = ProgressBar::new(num_bins);
        let sty = ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:50}] {msg}")
            .progress_chars("##-");
        progress_bar.set_style(sty.clone());


        // Train
        for epoch in 1..=epochs {
            let (x_train_shuffled, y_train_shuffled) = Tensor::shuffle(data.x_train(), data.y_train());
            let batches = BatchIterator::new((&x_train_shuffled, &y_train_shuffled), batch_size);

            // Reset progress bar
            if progress_bar.is_finished() {
                progress_bar = ProgressBar::new(num_bins);
                progress_bar.set_style(sty.clone());
            }
            progress_bar.set_message(&format!("epoch: {}/{}", epoch, epochs));


            // Iterate over the batches
            for (mut mini_batch_x, mini_batch_y) in batches {

                // Compute a pass on the network
                self.forward_mut(&mut mini_batch_x);
                self.backward(&mini_batch_x, &mini_batch_y);

                // Update the parameters of the model
                self.update_parameters();

                sync(device);
                progress_bar.inc(1);
            }

            // Compute and print the losses and the metrics
            if let Some(print_iter) = print_loss {
                if epoch % print_iter == 0 {

                    // Compute the loss and metrics evaluated on the training set
                    let (train_loss, train_pred) = self.compute_loss(data, batch_size, Mode::Train, Some(&progress_bar));
                    let train_metrics_values = self.compute_metrics(&train_pred, &data.y_train(), batch_size, &metrics);

                    // Compute the loss and metrics evaluated on the validation set
                    if data.num_valid_samples() > 0 {
                        let (valid_loss, valid_pred) = self.compute_loss(data, batch_size, Mode::Valid, Some(&progress_bar));
                        let valid_metrics_values = self.compute_metrics(&valid_pred, &data.y_valid().unwrap(), batch_size, &metrics);
                        progress_bar.finish_with_message(&format!("epoch: {}/{}, train_loss: {}, train_metrics: {:?}, valid_loss: {}, valid_metrics: {:?}", epoch, epochs, train_loss, train_metrics_values, valid_loss, valid_metrics_values));

                    } else {
                        progress_bar.finish_with_message(&format!("epoch: {}/{}, train_loss: {}, train_metrics: {:?}", epoch, epochs, train_loss, train_metrics_values));
                    }
                }
            }
        }
    }


    /// Initializes the parameters of the optimizer.
    fn initialize_optimizer(&mut self) {
        let mut dims = Vec::<(Dim4, Dim4)>::new();
        for layer in self.layers.iter() {
            match layer.parameters() {
                Some(param) => dims.push((param[0].dims(), param[1].dims())),
                None => dims.push((Dim4::new(&[1, 1, 1, 1]), Dim4::new(&[1, 1, 1, 1])))
            }
        }
        self.optimizer.initialize_parameters(dims);
    }


    /// Computes the loss and the predicted output.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset containing the training, validation, and optionally test data.
    /// * `batch_size` - The size of the mini-batches used to compute the loss.
    /// * `mode` - Flag specifying whether the loss is computed on the training, validation, or test set.
    /// * `bar` - The reference to a progress bar used to show the training progress.
    ///
    /// # Return value
    ///
    /// Tuple containing the loss and the predicted output.
    fn compute_loss<T: DataSet>(&self,
                    data: &T,
                    batch_size: u64,
                    mode: Mode,
                    progress_bar: Option<&ProgressBar>
    ) -> (PrimitiveType, Tensor) {
        let mut loss = 0.;
        let mut y_pred = Array::new_empty(self.output_shape);

        // Create batch iterator
        let (x, y) = match mode {
            Mode::Train => (data.x_train(), data.y_train()),
            Mode::Valid => (data.x_valid().unwrap(), data.y_valid().unwrap()),
            Mode::Test => (data.x_test().expect("No test samples have been provided."), data.y_test().expect("No test labels have been provided.")),
        };
        let batches = BatchIterator::new((x, y), batch_size);
        let num_batches = batches.num_batches() as PrimitiveType;

        for (count, (mini_batch_x, mini_batch_y)) in batches.enumerate() {
            let y_pred_batch = self.forward(&mini_batch_x);

            let regularization = match &self.regularizer {
                Some(regularizer) => {
                    let mut weights: Vec<&Tensor> = Vec::new();
                    for layer in self.layers.iter() {
                        if let Some(parameters) = layer.parameters() { weights.push(parameters[0]) }
                    }
                    regularizer.eval(weights)
                },
                None => 0.0,
            };
            loss += self.loss_function.eval(&y_pred_batch, &mini_batch_y) + regularization;

            if count == 0 {
                y_pred = y_pred_batch;
            } else {
                y_pred = join(3, &y_pred, &y_pred_batch);
            }

            if let Some(progress_bar) = progress_bar { progress_bar.inc(1) }
        }
        (loss / num_batches, y_pred)
    }


    /// Evaluates the model on the test set.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset containing the test data.
    /// * `metrics` - A vector containing the metrics that will be evaluated.
    pub fn evaluate<T: DataSet>(&self,
                                data: &T,
                                metrics: Option<Vec<Metrics>>
    ) {
        // TODO: find a way to automatically compute a batch size that fits in the available GPU/CPU memory
        let batch_size = 128;
        let (loss, y_pred) = self.compute_loss(data, batch_size, Mode::Test, None);
        let y_test = data.y_test().expect("The dataset does not contain any test data.");
        let metrics_values = self.compute_metrics(&y_pred, y_test, batch_size, &metrics);
        println!("Evaluation of the test set: loss: {}, metrics: {:?}", loss, metrics_values);
    }


    /// Evaluates the metrics.
    ///
    /// # Arguments
    ///
    /// * `y_pred` - The labels predicted by the model.
    /// * `y_true` - The true labels.
    /// * `batch_size` - y_pred and y_true are split in chunks of batch_size to reduce the memory footprint
    ///
    /// # Return value
    ///
    /// Vector containing the values for each metric.
    fn compute_metrics(&self,
                       y_pred: &Tensor,
                       y_true: &Tensor,
                       batch_size: u64,
                       metrics: &Option<Vec<Metrics>>,
    ) -> Vec<PrimitiveType> {
        let num_metrics = match metrics {
            Some(m) => m.len(),
            None => 0
        };

        let mut metrics_values: Vec<PrimitiveType> = vec![0.; num_metrics];

        match metrics {
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
            self.optimizer.update_parameters(&mut **layer, idx);
        }
    }


    /// Computes the output of the network for the given input.
    ///
    /// Multiple samples can be evaluated at once by stacking them along the fourth dimension of the tensor.
    ///
    /// # Return value
    ///
    /// Tensor of the predicted output
    pub fn predict(&self, input: &Tensor) -> Tensor {
        self.forward(&input)
    }

    /// Predicts the class for the input.
    ///
    /// Multiple samples can be evaluated at once by stacking them along the fourth dimension of the tensor.
    ///
    /// # Return value
    ///
    /// Vector of tuples containing the predicted class and the probability for each sample.
    ///
    /// # Panic
    ///
    /// Panics if the model doesn't contain a classes dictionary.
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
            None => panic!("The network does not contain any classes dictionary."),
        }
        predictions
    }


    /// Saves the model in HDF5 format.
    pub fn save(&self, filename: &str) -> Result<(), Error> {

        let file = hdf5::File::create(filename)?;

        let neuro_version: &'static str = env!("CARGO_PKG_VERSION");
        let version = file.new_dataset::<hdf5::types::VarLenUnicode>().create("neuro_version", 1)?;
        version.write(&[hdf5::types::VarLenUnicode::from_str(neuro_version).unwrap()])?;

        let loss = file.new_dataset::<u64>().create("loss", 1)?;
        loss.write(&[self.loss_function.id()])?;
        if let Some(regularizer) = self.regularizer { regularizer.save(&file)?; };
        self.optimizer.save(&file)?;

        let input_shape = file.new_dataset::<[u64; 4]>().create("input_shape", 1)?;
        input_shape.write(&[*self.input_shape.get()])?;

        let output_shape = file.new_dataset::<[u64; 4]>().create("output_shape", 1)?;
        output_shape.write(&[*self.output_shape.get()])?;

        if let Some(classes) = &self.classes {
            let classes_ds = file.new_dataset::<hdf5::types::VarLenUnicode>().create("classes", classes.len())?;
            let mut str = Vec::<hdf5::types::VarLenUnicode>::new();
            for class in classes {
                str.push(hdf5::types::VarLenUnicode::from_str(class).unwrap());
            }
            classes_ds.write(&str[..])?;
        }

        let layers_group = create_group(&file, "layers");
        for (i, layer) in self.layers.iter().enumerate() {
            layer.save(&layers_group, i)?;
        }

        println!("Model saved in: {}", filename);
        Ok(())
    }

    /// Loads a model from a HDF5 file.
    pub fn load(filename: &str) -> Result<Network, Error> {
        let _ = hdf5::silence_errors();
        let file = hdf5::File::open(filename);
        match file {
            Ok(file) => {

                // Shapes
                let input_shape = file.dataset("input_shape").and_then(|shape| shape.read_raw::<[u64; 4]>()).expect("No input shape in the file");
                let output_shape = file.dataset("output_shape").and_then(|shape| shape.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape");

                // Layers
                let mut layers: Vec<Box<dyn Layer>> = Vec::new();
                let layers_group = file.group("layers").expect("Could not retrieve the layers.");
                let layers_name = list_subgroups(&layers_group);
                for layer in &layers_name {
                    let group = layers_group.group(layer).unwrap();
                    let layer_type: Vec<&str> = layer.split('_').collect();

                    match layer_type[1] {
                        BatchNorm::NAME => layers.push(BatchNorm::from_hdf5_group(&group)),
                        Conv2D::NAME => layers.push(Conv2D::from_hdf5_group(&group)),
                        Dense::NAME =>  layers.push(Dense::from_hdf5_group(&group)),
                        Dropout::NAME => layers.push(Dropout::from_hdf5_group(&group)),
                        Flatten::NAME => layers.push(Flatten::from_hdf5_group(&group)),
                        MaxPool2D::NAME => layers.push(MaxPool2D::from_hdf5_group(&group)),
                        _ => panic!("Unknown layer."),
                    }
                }

                // Optimizer
                let optimizer_group = file.group("optimizer").expect("Could not retrieve the optimizer.");
                let opt_type = optimizer_group.dataset("type").and_then(|ds| ds.read_raw::<hdf5::types::VarLenUnicode>()).expect("Could not retrieve the optimizer type.");
                let optimizer: Box<dyn Optimizer> = match opt_type[0].as_str() {
                    Adam::NAME => Adam::from_hdf5_group(&optimizer_group),
                    AdaDelta::NAME => AdaDelta::from_hdf5_group(&optimizer_group),
                    RMSProp::NAME => RMSProp::from_hdf5_group(&optimizer_group),
                    SGD::NAME => SGD::from_hdf5_group(&optimizer_group),
                    _ => panic!("Unknown optimizer."),
                };

                let loss_function_id = file.dataset("loss").and_then(|loss| loss.read_raw::<u64>()).expect("No loss function in the file");
                let loss_function = loss_from_id(loss_function_id[0]);

                let regularizer = Regularizer::from_hdf5_group(&file);

                let classes = if let Ok(classes_group) = file.dataset("classes") {
                    let classes_vec = classes_group
                        .read_raw::<hdf5::types::VarLenUnicode>()
                        .unwrap()
                        .iter()
                        .map(|entry| String::from(entry.as_str()))
                        .collect::<Vec<String>>();
                    Some(classes_vec)
                } else { None };

                Ok(Network {
                    layers,
                    loss_function,
                    optimizer,
                    regularizer,
                    input_shape: Dim::new(&input_shape[0]),
                    output_shape: Dim::new(&output_shape[0]),
                    classes
                })
            },
            Err(err) => Err(Error::from(err)),
        }
    }
}


impl fmt::Display for Network
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "=====")?;
        writeln!(f, "Model")?;
        writeln!(f, "=====")?;
        writeln!(f, "Input shape: [{}, {}, {}]", self.input_shape[0], self.input_shape[1], self.input_shape[2])?;
        writeln!(f, "Output shape: [{}, {}, {}]", self.output_shape[0], self.output_shape[1], self.output_shape[2])?;
        writeln!(f, "Optimizer: {}", self.optimizer.name())?;
        if let Some(regularizer) = self.regularizer { writeln!(f, "Regularizer: {}", regularizer)?; }
        writeln!(f, "\nLayer \t\t Parameters \t Output shape")?;
        writeln!(f, "---------------------------------------------")?;
        for layer in self.layers.iter() {
            writeln!(f, "{}", layer)?;
        }
        Ok(())
    }
}