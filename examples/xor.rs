use neuro::activations::Activation;
use neuro::data::TabularDataSet;
use neuro::errors::*;
use neuro::initializers::*;
use neuro::layers::Dense;
use neuro::losses;
use neuro::metrics;
use neuro::models;
use neuro::optimizers::SGD;
use neuro::tensor::*;

fn main() -> Result<(), NeuroError> {
    // Create the dataset
    let input_values = [0., 0., 0., 1., 1., 0., 1., 1.];
    let x_train = Tensor::new(&input_values, Dim::new(&[2, 1, 1, 4]));
    let output_values = [0., 1., 1., 0.];
    let y_train = Tensor::new(&output_values, Dim::new(&[1, 1, 1, 4]));
    let data = TabularDataSet::from_tensor(x_train.copy(), y_train.copy(), x_train.copy(), y_train.copy(), None, None)?;

    // Create the neural network and add two layers
    let mut nn = models::Network::new(&data, losses::BinaryCrossEntropy, SGD::new(0.3), None);
    nn.add(Dense::with_param(2, Activation::ReLU, Initializer::Constant(0.01), Initializer::Zeros));
    nn.add(Dense::new(1, Activation::Sigmoid));

    // Fit the model
    nn.fit(4, 2000, Some(200), Some(vec![metrics::Metrics::Accuracy]));

    // Compute the output for the training data
    let predictions = nn.predict(&x_train);
    println!("Predictions:");
    Tensor::print_tensor(&predictions);

    Ok(())
}