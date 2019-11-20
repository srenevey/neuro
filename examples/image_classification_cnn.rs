// CURRENTLY BROKEN
use neuro::activations::Activation;
use neuro::data::{DataSetError, ImageDataSet};
use neuro::layers::{Dense, Conv2D, Padding, MaxPooling2D};
use neuro::losses;
use neuro::metrics::Metrics;
use neuro::models::Network;
use neuro::optimizers::SGD;

use std::path::Path;


fn main() -> Result<(), DataSetError> {

    // Load the data
    let path = Path::new("datasets/MNIST");
    let data = ImageDataSet::from_path(&path, 28, 0.01)?;
    println!("{}", data);

    // Create the neural network
    let mut nn = Network::new(&data, losses::SoftmaxCrossEntropy, SGD::new(0.001), Some(vec![Metrics::Accuracy]), None);
    nn.add(Conv2D::new(32, (5, 5), (1, 1), Padding::Same));
    nn.add(Conv2D::new(64, (3, 3), (1, 1), Padding::Same));
    nn.add(MaxPooling2D::new((2, 2), (2, 2)));
    nn.add(Dense::new(32, Activation::ReLU));
    nn.add(Dense::new(10, Activation::Softmax));

    println!("{}", nn);

    // Fit the model
    nn.fit(32, 5, Some(1));

    Ok(())
}
