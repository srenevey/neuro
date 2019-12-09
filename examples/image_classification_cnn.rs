use neuro::activations::Activation;
use neuro::data::ImageDataSet;
use neuro::errors::*;
use neuro::layers::{Dense, Conv2D, Padding, MaxPooling2D, Dropout};
use neuro::losses;
use neuro::metrics::Metrics;
use neuro::models::Network;
use neuro::optimizers::Adam;
use neuro::regularizers::*;

use std::path::Path;

fn main() -> Result<(), NeuroError> {

    // Load the data
    let path = Path::new("datasets/MNIST");
    let data = ImageDataSet::from_path(&path, 28, 0.1)?;
    println!("{}", data);

    // Create the neural network
    let mut nn = Network::new(&data, losses::SoftmaxCrossEntropy, Adam::new(0.001), None);
    nn.add(Conv2D::new(32, (3, 3), (1, 1), Padding::Same));
    nn.add(Conv2D::new(64, (3, 3), (1, 1), Padding::Same));
    nn.add(MaxPooling2D::new((2, 2), (2, 2)));
    nn.add(Dropout::new(0.5));
    nn.add(Dense::new(128, Activation::ReLU));
    nn.add(Dropout::new(0.25));
    nn.add(Dense::new(10, Activation::Softmax));

    println!("{}", nn);

    // Fit the model
    nn.fit(128, 10, Some(1), Some(vec![Metrics::Accuracy]));

    // Evaluate the trained model on the test set
    nn.evaluate(Some(vec![Metrics::Accuracy]));

    Ok(())
}
