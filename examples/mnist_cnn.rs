use neuro::activations::Activation;
use neuro::data::ImageDataSetBuilder;
use neuro::errors::*;
use neuro::layers::{Dense, Conv2D, Padding, MaxPool2D, Dropout, Flatten};
use neuro::losses;
use neuro::metrics::Metrics;
use neuro::models::Network;
use neuro::optimizers::Adam;
use neuro::tensor::*;

use std::path::Path;

fn main() -> Result<(), Error> {

    // Load and preprocess the data
    let path = Path::new("datasets/MNIST");
    let data = ImageDataSetBuilder::from_dir(&path, (28, 28))
        .one_hot_encode()
        .valid_split(0.1)
        .scale(1./255.)
        .build()?;
    println!("{}", data);

    // Create the neural network
    let mut nn = Network::new(Dim::new(&[28, 28, 1, 1]), losses::SoftmaxCrossEntropy, Adam::new(0.001), None)?;
    nn.add(Conv2D::new(32, (3, 3), (1, 1), Padding::Same));
    nn.add(Conv2D::new(64, (3, 3), (1, 1), Padding::Same));
    nn.add(MaxPool2D::new((2, 2)));
    nn.add(Dropout::new(0.5));
    nn.add(Flatten::new());
    nn.add(Dense::new(128, Activation::ReLU));
    nn.add(Dropout::new(0.25));
    nn.add(Dense::new(10, Activation::Softmax));
    println!("{}", nn);

    // Fit the model
    nn.fit(&data, 128, 10, Some(1), Some(vec![Metrics::Accuracy]));
    nn.save("mnist_cnn.h5")?;

    // Evaluate the trained model on the test set
    nn.evaluate(&data, Some(vec![Metrics::Accuracy]));

    Ok(())
}
