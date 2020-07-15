// This example reproduces the network described in Zeiler, M.D., Adadelta: An Adaptive Learning Rate Method, arXiv:1212.5701v1, 2012.
// An accuracy of about 97.4% is achieved on the test set.

use neuro::activations::Activation;
use neuro::data::{ImageDataSetBuilder, ImageDataSet};
use neuro::errors::*;
use neuro::layers::{Dense, Flatten};
use neuro::losses;
use neuro::metrics::Metrics;
use neuro::models::Network;
use neuro::optimizers::AdaDelta;
use neuro::tensor::*;

use std::path::Path;
use neuro::initializers::Initializer;

fn main() -> Result<(), Error> {

    // Create the dataset
    let path = Path::new("datasets/MNIST");
    let data = ImageDataSetBuilder::from_dir(&path, (28, 28))
        .valid_split(0.1)
        .one_hot_encode()
        .scale(1./255.)
        .build()?;
    println!("{}", data);

    // Create the neural network
    let mut nn = Network::new(Dim::new(&[28, 28, 1, 1]), losses::SoftmaxCrossEntropy::new(), AdaDelta::new(), None)?;
    nn.add(Flatten::new());
    nn.add(Dense::new(500, Activation::Tanh));
    nn.add(Dense::new(300, Activation::Tanh));
    nn.add(Dense::new(10, Activation::Softmax));
    println!("{}", nn);


    // Fit the network
    nn.fit(&data, 100, 6, Some(1), Some(vec![Metrics::Accuracy]));

    // Evaluate the trained model on the test set
    nn.evaluate(&data, Some(vec![Metrics::Accuracy]));


    // Predict the output of some images from the test set
    let input = ImageDataSet::load_image_vec(&vec![
        Path::new("datasets/MNIST/test/1/5.png"),
        Path::new("datasets/MNIST/test/3/2008.png"),
        Path::new("datasets/MNIST/test/5/59.png"),
        Path::new("datasets/MNIST/test/9/104.png")
    ], (28, 28), data.image_ops())?;

    let predictions = nn.predict_class(&input);
    print_prediction(&predictions);

    Ok(())
}

fn print_prediction(predictions: &Vec<(String, PrimitiveType)>) {
    println!("Predictions:");
    let mut index = 0;
    for (class, probability) in predictions {
        index += 1;
        println!("image {}: class: {}, probability: {}", index, class, probability);
    }
}