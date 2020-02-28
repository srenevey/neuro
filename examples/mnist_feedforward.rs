use neuro::activations::Activation;
use neuro::data::ImageDataSetBuilder;
use neuro::errors::*;
use neuro::layers::{Dense, Flatten};
use neuro::losses;
use neuro::metrics::Metrics;
use neuro::models::Network;
use neuro::optimizers::Adam;
use neuro::regularizers::Regularizer;
use neuro::tensor::*;

use std::path::Path;

fn main() -> Result<(), Error> {

    // Create the dataset
    let path = Path::new("datasets/MNIST");
    let data = ImageDataSetBuilder::from_dir(&path, (28, 28))
        .one_hot_encode()
        .valid_split(0.1)
        .scale(1./255.)
        .build()?;
    println!("{}", data);

    // Create the neural network
    let mut nn = Network::new(Dim::new(&[28, 28, 1, 1]), losses::SoftmaxCrossEntropy, Adam::new(0.003), Some(Regularizer::L2(1e-3)))?;
    nn.add(Flatten::new());
    nn.add(Dense::new(32, Activation::ReLU));
    nn.add(Dense::new(10, Activation::Softmax));
    println!("{}", nn);

    // Fit the network
    nn.fit(&data, 128, 10, Some(1), Some(vec![Metrics::Accuracy]));

    // Evaluate the trained model on the test set
    nn.evaluate(&data, Some(vec![Metrics::Accuracy]));


    // Predict the output of some images from the test set
    let input = data.load_image_vec(&vec![
        Path::new("datasets/MNIST/test/1/5.png"),
        Path::new("datasets/MNIST/test/3/2008.png"),
        Path::new("datasets/MNIST/test/5/59.png"),
        Path::new("datasets/MNIST/test/9/104.png")
    ])?;

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