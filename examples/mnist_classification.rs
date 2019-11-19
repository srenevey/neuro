use neuro::data::{DataSetError, ImageDataSet};
use neuro::models::Network;
use neuro::losses;
use neuro::activations::Activation;
use neuro::metrics::Metrics;
use neuro::optimizers::Adam;
use neuro::layers::Dense;
use neuro::tensor::*;
use neuro::regularizers::Regularizer;

use std::path::Path;

fn main() -> Result<(), DataSetError> {

    // Create the dataset
    let path = Path::new("datasets/MNIST");
    let data = ImageDataSet::from_path(&path, 28, 0.1)?;
    println!("{}", data);

    // Create the neural network
    let mut nn = Network::new(&data, losses::SoftmaxCrossEntropy, Adam::new(0.003), Some(vec![Metrics::Accuracy]), Some(Regularizer::L2(1e-4)));
    nn.add(Dense::new(32, Activation::ReLU));
    nn.add(Dense::new(10, Activation::Softmax));
    println!("{}", nn);

    // Fit the network
    nn.fit(128, 10, Some(1));

    // Predict output on test images
    let input = data.load_img_vec(&vec![
        Path::new("datasets/MNIST/test/2/img_1.jpg"),
        Path::new("datasets/MNIST/test/1/img_18.jpg"),
        Path::new("datasets/MNIST/test/0/img_9.jpg"),
        Path::new("datasets/MNIST/test/3/img_10.jpg")
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