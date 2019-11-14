use std::path::Path;

use neuro::data::{DataSetError, ImageDataSet, DataSet};
use neuro::models::Network;
use neuro::losses;
use neuro::activations::Activation;
use neuro::metrics::Metrics;
use neuro::optimizers::{Adam, SGD};
use neuro::regularizers::*;
use neuro::layers::{Dense, BatchNormalization, Dropout, MaxPooling2D};
use neuro::tensor::*;


fn main() -> Result<(), DataSetError> {

    // Create the dataset
    let path = Path::new("datasets/MNIST");
    let data = ImageDataSet::from_path(&path, 28, 0.2)?;
    println!("{}", data);

    let mut nn = Network::new(&data, losses::SoftmaxCrossEntropy, Adam::new(0.03), Some(vec![Metrics::Accuracy]), None);
    nn.add(Dense::new(32, Activation::ReLU));
    nn.add(Dense::new(10, Activation::Softmax));

    println!("{}", nn);

    nn.fit(128, 2, Some(1));


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