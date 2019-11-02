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

    let path = Path::new("datasets/MNIST");
    let data = ImageDataSet::from_path(&path, 28, 0.2)?;
    println!("{}", data);
    data.print_classes();


    let mut nn = Network::new(&data, losses::SoftmaxCrossEntropy, Adam::new(0.03), Some(vec![Metrics::Accuracy]), None);
    nn.add(Dense::new(32, Activation::ReLU));
    nn.add(Dense::new(10, Activation::Softmax));

    println!("{}", nn);

    nn.fit(128, 50, Some(1));


    // Predict output on test images
    let img1 = data.load_img(Path::new("datasets/MNIST/test/2/img_1.jpg"))?;
    let img2 = data.load_img(Path::new("datasets/MNIST/test/1/img_18.jpg"))?;
    let img3 = data.load_img(Path::new("datasets/MNIST/test/0/img_4.jpg"))?;
    let img4 = data.load_img(Path::new("datasets/MNIST/test/3/img_10.jpg"))?;

    let pred1 = nn.predict(&img1);
    let pred2 = nn.predict(&img2);
    let pred3 = nn.predict(&img3);
    let pred4 = nn.predict(&img4);

    println!("prediction for '2':");
    pred1.print_tensor();

    println!("prediction for '1':");
    pred2.print_tensor();

    println!("prediction for '0':");
    pred3.print_tensor();

    println!("prediction for '3':");
    pred4.print_tensor();


    Ok(())
}