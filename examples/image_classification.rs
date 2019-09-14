use std::path::Path;
use arrayfire::*;

use neuro::data::{DataSetError, ImageDataSet};
use neuro::models::Network;
use neuro::losses::Loss;
use neuro::activations::Activation;
use neuro::optimizers::Adam;
use neuro::layers::{Dense, Conv2D};


fn main() -> Result<(), DataSetError> {

    let path = Path::new("datasets/MNIST");
    let data = ImageDataSet::from_path(&path, 28)?;
    data.print_stats();

    //let mut nn = Network::new(&data, Loss::CrossEntropy, Adam::new(0.03));
    //nn.add(Conv2D::new(Activation::ReLU, 3, ConvMode::DEFAULT));
    //nn.add(Dense::new(2, Activation::Softmax));

    Ok(())
}
