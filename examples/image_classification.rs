use std::path::Path;

use neuro::data::{DataSetError, ImageDataSet};
use neuro::models::Network;
use neuro::losses::Loss;
use neuro::activations::Activation;
use neuro::optimizers::Adam;
use neuro::layers::{Dense, Conv2D, ConvMode};


fn main() -> Result<(), DataSetError> {

    let path = Path::new("datasets/MNIST");
    let mut data = ImageDataSet::from_path(&path, 28)?;
    data.print_stats();

    let mut nn = Network::new(&mut data, Loss::CrossEntropy, Adam::new(0.03));
    nn.add(Conv2D::new(2, (32, 32), ConvMode::Same, Activation::Linear));

    Ok(())
}
