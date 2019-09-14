use std::path::Path;

use neuro::layers::{Dense, BatchNormalization};
use neuro::activations::Activation;
use neuro::models::Network;
use neuro::losses::Loss;
use neuro::optimizers::Adam;
use neuro::data::{DataSetError, TabularDataSet};


fn main() -> Result<(), DataSetError> {

    let inputs = Path::new("datasets/inputs.csv");
    let outputs = Path::new("datasets/outputs.csv");

    let mut data = TabularDataSet::from_csv(&inputs, &outputs, 0.2)?;
    data.normalize_output();

    // Create the network
    let mut nn = Network::new(&mut data, Loss::MeanSquaredError, Adam::new(0.03));
    nn.add(BatchNormalization::new());
    nn.add(Dense::new(4, Activation::ReLU));
    nn.add(BatchNormalization::new());
    nn.add(Dense::new(5, Activation::ReLU));
    nn.add(BatchNormalization::new());
    nn.add(Dense::new(2, Activation::Linear));

    // Train
    nn.fit(32, 100);

    // Test
    nn.evaluate();


    Ok(())
}



