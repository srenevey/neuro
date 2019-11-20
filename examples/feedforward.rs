use neuro::activations::Activation;
use neuro::data::{DataSetError, TabularDataSet};
use neuro::layers::Dense;
use neuro::losses;
use neuro::models::Network;
use neuro::optimizers::Adam;
use neuro::tensor::*;

use std::path::Path;


fn main() -> Result<(), DataSetError> {

    // Load the data
    let inputs = Path::new("datasets/tabular_data/input_normalized.csv");
    let outputs = Path::new("datasets/tabular_data/single_output_normalized.csv");
    let data = TabularDataSet::from_csv(&inputs, &outputs, 0.1, true)?;
    println!("{}", data);

    // Create the network
    let mut nn = Network::new(&data, losses::MeanSquaredError, Adam::new(0.01), None, None);
    nn.add(Dense::new(32, Activation::ReLU));
    nn.add(Dense::new(16, Activation::ReLU));
    nn.add(Dense::new(1, Activation::Linear));

    // Train
    nn.fit(64, 50, Some(10));

    // Predictions: create two inputs: (-0.5, 0.92, 0.35) and (0.45, -0.72, -0.12).
    let inputs = Tensor::new(&[-0.5, 0.92, 0.35, 0.45, -0.72, -0.12], Dim::new(&[3, 1, 1, 2]));
    let res = nn.predict(&inputs); // expected: (-0.957, 1.1314) and (-0.4644, 0.7884) respectively.
    println!("Predictions:");
    res.print_tensor();

    Ok(())
}

