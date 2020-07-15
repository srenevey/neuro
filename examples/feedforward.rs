use neuro::activations::Activation;
use neuro::data::{TabularDataSet, DataSet};
use neuro::errors::*;
use neuro::layers::Dense;
use neuro::losses;
use neuro::models::Network;
use neuro::optimizers::Adam;
use neuro::tensor::*;

use std::path::Path;


fn main() -> Result<(), Error> {

    // Load the data
    let inputs = Path::new("datasets/tabular_data/input_normalized.csv");
    let outputs = Path::new("datasets/tabular_data/single_output_normalized.csv");
    let data = TabularDataSet::from_csv(&inputs, &outputs, 0.1, true)?;
    println!("{}", data);

    // Create the network
    let mut nn = Network::new(data.input_shape(), losses::MeanSquaredError::new(), Adam::new(0.01), None)?;
    nn.add(Dense::new(32, Activation::ReLU));
    nn.add(Dense::new(16, Activation::ReLU));
    nn.add(Dense::new(1, Activation::Linear));
    println!("{}", nn);

    // Train and save the model
    nn.fit(&data, 64, 50, Some(10), None);
    nn.save("feedforward.h5")?;

    // Predictions: create two inputs: (-0.5, 0.92, 0.35) and (0.45, -0.72, -0.12).
    let inputs = Tensor::new(&[-0.5, 0.92, 0.35, 0.45, -0.72, -0.12], Dim::new(&[3, 1, 1, 2]));
    let res = nn.predict(&inputs); // expected: -0.957 and -0.4644 respectively.
    println!("Predictions:");
    res.print_tensor();

    Ok(())
}

