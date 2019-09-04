use std::path::Path;
use arrayfire::*;

use neuro::layers::dense::*;
use neuro::activations::Activation;
use neuro::models::*;
use neuro::losses::Loss;
use neuro::optimizers::*;
use neuro::layers::batchnormalization::BatchNormalization;
use neuro::data::*;


fn main() -> Result<(), DataSetError> {

    /*
    // Input
    let in_values: [f64; 6] = [1., -1., 1.5, -0.2, 2.1, 1.2];
    let x = Array::new(&in_values, Dim4::new(&[2, 1, 1, 3])); // 3 examples

    // Approximate y = [x1*x1 + x1*x2, x2*x2 - x1*x2]
    let out_values: [f64; 6] = [0., 0., 1.95, 0.34, 6.93, -1.08];
    let y = Array::new(&out_values, Dim4::new(&[2, 1, 1, 3]));

    let data = DataSet::new(x, y, 0.2);

    // Create the network
    let input_shape: (u64, u64, u64) = (2, 1, 1);
    let mut nn = Network::new(input_shape, Loss::MeanSquaredError, Adam::new(0.03));
    nn.add(Dense::new(3, Activation::ReLU));
    nn.add(Dense::new(4, Activation::ReLU));
    nn.add(Dense::new(2, Activation::Linear));

    // Test
    nn.fit(&data, 1, 20);

    //let y_approx = nn.evaluate(&x);
    //af_print!("y_approx", y_approx);
    */

    let inputs = Path::new("datasets/inputs.csv");
    let outputs = Path::new("datasets/outputs.csv");

    let mut data = TabularDataSet::from_csv(&inputs, &outputs, 0.2)?;
    let (mb_x, mb_y) = data.mini_batch.next();

    //let mut data = DataSet::from_csv(&inputs, &outputs, 0.2)?;


    //data.normalize();

    //data.batch_iterator(10);

    // Create the network
    /*
    let mut nn = Network::new(&data, Loss::MeanSquaredError, Adam::new(0.03));
    nn.add(Dense::new(3, Activation::ReLU));
    nn.add(BatchNormalization::new());
    nn.add(Dense::new(5, Activation::ReLU));
    //nn.add(BatchNormalization::new());
    nn.add(Dense::new(2, Activation::Linear));

    // Test
    nn.fit(1, 60);

    nn.evaluate();
    */


    Ok(())
}