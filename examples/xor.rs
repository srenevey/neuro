use neuro::activations::Activation;
use neuro::data::{DataSetError, TabularDataSet};
use neuro::layers::{Dense, Initializer};
use neuro::losses;
use neuro::metrics;
use neuro::models;
use neuro::optimizers::SGD;
use neuro::Tensor;
use neuro::tensor::Dim;
use neuro::tensor::TensorTrait;

fn main() -> Result<(), DataSetError> {
    // Create the dataset
    let input_values = [0., 0., 0., 1., 1., 0., 1., 1.];
    let x_train = Tensor::new(&input_values, Dim::new(&[2, 1, 1, 4]));
    let output_values = [0., 1., 1., 0.];
    let y_train = Tensor::new(&output_values, Dim::new(&[1, 1, 1, 4]));
    let data = TabularDataSet::from_tensor(x_train.copy(), y_train.copy(), x_train.copy(), y_train.copy(), None, None)?;

    // Create the neural network and add two layers
    let mut nn = models::Network::new(&data, losses::BinaryCrossEntropy, SGD::new(0.3), Some(vec![metrics::Metrics::Accuracy]), None);
    nn.add(Dense::with_param(2, Activation::ReLU, Initializer::RandomUniform, Initializer::Zeros));
    nn.add(Dense::new(1, Activation::Sigmoid));

    // Fit the model
    nn.fit(4, 500, Some(100));


    // Compute the output for the train data
    //let predictions = nn.predict(&x_train);
    //af_print!("predictions: ", predictions);



    let test1 = Tensor::new(&[0., 0.], Dim::new(&[2, 1, 1, 1]));
    let pred1 = nn.predict(&test1);
    println!("prediction for [0, 0]:");
    pred1.print_tensor();
    let test2 = Tensor::new(&[0., 1.], Dim::new(&[2, 1, 1, 1]));
    let pred2 = nn.predict(&test2);
    println!("prediction for [0, 1]:");
    Tensor::print_tensor(&pred2);
    let test3 = Tensor::new(&[1., 0.], Dim::new(&[2, 1, 1, 1]));
    let pred3 = nn.predict(&test3);
    println!("prediction for [1, 0]:");
    Tensor::print_tensor(&pred3);
    let test4 = Tensor::new(&[1., 1.], Dim::new(&[2, 1, 1, 1]));
    let pred4 = nn.predict(&test4);
    println!("prediction for [1, 1]:");
    Tensor::print_tensor(&pred4);



    Ok(())
}