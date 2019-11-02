use std::path::Path;

use neuro::data::{DataSetError, ImageDataSet, DataSet};
use neuro::models::Network;
use neuro::losses::Loss;
use neuro::activations::Activation;
use neuro::optimizers::Adam;
use neuro::layers::{Dense, Conv2D, Padding, MaxPooling2D};

use arrayfire::*;

fn main() -> Result<(), DataSetError> {

    let path = Path::new("datasets/MNIST");
    let mut data = ImageDataSet::from_path(&path, 28)?;
    data.print_stats();

    let test = randu::<f64>(Dim4::new(&[3, 10, 1, 1]));
    af_print!("test", test);
    let tmp = gt(&test, &0.7, true);
    let res = mul(&test, &tmp, true);
    af_print!("res", res);

    println!("passed");
    //let test2 = randn::<f64>(Dim4::new(&[10976000, 1, 1, 1]));



    /*
    let mut nn = Network::new(&mut data, Loss::CrossEntropy, Adam::new(0.001));
    nn.add(Conv2D::new(28, (3, 3), (1, 1), Padding::Same));
    nn.add(MaxPooling2D::new((2, 2)));
    nn.add(Dense::new(128, Activation::ReLU));
    nn.add(Dense::new(10, Activation::Softmax));

    nn.fit(32, 10);
*/


    Ok(())
}
