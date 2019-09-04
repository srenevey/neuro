use arrayfire::*;
use crate::assert_approx_eq;


pub enum Loss {
    MeanSquaredError,
    CrossEntropy, // Must be used with a softmax layer
}

impl Loss {
    pub fn eval(&self, y: &Array<f64>, y_expected: &Array<f64>) -> f64 {
        match &self {
            Loss::MeanSquaredError  => {
                let num_samples = y.dims().get()[3];
                let error = sub(y, y_expected, true);
                let sum = sum_all(&(&error * &error));
                sum.0 / num_samples as f64
            },
            Loss::CrossEntropy      => {
                let num_samples = y.dims().get()[3];
                let loss = sum(&sub(&0., &mul(y_expected, &log(y), true), true), 0);
                let sum = sum_all(&loss);
                sum.0 / num_samples as f64
            },
        }
    }

    pub fn grad(&self, y: &Array<f64>, y_expected: &Array<f64>) -> Array<f64> {
        match &self {
            Loss::MeanSquaredError  => {
                let fac = 2. / (y.dims().get()[3] as f64);
                mul(&fac, &sub(y, y_expected, true), true)
            },
            Loss::CrossEntropy      => {
                y - y_expected
            },
        }
    }
}



#[test]
fn test_mse_eval() {
    let loss = Loss::MeanSquaredError;

    // 1 sample, 3 outputs
    let mut y = Array::new(&[2.1, -1.5, 10.9], Dim4::new(&[3, 1, 1, 1]));
    let mut y_expected = Array::new(&[2.5, 0.1, 11.4], Dim4::new(&[3, 1, 1, 1]));
    let output1 = loss.eval(&y, &y_expected);
    let expected_output1 = 2.97;

    // 2 samples, 2 outputs
    y = Array::new(&[-16.8, 2.34, -2.2, 31.7], Dim4::new(&[2, 1, 1, 2]));
    y_expected = Array::new(&[-16.5, -0.9, -3.4, 29.6], Dim4::new(&[2, 1, 1, 2]));
    let output2 = loss.eval(&y, &y_expected);
    let expected_output2 = 8.2188;

    // 3 samples, 4 outputs
    y = Array::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim4::new(&[4, 1, 1, 3]));
    y_expected = Array::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim4::new(&[4, 1, 1, 3]));
    let output3 = loss.eval(&y, &y_expected);
    let expected_output3 = 9666.647867;

    assert_approx_eq!([output1, output2, output3], [expected_output1, expected_output2, expected_output3]);
}