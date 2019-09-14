use arrayfire::*;

pub enum Loss {
    CrossEntropy,
    MeanAbsoluteError,
    MeanSquaredError,
}

impl Loss {
    pub fn eval(&self, y: &Array<f64>, y_expected: &Array<f64>) -> f64 {
        let num_samples = y.dims().get()[3] as f64;
        match &self {
            Loss::CrossEntropy      => {
                - 1. / num_samples * sum_all(&mul(y_expected, &log(y), true)).0 // first element of sum_all corresponds to real part
            },
            Loss::MeanAbsoluteError => {
                1. / num_samples * sum_all(&abs(&sub(y, y_expected, true))).0
            },
            Loss::MeanSquaredError  => {
                1. / num_samples * sum_all(&pow(&sub(y, y_expected, true), &2., false)).0
            }
        }
    }

    pub fn grad(&self, y: &Array<f64>, y_expected: &Array<f64>) -> Array<f64> {
        match &self {
            Loss::CrossEntropy      => {
                y - y_expected
            },
            Loss::MeanAbsoluteError => {
                let cond = ge(y, y_expected, true);
                selectr(&constant(1.0f64, y.dims()), &cond, -1.0f64)
            },
            Loss::MeanSquaredError  => {
                let fac = 2. / (y.dims().get()[3] as f64);
                mul(&fac, &sub(y, y_expected, true), true)
            }
        }
    }
}



#[cfg(test)]
mod tests {
    use crate::losses::Loss;
    use crate::assert_approx_eq;
    use arrayfire::*;

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

    #[test]
    fn test_mse_grad() {
        let loss = Loss::MeanSquaredError;
        let mut y = Array::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim4::new(&[4, 1, 1, 3]));
        let mut y_expected = Array::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim4::new(&[4, 1, 1, 3]));
        let grad = loss.grad(&y, &y_expected);
        let mut output: [f64; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output = [-3.54000000, -2.06666667, -11.65333333, 112.05333333, -4.43333333, 8.14000000, -0.80666667, 0.53333333, -2.20000000, 3.20000000, 7.06666667, 5.33333333];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn test_cross_entropy_eval() {
        let loss = Loss::CrossEntropy;

        // 4 samples, 3 outputs
        let mut y = Array::new(&[0.2337, 0.3056, 0.4608, 0.4079, 0.1819, 0.4102, 0.4034, 0.2517, 0.3449, 0.2227, 0.2946, 0.4828], Dim4::new(&[3, 1, 1, 4]));
        let mut y_expected = Array::new(&[0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.], Dim4::new(&[3, 1, 1, 4]));
        let output1 = loss.eval(&y, &y_expected);
        let expected_output1 = 0.825470261541275;
        assert_approx_eq!([output1], [expected_output1]);

    }

    #[test]
    fn test_cross_entropy_grad() {
        let loss = Loss::CrossEntropy;
        let mut y = Array::new(&[0.2337, 0.3056, 0.4608, 0.4079, 0.1819, 0.4102, 0.4034, 0.2517, 0.3449, 0.2227, 0.2946, 0.4828], Dim4::new(&[3, 1, 1, 4]));
        let mut y_expected = Array::new(&[0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.], Dim4::new(&[3, 1, 1, 4]));
        let grad = loss.grad(&y, &y_expected);
        let mut output: [f64; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output = [0.2337, 0.3056, -0.5392, 0.4079, 0.1819, -0.5898, -0.5966, 0.2517, 0.3449, 0.2227, 0.2946, -0.5172];
        assert_approx_eq!(output, expected_output);

    }

    #[test]
    fn test_mae_eval() {
        let loss = Loss::MeanAbsoluteError;

        // 3 samples, 4 outputs
        let y = Array::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim4::new(&[4, 1, 1, 3]));
        let y_expected = Array::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim4::new(&[4, 1, 1, 3]));
        let output = loss.eval(&y, &y_expected);
        let expected_output = 80.513333333333335;
        assert_approx_eq!([output], [expected_output]);
    }

    #[test]
    fn test_mae_grad() {
        let loss = Loss::MeanAbsoluteError;

        // 3 samples, 4 outputs
        let y = Array::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim4::new(&[4, 1, 1, 3]));
        let y_expected = Array::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim4::new(&[4, 1, 1, 3]));
        let grad = loss.grad(&y, &y_expected);
        let mut output: [f64; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output = [-1., -1., -1., 1., -1., 1., -1., 1., -1., 1., 1., 1.];
        assert_approx_eq!(output, expected_output);
    }
}