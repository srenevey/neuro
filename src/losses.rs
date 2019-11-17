//! Loss functions.
use arrayfire::*;
use crate::Tensor;
use crate::tensor::*;

/// Defines the behaviors of a loss function.
pub trait Loss {
    /// Returns a unique identifier.
    fn id(&self) -> u64;

    /// Computes the value of the loss function.
    ///
    /// # Arguments
    /// * `y_pred`: predicted labels
    /// * `y_true`: true labels
    ///
    fn eval(&self, y_pred: &Tensor, y_true: &Tensor) -> PrimitiveType;

    /// Computes the gradient of the loss function.
    ///
    /// # Arguments
    /// * `y_pred`: predicted labels
    /// * `y_true`: true labels
    ///
    fn grad(&self, y_pred: &Tensor, y_true: &Tensor) -> Tensor;
}

/// Defines the binary cross entropy loss.
#[derive(Debug, Copy, Clone)]
pub struct BinaryCrossEntropy;

impl Loss for BinaryCrossEntropy {
    fn id(&self) -> u64 {
        1
    }

    fn eval(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> PrimitiveType {
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        // Prevent the log to explode by clipping the predicted values
        let mut loss = clamp(y_pred, &(1e-8 as PrimitiveType), &((1. - 1e-8) as PrimitiveType), true);
        loss = y_true * log(&loss) + (Tensor::ones(y_true.dims()) - y_true) * log(&sub(&Tensor::ones(loss.dims()), &loss, true));
        -1. / batch_size * sum_all(&loss).0 as PrimitiveType
        //*loss = sum(&sum(&sum(&sum(&loss, 3), 2), 1), 0);
        //*loss = mul(loss, &(-1. / batch_size), true);
        //-1. / batch_size * sum_all(&(y_true * log(&output) + (Tensor::ones(y_true.dims()) - y_true) * log(&(Tensor::ones(output.dims()) - output)))).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        let ones = Tensor::ones(y_true.dims());
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        let factor = 1. / batch_size;
        -  (y_true / y_pred - (&ones - y_true) / (&ones - y_pred)) * factor
    }
}

/// Defines the cross entropy loss.
#[derive(Debug, Copy, Clone)]
pub struct CrossEntropy;

impl Loss for CrossEntropy {
    fn id(&self) -> u64 {
        2
    }

    fn eval(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> PrimitiveType {
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        // Prevent the log to explode by clipping the predicted values
        let mut loss = clamp(y_pred, &(1e-8 as PrimitiveType), &((1. - 1e-8) as PrimitiveType), true);
        loss = mul(y_true, &log(&loss), true);
        -1. / batch_size * sum_all(&loss).0 as PrimitiveType
        //*loss = sum(&sum(&sum(&sum(loss, 3), 2), 1), 0);
        //*loss = mul(loss, &(-1. / batch_size), true);
        //- 1. / batch_size * sum_all(&mul(y_true, &log(&output), true)).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        let factor = 1. / batch_size;
        - (y_true / y_pred) * factor
    }
}

/// Defines the mean absolute error loss (MAE).
#[derive(Debug, Copy, Clone)]
pub struct MeanAbsoluteError;

impl Loss for MeanAbsoluteError {
    fn id(&self) -> u64 {
        3
    }

    fn eval(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> PrimitiveType {
        //*loss = abs(&sub(y_pred, y_true, true));
        //*loss = mean(&mean(&mean(&mean(loss, 3), 2), 1), 0);
        mean_all(&abs(&sub(y_pred, y_true, true))).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        let diff = y_pred - y_true;
        let cond = ge(y_pred, y_true, true);
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        let factor = 1. / batch_size;
        mul(&constant(factor, y_pred.dims()), &selectr(&Tensor::ones(y_pred.dims()), &cond, -1.0f64), true)
    }
}

/// Defines the mean squared error loss (MSE).
#[derive(Debug, Copy, Clone)]
pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn id(&self) -> u64 {
        4
    }

    fn eval(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> PrimitiveType {
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        //*loss = pow(&(y_pred - y_true), &(2.0 as PrimitiveType), true);
        //*loss = sum(&sum(&sum(&sum(loss, 3), 2), 1), 0);
        //*loss = mul(loss, &(1. / batch_size), true);
        1. / batch_size * sum_all(&pow(&sub(y_pred, y_true, true), &2.0f64, true)).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        let factor = 2. / batch_size;
        (y_pred - y_true) * factor
    }
}


/// Applies the softmax function on the input and then computes the cross entropy loss.
#[derive(Debug, Copy, Clone)]
pub struct SoftmaxCrossEntropy;

impl Loss for SoftmaxCrossEntropy {
    fn id(&self) -> u64 {
        5
    }

    fn eval(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> PrimitiveType {
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        // Prevent the log to explode by clipping the predicted values
        let mut loss = clamp(y_pred, &(1e-8 as PrimitiveType), &((1. - 1e-8) as PrimitiveType), true);
        loss = y_true * &log(&loss);
        //*loss = sum(&sum(&sum(&sum(loss, 3), 2), 1), 0);
        //*loss = mul(loss, &(- 1. / batch_size), true);
        - 1. / batch_size * sum_all(&loss).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        let factor = 1. / batch_size;
        (y_pred - y_true) * factor
    }
}




#[cfg(test)]
mod tests {
    use crate::losses::*;
    use crate::Tensor;
    use crate::tensor::*;
    use crate::assert_approx_eq;

    #[test]
    fn test_mse_eval() {
        let loss_fun = MeanSquaredError;

        // 1 sample, 3 outputs
        let mut y_true = Tensor::new(&[2.1, -1.5, 10.9], Dim::new(&[3, 1, 1, 1]));
        let mut y_pred = Tensor::new(&[2.5, 0.1, 11.4], Dim::new(&[3, 1, 1, 1]));
        //let mut loss_value1 = Tensor::new(&[0.0], Dim::new(&[1, 1, 1, 1]));
        let loss1 = loss_fun.eval(&y_true, &y_pred);
        //let loss1 = loss_value1.get_scalar();
        let expected_output1 = 2.97;

        // 2 samples, 2 outputs
        y_true = Tensor::new(&[-16.8, 2.34, -0.2, 31.7], Dim::new(&[2, 1, 1, 2]));
        y_pred = Tensor::new(&[-16.5, -0.9, -3.4, 29.6], Dim::new(&[2, 1, 1, 2]));
        //let mut loss_value2 = Tensor::new(&[0.0], Dim::new(&[1, 1, 1, 1]));
        let loss2 = loss_fun.eval(&y_true, &y_pred);
        //let loss2 = loss_value2.get_scalar();
        let expected_output2 = 12.6188 as PrimitiveType;

        // 3 samples, 4 outputs
        y_true = Tensor::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim::new(&[4, 1, 1, 3]));
        y_pred = Tensor::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim::new(&[4, 1, 1, 3]));
        //let mut loss_value3 = Tensor::new(&[0.0], Dim::new(&[1, 1, 1, 1]));
        let loss3 = loss_fun.eval(&y_true, &y_pred);
        //let loss3 = loss_value3.get_scalar();
        let expected_output3 = 9666.647867 as PrimitiveType;

        assert_approx_eq!([loss1, loss2, loss3], [expected_output1, expected_output2, expected_output3]);
    }

    #[test]
    fn test_mse_grad() {
        let loss = MeanSquaredError;
        let mut y = Tensor::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim::new(&[4, 1, 1, 3]));
        let mut y_expected = Tensor::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim::new(&[4, 1, 1, 3]));
        let grad = loss.grad(&y, &y_expected);
        let mut output: [f64; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output = [-3.54000000, -2.06666667, -11.65333333, 112.05333333, -4.43333333, 8.14000000, -0.80666667, 0.53333333, -2.20000000, 3.20000000, 7.06666667, 5.33333333];
        assert_approx_eq!(output, expected_output);
    }

    /*
    #[test]
    fn test_cross_entropy_eval() {
        let loss = CrossEntropy;

        // 4 samples, 3 outputs
        let mut y = Tensor::new(&[0.2337, 0.3056, 0.4608, 0.4079, 0.1819, 0.4102, 0.4034, 0.2517, 0.3449, 0.2227, 0.2946, 0.4828], Dim::new(&[3, 1, 1, 4]));
        let mut y_expected = Tensor::new(&[0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.], Dim::new(&[3, 1, 1, 4]));
        let output1 = loss.eval(&y, &y_expected);
        let expected_output1 = 0.825470261541275;
        assert_approx_eq!([output1], [expected_output1]);

    }

    #[test]
    fn test_cross_entropy_grad() {
        let loss = CrossEntropy;
        let mut y = Tensor::new(&[0.2337, 0.3056, 0.4608, 0.4079, 0.1819, 0.4102, 0.4034, 0.2517, 0.3449, 0.2227, 0.2946, 0.4828], Dim::new(&[3, 1, 1, 4]));
        let mut y_expected = Tensor::new(&[0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.], Dim::new(&[3, 1, 1, 4]));
        let grad = loss.grad(&y, &y_expected);
        let mut output: [f64; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output = [0.2337, 0.3056, -0.5392, 0.4079, 0.1819, -0.5898, -0.5966, 0.2517, 0.3449, 0.2227, 0.2946, -0.5172];
        assert_approx_eq!(output, expected_output);

    }
    */

    #[test]
    fn test_mae_eval() {
        let loss = MeanAbsoluteError;

        // 3 samples, 4 outputs
        let y = Tensor::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim::new(&[4, 1, 1, 3]));
        let y_expected = Tensor::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim::new(&[4, 1, 1, 3]));
        //let mut loss_value = Tensor::new(&[0.0], Dim::new(&[1, 1, 1, 1]));
        let loss_value= loss.eval(&y, &y_expected);
        //let loss = loss_value.get_scalar();
        let expected_output = 80.513333333333335;
        assert_approx_eq!([loss_value], [expected_output]);
    }

    #[test]
    fn test_mae_grad() {
        let loss = MeanAbsoluteError;

        // 3 samples, 4 outputs
        let y = Tensor::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim::new(&[4, 1, 1, 3]));
        let y_expected = Tensor::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim::new(&[4, 1, 1, 3]));
        let grad = loss.grad(&y, &y_expected);
        let mut output: [f64; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output = [-1., -1., -1., 1., -1., 1., -1., 1., -1., 1., 1., 1.];
        assert_approx_eq!(output, expected_output);
    }
}
