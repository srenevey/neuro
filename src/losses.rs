//! Loss functions.
use arrayfire::*;

use crate::tensor::*;

/// Defines the behaviors of a loss function.
pub trait Loss {
    /// Returns a unique identifier.
    fn id(&self) -> u64;

    /// Computes the value of the loss function from the predicted and true labels.
    fn eval(&self, y_pred: &Tensor, y_true: &Tensor) -> PrimitiveType;

    /// Computes the gradient of the loss function from the predicted and true labels.
    fn grad(&self, y_pred: &Tensor, y_true: &Tensor) -> Tensor;
}

pub(crate) fn loss_from_id(id: u64) -> Box<dyn Loss> {
    match id {
        1 => Box::new(BinaryCrossEntropy),
        2 => Box::new(CrossEntropy),
        3 => Box::new(MeanAbsoluteError),
        4 => Box::new(MeanSquaredError),
        5 => Box::new(SoftmaxCrossEntropy),
        _ => panic!("Unrecognized loss id"),
    }
}

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
        let mut loss = clamp(y_pred, &(1e-15 as PrimitiveType), &((1. - 1e-15) as PrimitiveType), true);
        loss = y_true * log(&loss) + (Tensor::ones(y_true.dims()) - y_true) * log(&sub(&Tensor::ones(loss.dims()), &loss, true));
        -1. / batch_size * sum_all(&loss).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        let ones = Tensor::ones(y_true.dims());
        -  (y_true / y_pred - (&ones - y_true) / (&ones - y_pred))
    }
}

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
        let mut loss = clamp(y_pred, &(1e-15 as PrimitiveType), &((1. - 1e-15) as PrimitiveType), true);
        loss = mul(y_true, &log(&loss), true);
        -1. / batch_size * sum_all(&loss).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        - (y_true / y_pred)
    }
}

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
        let batch_size = y_pred.dims()[3] as PrimitiveType;
        sum_all(&abs(&(y_pred - y_true))).0 as PrimitiveType / batch_size
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        let cond = ge(y_pred, y_true, true);
        selectr(&Tensor::ones(y_pred.dims()), &cond, -1.0f64)
    }
}

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
        1. / batch_size * sum_all(&pow(&(y_pred - y_true), &(2.0 as PrimitiveType), true)).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        (y_pred - y_true) * 2. as PrimitiveType
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
        let batch_size = y_pred.dims().get()[3] as PrimitiveType;
        // Prevent the log to explode by clipping the predicted values
        let mut loss = clamp(y_pred, &(1e-15 as PrimitiveType), &((1. - 1e-15) as PrimitiveType), true);
        loss = y_true * &log(&loss);
         - 1. / batch_size * sum_all(&loss).0 as PrimitiveType
    }

    fn grad(&self,
            y_pred: &Tensor,
            y_true: &Tensor
    ) -> Tensor {
        y_pred - y_true
    }
}



#[cfg(test)]
mod tests {
    use crate::losses::*;
    use crate::tensor::*;
    use crate::assert_approx_eq;
    use arrayfire::*;

    // These tests might fail if PrimitiveType = f32 due to the assertion that the true output and
    // the predicted output must be approximately equal (to 1e-6).

    #[test]
    fn test_mse_eval() {
        let device_id = get_device();
        let loss_fun = MeanSquaredError;

        // 1 sample, 3 outputs
        let mut y_true = Tensor::new(&[2.1, -1.5, 10.9], Dim::new(&[3, 1, 1, 1]));
        let mut y_pred = Tensor::new(&[2.5, 0.1, 11.4], Dim::new(&[3, 1, 1, 1]));
        let loss1 = loss_fun.eval(&y_true, &y_pred);
        let expected_output1: PrimitiveType = 2.97;

        // 2 samples, 2 outputs
        y_true = Tensor::new(&[-16.8, 2.34, -0.2, 31.7], Dim::new(&[2, 1, 1, 2]));
        y_pred = Tensor::new(&[-16.5, -0.9, -3.4, 29.6], Dim::new(&[2, 1, 1, 2]));
        let loss2 = loss_fun.eval(&y_true, &y_pred);
        let expected_output2: PrimitiveType = 12.6188;

        // 3 samples, 4 outputs
        y_true = Tensor::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim::new(&[4, 1, 1, 3]));
        y_pred = Tensor::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim::new(&[4, 1, 1, 3]));
        let loss3 = loss_fun.eval(&y_true, &y_pred);
        let expected_output3: PrimitiveType = 9666.647867;

        sync(device_id);
        assert_approx_eq!([loss1, loss2, loss3], [expected_output1, expected_output2, expected_output3]);
    }

    #[test]
    fn test_mse_grad() {
        let device_id = get_device();
        let loss = MeanSquaredError;
        let y_pred = Tensor::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim::new(&[4, 1, 1, 3]));
        let y_true = Tensor::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim::new(&[4, 1, 1, 3]));
        let grad = loss.grad(&y_pred, &y_true);
        sync(device_id);
        let mut output: [PrimitiveType; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output: [PrimitiveType; 12] = [-10.62, -6.20, -34.96, 336.16, -13.3, 24.42, -2.42, 1.6, -6.6, 9.6, 21.2, 16.];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn test_softmax_cross_entropy_eval() {
        let loss = SoftmaxCrossEntropy;

        // 4 samples, 3 outputs
        let mut y_pred = Tensor::new(&[0.2337, 0.3056, 0.4608, 0.4079, 0.1819, 0.4102, 0.4034, 0.2517, 0.3449, 0.2227, 0.2946, 0.4828], Dim::new(&[3, 1, 1, 4]));
        let mut y_true = Tensor::new(&[0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.], Dim::new(&[3, 1, 1, 4]));
        let output = loss.eval(&y_pred, &y_true);
        let expected_output: PrimitiveType = 0.825470261541275;
        assert_approx_eq!([output], [expected_output]);

    }

    #[test]
    fn test_softmax_cross_entropy_grad() {
        let loss = SoftmaxCrossEntropy;
        let mut y_pred = Tensor::new(&[0.2337, 0.3056, 0.4608, 0.4079, 0.1819, 0.4102, 0.4034, 0.2517, 0.3449, 0.2227, 0.2946, 0.4828], Dim::new(&[3, 1, 1, 4]));
        let mut y_true = Tensor::new(&[0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.], Dim::new(&[3, 1, 1, 4]));
        let grad = loss.grad(&y_pred, &y_true);
        let mut output: [PrimitiveType; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output: [PrimitiveType; 12] = [0.2337, 0.3056, -0.5392, 0.4079, 0.1819, -0.5898, -0.5966, 0.2517, 0.3449, 0.2227, 0.2946, -0.5172];
        assert_approx_eq!(output, expected_output);

    }

    #[test]
    fn test_mae_eval() {
        let device_id = get_device();
        let loss = MeanAbsoluteError;

        // 3 samples, 4 outputs
        let y_pred = Tensor::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim::new(&[4, 1, 1, 3]));
        let y_true = Tensor::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim::new(&[4, 1, 1, 3]));
        let loss_value= loss.eval(&y_pred, &y_true);
        sync(device_id);
        let expected_output: PrimitiveType = 80.51333333;
        assert_approx_eq!([loss_value], [expected_output]);
    }

    #[test]
    fn test_mae_grad() {
        let device_id = get_device();
        let loss = MeanAbsoluteError;

        // 3 samples, 4 outputs
        let y = Tensor::new(&[254.89, 199.9, -4.78, -782.12, 34.65, 12.4, 5.89, -3.2, 78.1, -90.5, -220.6, 136.4], Dim::new(&[4, 1, 1, 3]));
        let y_expected = Tensor::new(&[260.2, 203.0, 12.7, -950.2, 41.3, 0.19, 7.1, -4.0, 81.4, -95.3, -231.2, 128.4], Dim::new(&[4, 1, 1, 3]));
        let grad = loss.grad(&y, &y_expected);
        sync(device_id);
        let mut output: [PrimitiveType; 12] = [0.; 12];
        grad.host(&mut output);
        let expected_output: [PrimitiveType; 12] = [-1., -1., -1., 1., -1., 1., -1., 1., -1., 1., 1., 1.];
        assert_approx_eq!(output, expected_output);
    }
}
