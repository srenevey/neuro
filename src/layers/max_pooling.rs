//! # MaxPooling2D
//!
//! This layer performs max pooling operation on 2D arrays. The pooling is defined by a kernel size and stride.

use crate::layers::Layer;
use arrayfire::*;

pub struct MaxPooling2D {
    kernel_size: (u64, u64),
    stride: (u64, u64),
    output_shape: Dim4,
    grad: Array<f64>,
}

impl MaxPooling2D {
    /// Create a max pooling 2D layer.
    ///
    /// # Arguments
    /// * `kernel_size`: tuple containing the height and width of the kernel
    ///
    /// # Notes
    /// By default the stride is set to (1, 1).
    pub fn new(kernel_size: (u64, u64)) -> Box<MaxPooling2D> {
        Box::new(MaxPooling2D {
            kernel_size,
            stride: (1, 1),
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            grad: Array::new_empty(Dim4::new(&[0, 0, 0, 0]))
        })
    }

    /// Create a max pooling 2D layer with the specified parameters.
    ///
    /// # Arguments
    /// * `kernel_size` - tuple containing the height and width of the kernel
    /// * `stride` - tuple containing the vertical and horizontal stride
    ///
    /// # Notes
    /// The dimension of the input minus the kernel size must be divisible by the stride.
    pub fn with_param(kernel_size: (u64, u64), stride: (u64, u64)) -> Box<MaxPooling2D> {
        Box::new(MaxPooling2D {
            kernel_size,
            stride,
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            grad: Array::new_empty(Dim4::new(&[0, 0, 0, 0]))
        })
    }
}

impl Layer for MaxPooling2D {
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let output_height = (input_shape.get()[0] - self.kernel_size.0) / self.stride.0 + 1;
        let output_width = (input_shape.get()[1] - self.kernel_size.1) / self.stride.0 + 1;
        self.output_shape = Dim4::new(&[output_height, output_width, input_shape.get()[2], input_shape.get()[3]]);
    }

    fn compute_activation(&self, input: &Array<f64>) -> Array<f64> {
        let height = input.dims().get()[0];
        let width = input.dims().get()[1];
        let n_channels = input.dims().get()[2];
        let mb_size = input.dims().get()[3];

        // Check if values work
        if (height - self.kernel_size.0) % self.stride.0 != 0 || (width - self.kernel_size.1) % self.stride.1 != 0 {
            panic!("The input dimensions, kernel size and stride are incompatible. (dimension - kernel_size) / stride must be an integer.");
        }

        let i_max = (height - self.kernel_size.0) / self.stride.0 + 1;
        let j_max = (width - self.kernel_size.1) / self.stride.1 + 1;

        let mut max_values = Array::new_empty(Dim4::new(&[1, 1, n_channels, mb_size]));

        // Generate indices to select values in the filtering window
        for i in 0..i_max {
            let lb_i = i * self.stride.0;
            let ub_i = lb_i + self.kernel_size.0 - 1;
            let seq_i = Seq::new(lb_i as f32, ub_i as f32, 1.0);

            for j in 0..j_max {
                let lb_j = j * self.stride.1;
                let ub_j = lb_j + self.kernel_size.1 - 1;
                let seq_j = Seq::new(lb_j as f32, ub_j as f32, 1.0);
                let seqs = &[seq_i, seq_j, Seq::default(), Seq::default()];

                // Select values in the filtering window and pick largest one
                let sub = index(&input, seqs);
                let reshaped = moddims(&sub, Dim4::new(&[self.kernel_size.0 * self.kernel_size.1, 1, n_channels, mb_size]));
                let max = max(&reshaped, 0);

                // Rearrange max values
                if i == 0 && j == 0 {
                    max_values = max;
                } else {
                    max_values = join(0, &max_values, &max);
                }
            }
        }
        // Reshape the array containing the max values and return result
        transpose(&moddims(&max_values, Dim4::new(&[j_max, i_max, n_channels, mb_size])), false)
    }

    fn compute_activation_mut(&mut self, input: &Array<f64>) -> Array<f64> {
        let height = input.dims().get()[0];
        let width = input.dims().get()[1];
        let n_channels = input.dims().get()[2];
        let mb_size = input.dims().get()[3];

        // Check if values work
        if (height - self.kernel_size.0) % self.stride.0 != 0 || (width - self.kernel_size.1) % self.stride.1 != 0 {
            panic!("The input dimensions, kernel size and stride are incompatible. (dimension - kernel_size) / stride must be an integer.");
        }

        let i_max = (height - self.kernel_size.0) / self.stride.0 + 1;
        let j_max = (width - self.kernel_size.1) / self.stride.1 + 1;

        let mut max_values = Array::new_empty(Dim4::new(&[1, 1, n_channels, mb_size]));
        let mut grad_values = constant(0.0f64, Dim4::new(&[height, width, n_channels, mb_size]));

        // Generate indices to select values in the filtering window
        for i in 0..i_max {
            let lb_i = i * self.stride.0;
            let ub_i = lb_i + self.kernel_size.0 - 1;
            let seq_i = Seq::new(lb_i as f32, ub_i as f32, 1.0);

            for j in 0..j_max {
                let lb_j = j * self.stride.1;
                let ub_j = lb_j + self.kernel_size.1 - 1;
                let seq_j = Seq::new(lb_j as f32, ub_j as f32, 1.0);
                let seqs = &[seq_i, seq_j, Seq::default(), Seq::default()];

                // Select values in the filtering window and pick largest one
                let sub = index(&input, seqs);
                let reshaped = moddims(&sub, Dim4::new(&[self.kernel_size.0 * self.kernel_size.1, 1, n_channels, mb_size]));
                let max = max(&reshaped, 0);

                // Compute the gradient
                let mut mask = eq(&reshaped, &max, true);
                mask = moddims(&mask, Dim4::new(&[self.kernel_size.0, self.kernel_size.1, n_channels, mb_size]));
                let grad_sub = index(&grad_values, seqs);
                let max_mask_grad = maxof(&mask, &grad_sub, true);
                grad_values = assign_seq(&grad_values, seqs, &max_mask_grad);

                // Rearrange max values
                if i == 0 && j == 0 {
                    max_values = max;
                } else {
                    max_values = join(0, &max_values, &max);
                }
            }
        }
        self.grad = grad_values;

        // Reshape the array containing the max values and return result
        transpose(&moddims(&max_values, Dim4::new(&[j_max, i_max, n_channels, mb_size])), false)
    }

    fn output_shape(&self) -> Dim4 {
        self.output_shape
    }

    fn compute_dactivation_mut(&mut self, _dz: &Array<f64>) -> Array<f64> {
        self.grad.copy()
    }

    fn parameters(&self) -> Vec<&Array<f64>> {
        Vec::new()
    }

    fn dparameters(&self) -> Vec<&Array<f64>> {
        Vec::new()
    }

    fn set_parameters(&mut self, parameters: Vec<Array<f64>>) {
        {}
    }
}


#[cfg(test)]
mod tests {
    use arrayfire::*;
    use crate::layers::{max_pooling, MaxPooling2D, Layer};
    use crate::assert_approx_eq;

    #[test]
    fn test_max_pooling() {
        // Generate array of dimension [3 4 1 2]
        let val = [2., 9., -2., 8., 13., -21., -5., 10., 1., 0., -1., 14., -17., 6., 22., 4., -2., -8., 0., 11., -1., -20., 19., 12.];
        let arr = Array::new(&val, Dim4::new(&[3, 4, 1, 2]));

        let max_pooling = MaxPooling2D::with_param((2, 2), (1, 1));
        let activation = max_pooling.compute_activation(&arr);

        let expected_output = [13., 13., 13., 13., 10., 14., 6., 22., 11., 11., 19., 19.];
        let mut output: [f64; 12] = [0.; 12];
        activation.host(&mut output);
        assert_approx_eq!(expected_output, output);
    }
}