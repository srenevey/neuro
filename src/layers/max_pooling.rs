//! 2D max pooling layer
use crate::layers::Layer;
use crate::tensor::*;

use std::fmt;
use std::fs;
use std::io;
use std::io::BufWriter;

use arrayfire::*;

/// Defines a 2D max pooling layer.
pub struct MaxPooling2D {
    filter_size: (u64, u64),
    stride: (u64, u64),
    input_shape: Dim4,
    output_shape: Dim4,
    row_indices: Array<i32>,
    col_indices: Array<i32>,
}

impl MaxPooling2D {

    /// Creates a 2D max pooling layer with the specified parameters.
    ///
    /// # Arguments
    /// * `filter_size`: height and width of the filter
    /// * `stride`: horizontal and vertical stride
    ///
    pub fn new(filter_size: (u64, u64), stride: (u64, u64)) -> Box<MaxPooling2D> {
        Box::new(MaxPooling2D {
            filter_size,
            stride,
            input_shape: Dim4::new(&[0, 0, 0, 0]),
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            row_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
            col_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
        })
    }

    /// Computes the maximum value in the window
    fn max_pool(&self, input: &Tensor) -> (Tensor, Array<i32>, Array<i32>) {
        let cols = unwrap(input, self.filter_size.0 as i64, self.filter_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true);
        let cols_reshaped = moddims(&cols, Dim4::new(&[cols.dims().get()[0], cols.elements() as u64 / cols.dims().get()[0], 1, 1]));

        // Computes max values and indices
        let (mut max_values, row_indices_u32) = imax(&cols_reshaped, 0);

        // Creates the output
        let output = moddims(&max_values, Dim4::new(&[self.output_shape.get()[0], self.output_shape.get()[1], input.dims().get()[2], input.dims().get()[3]]));

        // Creates rows and columns indices
        let mut row_indices: Array<i32> = row_indices_u32.cast();
        row_indices = reorder(&row_indices, Dim4::new(&[1, 0, 2, 3]));
        max_values = reorder(&max_values, Dim4::new(&[1, 0, 2, 3]));
        let num_cols = max_values.dims().get()[0];
        let col_indices_vec: Vec<i32> = (0..num_cols as i32).collect();
        let mut col_indices = Array::new(&col_indices_vec[..], Dim4::new(&[num_cols, 1, 1, 1]));
        col_indices = tile(&col_indices, Dim4::new(&[cols_reshaped.dims().get()[2] * cols_reshaped.dims().get()[3], 1, 1, 1]));

        (output, row_indices, col_indices)
    }
}

impl Layer for MaxPooling2D {
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let output_height = ((input_shape.get()[0] - self.filter_size.0) as f64 / self.stride.0 as f64 + 1.).floor() as u64;
        let output_width = ((input_shape.get()[1] - self.filter_size.1) as f64 / self.stride.1 as f64 + 1.).floor() as u64;
        self.input_shape = input_shape;
        self.output_shape = Dim4::new(&[output_height, output_width, input_shape.get()[2], input_shape.get()[3]]);
    }

    fn compute_activation(&self, input: &Tensor) -> Tensor {
        let (output, _, _) = self.max_pool(input);
        output
    }

    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor {
        let (output, row_indices, col_indices) = self.max_pool(input);
        self.row_indices = row_indices;
        self.col_indices = col_indices;
        output
    }

    fn compute_dactivation_mut(&mut self, input: &Tensor) -> Tensor {
        let batch_size = input.dims().get()[3];
        let flat_input = flat(input);
        let sparse = sparse(self.filter_size.0 * self.filter_size.1, input.elements() as u64, &flat_input, &self.row_indices, &self.col_indices, SparseFormat::COO);
        let mut dense = sparse_to_dense(&sparse);
        let num_channels = self.input_shape.get()[2];
        let num_cols = dense.dims().get()[1] / (num_channels * batch_size);
        dense = moddims(&dense, Dim4::new(&[dense.dims().get()[0], num_cols, num_channels, batch_size]));
        wrap(&dense, self.input_shape.get()[0] as i64, self.input_shape.get()[1] as i64, self.filter_size.0 as i64, self.filter_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true)
    }

    fn output_shape(&self) -> Dim4 {
        self.output_shape
    }


    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        Ok(())
    }
}


impl fmt::Display for MaxPooling2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MaxPooling2D \t 0")
    }
}


#[cfg(test)]
mod tests {
    use arrayfire::*;
    use crate::layers::{MaxPooling2D, Layer};
    use crate::assert_approx_eq;
    use crate::tensor::*;

    #[test]
    fn test_max_pooling() {
        // Generate array of dimension [3 4 1 2]
        let val = [2., 9., -2., 8., 13., -21., -5., 10., 1., 0., -1., 14., -17., 6., 22., 4., -2., -8., 0., 11., -1., -20., 19., 12.];
        let arr = Tensor::new(&val, Dim4::new(&[3, 4, 1, 2]));

        let max_pooling = MaxPooling2D::new((2, 2), (1, 1));
        let activation = max_pooling.compute_activation(&arr);

        let expected_output = [13., 13., 13., 13., 10., 14., 6., 22., 11., 11., 19., 19.];
        let mut output: [PrimitiveType; 12] = [0.; 12];
        activation.host(&mut output);
        assert_approx_eq!(expected_output, output);
    }
}
