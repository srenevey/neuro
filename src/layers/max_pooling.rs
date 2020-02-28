//! 2D max pooling layer
use arrayfire::*;
use std::fmt;

use crate::errors::Error;
use crate::layers::Layer;
use crate::tensor::*;

/// Defines a 2D max pooling layer.
pub struct MaxPool2D {
    pool_size: (u64, u64),
    stride: (u64, u64),
    input_shape: Dim,
    output_shape: Dim,
    row_indices: Array<i32>,
    col_indices: Array<i32>,
}

impl MaxPool2D {

    pub(crate) const NAME: &'static str = "MaxPool2D";

    /// Creates a 2D max pooling layer.
    ///
    /// By default, the horizontal and vertical strides are set to the height and width of the pooling window.
    ///
    /// # Arguments
    ///
    /// * `pool_size` - The height and width of the pooling window.
    pub fn new(pool_size: (u64, u64)) -> Box<MaxPool2D> {
        Box::new(MaxPool2D {
            pool_size,
            stride: pool_size,
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            row_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
            col_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
        })
    }


    /// Creates a 2D max pooling layer with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `pool_size` - The height and width of the moving window.
    /// * `stride` - The vertical and horizontal stride.
    pub fn with_param(pool_size: (u64, u64), stride: (u64, u64)) -> Box<MaxPool2D> {
        Box::new(MaxPool2D {
            pool_size,
            stride,
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            row_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
            col_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
        })
    }

    /// Creates a MaxPool2D layer from an HDF5 group.
    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<MaxPool2D> {
        let pool_size = group.dataset("pool_size").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the pool size.");
        let stride = group.dataset("stride").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the stride.");
        let input_shape = group.dataset("input_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the input shape.");
        let output_shape = group.dataset("output_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape.");

        Box::new(MaxPool2D {
            pool_size: (pool_size[0][0], pool_size[0][1]),
            stride: (stride[0][0], stride[0][1]),
            input_shape: Dim::new(&input_shape[0]),
            output_shape: Dim::new(&output_shape[0]),
            row_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
            col_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
        })
    }

    /// Computes the maximum value in the pooling window.
    fn max_pool(&self, input: &Tensor) -> (Tensor, Array<i32>, Array<i32>) {
        let cols = unwrap(input, self.pool_size.0 as i64, self.pool_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true);
        let cols_reshaped = moddims(&cols, Dim4::new(&[cols.dims().get()[0], cols.elements() as u64 / cols.dims().get()[0], 1, 1]));

        // Computes max values and indices
        let (mut max_values, row_indices_u32) = imax(&cols_reshaped, 0);

        // Creates the output
        let output = moddims(&max_values, Dim4::new(&[self.output_shape.get()[0], self.output_shape.get()[1], input.dims().get()[2], input.dims().get()[3]]));

        // Creates rows and columns indices
        let mut row_indices: Array<i32> = row_indices_u32.cast();
        //row_indices = reorder(&row_indices, Dim4::new(&[1, 0, 2, 3]));
        row_indices = reorder_v2(&row_indices, 1, 0, Some(vec![2, 3]));

        //max_values = reorder(&max_values, Dim4::new(&[1, 0, 2, 3]));
        max_values = reorder_v2(&max_values, 1, 0, Some(vec![2, 3]));
        let num_cols = max_values.dims().get()[0];
        let col_indices_vec: Vec<i32> = (0..num_cols as i32).collect();
        let mut col_indices = Array::new(&col_indices_vec[..], Dim4::new(&[num_cols, 1, 1, 1]));
        col_indices = tile(&col_indices, Dim4::new(&[cols_reshaped.dims().get()[2] * cols_reshaped.dims().get()[3], 1, 1, 1]));

        (output, row_indices, col_indices)
    }
}

impl Layer for MaxPool2D {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let output_height = ((input_shape.get()[0] - self.pool_size.0) as f64 / self.stride.0 as f64 + 1.).floor() as u64;
        let output_width = ((input_shape.get()[1] - self.pool_size.1) as f64 / self.stride.1 as f64 + 1.).floor() as u64;
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
        let sparse = sparse(self.pool_size.0 * self.pool_size.1, input.elements() as u64, &flat_input, &self.row_indices, &self.col_indices, SparseFormat::COO);
        let mut dense = sparse_to_dense(&sparse);
        let num_channels = self.input_shape.get()[2];
        let num_cols = dense.dims().get()[1] / (num_channels * batch_size);
        dense = moddims(&dense, Dim4::new(&[dense.dims().get()[0], num_cols, num_channels, batch_size]));
        wrap(&dense, self.input_shape.get()[0] as i64, self.input_shape.get()[1] as i64, self.pool_size.0 as i64, self.pool_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true)
    }

    fn output_shape(&self) -> Dim {
        self.output_shape
    }


    fn save(&self, group: &hdf5::Group, layer_number: usize) -> Result<(), Error> {
        let group_name = layer_number.to_string() + &String::from("_") + Self::NAME;
        let max_pool = group.create_group(&group_name)?;

        let pool_size = max_pool.new_dataset::<[u64; 2]>().create("pool_size", 1)?;
        pool_size.write(&[[self.pool_size.0, self.pool_size.1]])?;

        let stride = max_pool.new_dataset::<[u64; 2]>().create("stride", 1)?;
        stride.write(&[[self.stride.0, self.stride.1]])?;

        let input_shape = max_pool.new_dataset::<[u64; 4]>().create("input_shape", 1)?;
        input_shape.write(&[*self.input_shape.get()])?;

        let output_shape = max_pool.new_dataset::<[u64; 4]>().create("output_shape", 1)?;
        output_shape.write(&[*self.output_shape.get()])?;

        Ok(())
    }
}


impl fmt::Display for MaxPool2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} \t 0 \t\t [{}, {}, {}]", Self::NAME, self.output_shape[0], self.output_shape[1], self.output_shape[2])
    }
}


#[cfg(test)]
mod tests {
    use arrayfire::*;
    use crate::layers::{MaxPool2D, Layer};
    use crate::assert_approx_eq;
    use crate::tensor::*;

    fn create_test_layer() -> MaxPool2D {
        MaxPool2D {
            pool_size: (2, 2),
            stride: (2, 2),
            input_shape: Dim::new(&[4, 4, 2, 1]),
            output_shape: Dim::new(&[2, 2, 2, 1]),
            row_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
            col_indices: Array::new(&[0], Dim4::new(&[1, 1, 1, 1])),
        }
    }

    #[test]
    fn test_maxpool2d_forward() {
        // Generate array of dimension [3 4 1 2]
        let input_val = [3., -1., -8., 2., 5., -4., 1., 7., 0., 3., 1., 1., -2., 6., 8., -5., -1., 8., 3., -4., 5., 6., -2., 0., -1., -3., -8., 4., 2., 9., -1., 5., 6., -1., 0., 1., 4., -2., -3., 1., 5., 8., -2., 6., 5., 3., 1., -4., 2., 9., -7., 5., 1., 4., 0., 3., -2., -6., 1., 8., -7., 2., -3., -1.];
        let input = Tensor::new(&input_val, Dim4::new(&[4, 4, 2, 2]));

        let mut layer = create_test_layer();
        let layer_output = layer.compute_activation_mut(&input);

        let mut output: [PrimitiveType; 16] = [0.; 16];
        layer_output.host(&mut output);
        let expected_output = [5., 7., 6., 8., 8., 3., 9., 5., 6., 1., 8., 6., 9., 5., 2., 8.];

        assert_approx_eq!(expected_output, output);
    }

    #[test]
    fn test_maxpool2d_backward() {
        // Generate array of dimension [3 4 1 2]
        let input_val = [3., -1., -8., 2., 5., -4., 1., 7., 0., 3., 1., 1., -2., 6., 8., -5., -1., 8., 3., -4., 5., 6., -2., 0., -1., -3., -8., 4., 2., 9., -1., 5., 6., -1., 0., 1., 4., -2., -3., 1., 5., 8., -2., 6., 5., 3., 1., -4., 2., 9., -7., 5., 1., 4., 0., 3., -2., -6., 1., 8., -7., 2., -3., -1.];
        let input_forward = Tensor::new(&input_val, Dim4::new(&[4, 4, 2, 2]));

        let mut layer = create_test_layer();
        let _ = layer.compute_activation_mut(&input_forward);

        let input_backward = Tensor::new(&[-1., 2., 3., 1., -2., 4., -1., 1., 2., 1.,-3., 1., -2., 0., 1., 4.], Dim::new(&[2, 2, 2, 2]));
        let layer_output = layer.compute_dactivation_mut(&input_backward);
        let mut output: [PrimitiveType; 64] = [0.; 64];
        layer_output.host(&mut output);
        let expected_output: [PrimitiveType; 64] = [0., 0., 0., 0., -1., 0., 0., 2., 0., 0., 0., 0., 0., 3., 1., 0., 0., -2., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 1., 2., 0., 0., 1., 0., 0., 0., 0., 0., -3., 0., 1., 0., 0., 0., 0., 0., -2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0., 1., 0., 0.];
    }
}
