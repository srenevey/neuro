//! 2D convolution layer
use arrayfire::*;
use crate::Tensor;
use crate::tensor::*;
use crate::activations::*;
use crate::initializers::*;
use super::Layer;
use std::io;
use std::io::BufWriter;
use std::fs;
use std::fmt;

/// Defines the type of padding applied to the inputs.
///
/// * Same: a same convolution is such that the dimensions of the output of the convolution is the
/// same as the dimensions of the input, provided a stride of 1.
/// * Valid: a valid convolution is such that the kernel is moved as long as the shift results in a valid convolution operation. No padding is applied.
///
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Padding {
    Same,
    Valid,
}


/// Defines a 2D convolution layer.
pub struct Conv2D {
    activation: Activation,
    kernel_size: (u64, u64),
    stride: (u64, u64),
    padding: Padding,
    padding_size: (u64, u64, u64, u64), // top, right, bottom, left
    num_filters: u64,
    input_shape: Dim4,
    output_shape: Dim4,
    weights: Tensor,
    biases: Tensor,
    dweights: Tensor,
    dbiases: Tensor,
    z: Option<Tensor>,
    a_prev: Option<Tensor>,
    reshaped_input: Tensor,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
}

impl Conv2D {
    /// Creates a 2D convolution layer with the given parameters.
    ///
    /// By default, a ReLU activation is used and the parameters of the kernels are initialized
    /// using a HeUniform initializer and the biases of the layer a Zeros initializer.
    ///
    /// # Arguments
    /// * `num_filters`: number of filters in the layer
    /// * `kernel_size`: height and width of the convolution kernels
    /// * `stride`: vertical and horizontal stride used for the convolution
    /// * `padding`: padding used for the convolution. Must be a variant of Padding.
    ///
    pub fn new(num_filters: u64,
               kernel_size: (u64, u64),
               stride: (u64, u64),
               padding: Padding
    ) -> Box<Conv2D> {
        Box::new(Conv2D {
            activation: Activation::ReLU,
            kernel_size,
            stride,
            padding,
            padding_size: (0, 0, 0, 0),
            num_filters,
            input_shape: Dim4::new(&[0, 0, 0, 0]),
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            weights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            z: None,
            a_prev: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer: Initializer::HeUniform,
            biases_initializer: Initializer::Zeros,
        })
    }

    /// Creates a 2D convolution layer with the given parameters.
    ///
    /// By default, the parameters of the kernels are initialized using a HeUniform initializer and the biases
    /// of the layer a Zeros initializer.
    ///
    /// # Arguments
    /// * `num_filters`: number of filters in the layer
    /// * `kernel_size`: height and width of the convolution kernels
    /// * `stride`: vertical and horizontal stride used for the convolution
    /// * `padding`: padding used for the convolution. Must be a variant of Padding.
    /// * `activation`: activation function used by the layer
    /// * `weights_initializer`: initializer used to initialize the weights of the layer
    /// * `biases_initializer`: initializer used to initialize the biases of the layer
    ///
    pub fn with_param(num_filters: u64,
                      kernel_size: (u64, u64),
                      stride: (u64, u64),
                      padding: Padding,
                      activation: Activation,
                      weights_initializer: Initializer,
                      biases_initializer: Initializer
    ) -> Box<Conv2D> {

        Box::new(Conv2D {
            activation,
            kernel_size,
            stride,
            padding,
            padding_size: (0, 0, 0, 0),
            num_filters,
            input_shape: Dim4::new(&[0, 0, 0, 0]),
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            weights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            z: None,
            a_prev: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer,
            biases_initializer,
        })
    }


    /// Computes the convolution.
    fn compute_convolution(&self, input: &Tensor) -> (Tensor, Tensor) {
        let batch_size = input.dims().get()[3];

        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];

        // Pad input if necessary
        let padded = self.pad_input(&input);

        // Transform input into column array
        let input_values = match &padded {
            Some(p) => self.img_to_col(&p),
            None => self.img_to_col(input)
        };

        // Compute the convolution and add biases
        let conv = add(&matmul(&self.weights, &input_values, MatProp::NONE, MatProp::NONE), &self.biases, true);

        // Reshape to have each mini-batch on the last dimension
        let tmp = moddims(&conv, Dim4::new(&[self.num_filters, h_out * w_out, 1, batch_size]));

        // Reshape to have correct output dimensions
        let linear_activation = moddims(&transpose(&tmp, false), Dim4::new(&[h_out, w_out, self.num_filters, batch_size]));
        (linear_activation, input_values)
    }

    /// Computes the padding that must be added to the images.
    fn compute_padding_size(&mut self, height: u64, width: u64, h_out: u64, w_out: u64) {
        match self.padding {
            Padding::Same => {
                let pad_along_h = std::cmp::max((h_out - 1) * self.stride.0 + self.kernel_size.0 - height, 0);
                let pad_along_w = std::cmp::max((w_out - 1) * self.stride.1 + self.kernel_size.1 - width, 0);
                if pad_along_h != 0 {
                    if pad_along_h % 2 == 0 {
                        self.padding_size.0 = pad_along_h / 2;
                        self.padding_size.2 = pad_along_h / 2;
                    } else {
                        self.padding_size.0 = (pad_along_h - 1) / 2;
                        self.padding_size.2 = (pad_along_h + 1) / 2;
                    }
                }
                if pad_along_w != 0 {
                    if pad_along_w % 2 == 0 {
                        self.padding_size.1 = pad_along_w / 2;
                        self.padding_size.3 = pad_along_w / 2;
                    } else {
                        self.padding_size.1 = (pad_along_w + 1) / 2;
                        self.padding_size.3 = (pad_along_w - 1) / 2;
                    }
                }
            },
            Padding::Valid => {}
        }
    }

    /// Applies the padding to the layer's inputs.
    fn pad_input(&self, input: &Tensor) -> Option<Tensor> {
        let height = input.dims().get()[0];
        let width = input.dims().get()[1];
        let num_channels = input.dims().get()[2];
        let mb_size = input.dims().get()[3];

        // Create padded input
        match self.padding {
            Padding::Same => {
                let pad_top = constant(0.0 as PrimitiveType, Dim4::new(&[self.padding_size.0, width, num_channels, mb_size]));
                let pad_right = constant(0.0 as PrimitiveType, Dim4::new(&[height + self.padding_size.0, self.padding_size.1, num_channels, mb_size]));
                let pad_bottom = constant(0.0 as PrimitiveType, Dim4::new(&[self.padding_size.2, width + self.padding_size.1, num_channels, mb_size]));
                let pad_left = constant(0.0 as PrimitiveType, Dim4::new(&[height + self.padding_size.0 + self.padding_size.2, self.padding_size.3, num_channels, mb_size]));
                let mut padded = join(0, &pad_top, input);
                padded = join(1, &padded, &pad_right);
                padded = join(0, &padded, &pad_bottom);
                padded = join(1, &pad_left, &padded);
                Some(padded)
            },
            Padding::Valid => {
                None
            }
        }
    }

    /*
    fn img_to_col_indices(&mut self) {
        let kernel_size: (u64, u64) = (2, 2);
        let stride: (u64, u64) = (1, 2);
        let num_batches = 1;
        let num_channels = 3;

        // Create indices for the first dimension
        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];

        let num_step_v = 2;
        let idxs_dim1_val: Vec<u64> = (0..num_step_v).map( |i| {
            (0..kernel_size.0).map(move |j|  j + i * stride.0)
        }).flatten().collect();
        println!("Indices along the first dimension: {:?}", idxs_dim1_val);
        let i = Array::new(&idxs_dim1_val, Dim4::new(&[num_step_v * kernel_size.0, 1, 1, 1]));

        // Create indices for the second dimension
        let num_step_h = 2;
        let idxs_dim0_val: Vec<u64> = (0..num_step_h).map( |i| {
            (0..kernel_size.1).map(move |j|  j + i * stride.1)
        }).flatten().collect();
        println!("Indices along the second dimension: {:?}", idxs_dim0_val);
        let j = Array::new(&idxs_dim0_val, Dim4::new(&[num_step_h * kernel_size.1, 1, 1, 1]));
    }
    */

    /// Convert the image into a column-equivalent representation. This is required for computation speed.
    /// There is a memory cost though.
    fn img_to_col(&self, input: &Tensor) -> Tensor {

        let num_channels = input.dims().get()[2];
        let batch_size = input.dims().get()[3];

        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];

        /*
        // Iterate over all positions of the filtering window
        let mut input_values = Tensor::new_empty(Dim4::new(&[self.kernel_size.0 * self.kernel_size.1 * num_channels, 1, 1, batch_size]));
        for i in 0..h_out {
            let lb_i = i * self.stride.0;
            let ub_i = lb_i + self.kernel_size.0 - 1;
            let seq_i = Seq::new(lb_i as f32, ub_i as f32, 1.0);

            for j in 0..w_out {
                let lb_j = j * self.stride.1;
                let ub_j = lb_j + self.kernel_size.1 - 1;
                let seq_j = Seq::new(lb_j as f32, ub_j as f32, 1.0);
                let seqs = &[seq_i, seq_j, Seq::default(), Seq::default()];

                // Select values in the filtering window and rearrange in columns
                let sub = index(input, seqs);
                let sub_col = moddims(&sub, Dim4::new(&[self.kernel_size.0 * self.kernel_size.1 * num_channels, 1, 1, batch_size]));

                if i == 0 && j == 0 {
                    input_values = sub_col;
                } else {
                    input_values = join(1, &input_values, &sub_col);
                }
            }
        }
        // Stack mini-batches along second dimension
        moddims(&input_values, Dim4::new(&[self.kernel_size.0 * self.kernel_size.1 * num_channels, input_values.dims().get()[1] * batch_size , 1, 1]))
        */

        let mut col = unwrap(input, self.kernel_size.0 as i64, self.kernel_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true);
        col = reorder(&col, Dim4::new(&[0, 2, 1, 3]));
        moddims(&col, Dim4::new(&[self.kernel_size.0 * self.kernel_size.1 * num_channels, h_out * w_out * batch_size, 1, 1]))
    }

    fn dout_to_col(&self, input: &Tensor) -> Tensor {
        let h_out = input.dims().get()[0];
        let w_out = input.dims().get()[1];
        let mb_size = input.dims().get()[3];

        let mut tmp = moddims(input, Dim4::new(&[h_out * w_out, 1, self.num_filters, mb_size]));
        tmp = reorder(&tmp, Dim4::new(&[0, 3, 2, 1]));
        transpose(&moddims(&tmp, Dim4::new(&[h_out * w_out * mb_size, self.num_filters, 1, 1])), false)
    }

    /// Transforms a columns representation of an image into an image with dimensions height x width x channels.
    fn col_to_img(&self, input: &Tensor) -> Tensor {
        let height = self.input_shape.get()[0];
        let width = self.input_shape.get()[1];
        let num_channels = self.input_shape.get()[2];
        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];
        let num_col = h_out * w_out;
        let batch_size = input.dims().get()[1] / num_col;
        let height_padded = (h_out - 1) * self.stride.0 + self.kernel_size.0;
        let width_padded = (w_out - 1) * self.stride.1 + self.kernel_size.1;
        //let mut out = Tensor::zeros(Dim4::new(&[height_padded, width_padded, num_channels, mb_size]));


        /*

        let tmp = moddims(input, Dim4::new(&[input.dims().get()[0], num_col, 1, mb_size]));

        // Populate output
        let mut col_index = 0.;
        for i in 0..h_out {
            let lb_i = i * self.stride.0;
            let ub_i = lb_i + self.kernel_size.0 - 1;
            let seq_i = Seq::new(lb_i as f32, ub_i as f32, 1.0);

            for j in 0..w_out {
                let lb_j = j * self.stride.1;
                let ub_j = lb_j + self.kernel_size.1 - 1;
                let seq_j = Seq::new(lb_j as f32, ub_j as f32, 1.0);
                let seqs = &[seq_i, seq_j, Seq::default(), Seq::default()];

                let col = index(&tmp, &[Seq::default(), Seq::new(col_index, col_index, 1.), Seq::default(), Seq::default()]);
                let filter = moddims(&col, Dim4::new(&[self.kernel_size.0, self.kernel_size.1, num_channels, mb_size]));

                let sub = index(&out, seqs) + filter;
                out = assign_seq(&out, seqs, &sub);

                col_index += 1.;
            }
        }*/

        let mut img = moddims(input, Dim4::new(&[self.kernel_size.0 * self.kernel_size.1, num_channels, h_out * w_out, batch_size]));
        img = reorder(&img, Dim4::new(&[0, 2, 1, 3]));
        img = wrap(&img, height_padded as i64, width_padded as i64, self.kernel_size.0 as i64, self.kernel_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true);

        // Remove padding
        index(&img, &[Seq::new(self.padding_size.0 as f32, (height_padded - self.padding_size.2 - 1) as f32, 1.0), Seq::new(self.padding_size.3 as f32, (width_padded - self.padding_size.1 - 1) as f32, 1.0), Seq::default(), Seq::default()])
    }
}

impl Layer for Conv2D {
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let height = input_shape.get()[0];
        let width = input_shape.get()[1];
        let num_channels = input_shape.get()[2];
        let batch_size = input_shape.get()[3];

        // Compute output dimensions and padding size
        let mut h_out = 0;
        let mut w_out = 0;
        match self.padding {
            Padding::Same => {
                h_out = (height as f64 / self.stride.0 as f64).ceil() as u64;
                w_out = (width as f64 / self.stride.1 as f64).ceil() as u64;
            },
            Padding::Valid => {
                h_out = (((height - self.kernel_size.0 + 1) as f64) / self.stride.0 as f64).ceil() as u64;
                w_out = (((width - self.kernel_size.1 + 1) as f64) / self.stride.1 as f64).ceil() as u64;
            }
        }
        self.compute_padding_size(height, width, h_out, w_out);

        let fan_in = height * width * num_channels;
        let fan_out = h_out * w_out * self.num_filters;
        self.output_shape = Dim4::new(&[h_out, w_out, self.num_filters, 1]);
        self.input_shape = input_shape;

        // Initialize weights and biases
        self.weights = self.weights_initializer.new(Dim4::new(&[self.num_filters, self.kernel_size.0 * self.kernel_size.1 * num_channels, 1, 1]), fan_in, fan_out);
        self.biases = self.biases_initializer.new(Dim4::new(&[self.num_filters, 1, 1, 1]), fan_in, fan_out);
    }

    fn compute_activation(&self, input: &Tensor) -> Tensor {
        let (linear_activation, _) = self.compute_convolution(input);
        linear_activation.eval();
        let nonlinear_activation = self.activation.eval(&linear_activation);
        nonlinear_activation
    }

    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor {
        //println!("Computing linear activation");
        let (linear_activation, reshaped_input) = self.compute_convolution(input);
        linear_activation.eval();
        reshaped_input.eval();
        self.reshaped_input = reshaped_input;

        //println!("Computing nonlinear activation");
        let nonlinear_activation = self.activation.eval(&linear_activation);
        //nonlinear_activation.eval();

        self.z = Some(linear_activation);
        self.a_prev = Some(input.copy());

        //println!("Forward: Conv2D ouput: {} x {} x {} x {}", nonlinear_activation.dims().get()[0], nonlinear_activation.dims().get()[1], nonlinear_activation.dims().get()[2], nonlinear_activation.dims().get()[3]);
        nonlinear_activation
    }


    fn compute_dactivation_mut(&mut self, input: &Tensor) -> Tensor {

        //println!("Backward: Conv2D input: {} x {} x {} x {}", input.dims().get()[0], input.dims().get()[1], input.dims().get()[2], input.dims().get()[3]);

        let db = sum(&sum(&sum(input, 0), 1), 3) / input.dims().get()[3];
        self.dbiases = reorder(&db, Dim4::new(&[2, 1, 0, 3]));
        self.dbiases.eval();
        let input_flat = self.dout_to_col(&self.activation.grad(input));
        self.dweights = matmul(&input_flat, &self.reshaped_input, MatProp::NONE, MatProp::TRANS);
        self.dweights.eval();

        // Compute the derivative wrt the input
        let mut dinput = matmul(&self.weights, &input_flat, MatProp::TRANS, MatProp::NONE);
        self.col_to_img(&dinput)
    }

    fn output_shape(&self) -> Dim4 {
        self.output_shape
    }


    fn parameters(&self) -> Option<Vec<&Tensor>> {
        Some(vec![&self.weights, &self.biases])
    }


    fn parameters_mut(&mut self) -> Option<(Vec<&mut Tensor>, Vec<&Tensor>)> {
        Some((vec![&mut self.weights, &mut self.biases], vec![&self.dweights, &self.dbiases]))
    }


    fn save(&self, writer: &mut BufWriter<fs::File>) -> io::Result<()> {
        Ok(())
    }

}

impl fmt::Display for Conv2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let num_parameters = self.kernel_size.0 * self.kernel_size.1 * self.num_filters + self.num_filters;
        write!(f, "Conv2D \t\t {}", num_parameters)
    }
}