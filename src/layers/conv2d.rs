use arrayfire::*;
use crate::Tensor;
use crate::tensor::*;
use crate::activations::*;
use crate::layers::initializers::*;
use super::Layer;
use std::io;
use std::io::BufWriter;
use std::fs;
use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Padding {
    Same,
    Valid,
}


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

    fn compute_convolution_mut(&mut self, input: &Tensor) -> Tensor {
        let mb_size = input.dims().get()[3];

        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];

        // Pad input if necessary
        let padded = self.pad_input(&input);

        // Flatten input into column array
        let input_values = match &padded {
            Some(p) => self.img_to_flat(&p),
            None => self.img_to_flat(input)
        };

        // Compute the convolution and add biases
        let conv = add(&matmul(&self.weights, &input_values, MatProp::NONE, MatProp::NONE), &self.biases, true);

        // Save reshaped input for efficient backward prop
        self.reshaped_input = input_values;

        // Reshape to have each mini-batch on the last dimension
        let tmp = moddims(&conv, Dim4::new(&[self.num_filters, h_out * w_out, 1, mb_size]));

        // Reshape to have correct output dimensions
        moddims(&transpose(&tmp, false), Dim4::new(&[h_out, w_out, self.num_filters, mb_size]))
    }

    fn compute_convolution(&self, input: &Tensor) -> Tensor {
        let mb_size = input.dims().get()[3];

        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];

        // Pad input if necessary
        let padded = self.pad_input(&input);

        // Transform input into column array
        let input_values = match &padded {
            Some(p) => self.img_to_flat(&p),
            None => self.img_to_flat(input)
        };

        // Compute the convolution and add biases
        let conv = add(&matmul(&self.weights, &input_values, MatProp::NONE, MatProp::NONE), &self.biases, true);

        // Reshape to have each mini-batch on the last dimension
        let tmp = moddims(&conv, Dim4::new(&[self.num_filters, h_out * w_out, 1, mb_size]));

        // Reshape to have correct output dimensions
        moddims(&transpose(&tmp, false), Dim4::new(&[h_out, w_out, self.num_filters, mb_size]))
    }

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

    fn img_to_flat(&self, input: &Tensor) -> Tensor {
        let num_channels = input.dims().get()[2];
        let mb_size = input.dims().get()[3];

        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];

        // Iterate over all positions of the filtering window
        let mut input_values = Tensor::new_empty(Dim4::new(&[self.kernel_size.0 * self.kernel_size.1 * num_channels, 1, 1, mb_size]));
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
                let sub_col = moddims(&sub, Dim4::new(&[self.kernel_size.0 * self.kernel_size.1 * num_channels, 1, 1, mb_size]));

                if i == 0 && j == 0 {
                    input_values = sub_col;
                } else {
                    input_values = join(1, &input_values, &sub_col);
                }
            }
        }
        // Stack mini-batches along second dimension
        moddims(&input_values, Dim4::new(&[self.kernel_size.0 * self.kernel_size.1 * num_channels, input_values.dims().get()[1] * mb_size , 1, 1]))
    }

    fn dout_to_flat(&self, input: &Tensor) -> Tensor {
        let h_out = input.dims().get()[0];
        let w_out = input.dims().get()[1];
        let mb_size = input.dims().get()[3];

        let mut tmp = moddims(input, Dim4::new(&[h_out * w_out, 1, self.num_filters, mb_size]));
        tmp = reorder(&tmp, Dim4::new(&[0, 3, 2, 1]));
        transpose(&moddims(&tmp, Dim4::new(&[h_out * w_out * mb_size, self.num_filters, 1, 1])), false)
    }

    fn flat_to_img(&self, input: &Tensor) -> Tensor {
        let num_channels = self.input_shape.get()[2];
        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];
        let num_col = h_out * w_out;
        let mb_size = input.dims().get()[1] / num_col;
        let height_padded = (h_out - 1) * self.stride.0 + self.kernel_size.0;
        let width_padded = (w_out - 1) * self.stride.1 + self.kernel_size.1;
        let mut out = Tensor::zeros(Dim4::new(&[height_padded, width_padded, num_channels, mb_size]));

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
        }

        // Remove padding
        index(&out, &[Seq::new(self.padding_size.0 as f32, (height_padded - self.padding_size.2 - 1) as f32, 1.0), Seq::new(self.padding_size.3 as f32, (width_padded - self.padding_size.1 - 1) as f32, 1.0), Seq::default(), Seq::default()])
    }
}

impl Layer for Conv2D {
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let height = input_shape.get()[0];
        let width = input_shape.get()[1];
        let num_channels = input_shape.get()[2];
        let mb_size = input_shape.get()[3];

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
        let linear_activation = self.compute_convolution(input);
        self.activation.eval(&linear_activation)
    }

    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor {
        //println!("Computing linear activation");
        let linear_activation = self.compute_convolution_mut(input);
        //println!("Computing nonlinear activation");
        let nonlinear_activation = self.activation.eval(&linear_activation);

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
        let input_flat = self.dout_to_flat(&self.activation.grad(input));
        self.dweights = matmul(&input_flat, &self.reshaped_input, MatProp::NONE, MatProp::TRANS);
        self.dweights.eval();

        // Compute the derivative wrt the input
        let mut dinput = matmul(&self.weights, &input_flat, MatProp::TRANS, MatProp::NONE);
        self.flat_to_img(&dinput)
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

    fn dparameters(&self) -> Option<Vec<&Tensor>> {
        Some(vec![&self.dweights, &self.dbiases])
    }

    /*
    fn set_parameters(&mut self, parameters: Vec<Tensor>) {
        self.weights = parameters[0].copy();
        self.biases = parameters[1].copy();
    }
    */


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