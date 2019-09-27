use arrayfire::*;
use crate::activations::*;
use crate::layers::initializers::*;
use super::Layer;

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
    num_filters: u64,
    output_shape: Dim4,
    weights: Array<f64>,
    biases: Array<f64>,
    dweights: Array<f64>,
    dbiases: Array<f64>,
    z: Option<Array<f64>>,
    a_prev: Option<Array<f64>>,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
}

impl Conv2D {
    pub fn new(num_filters: u64, kernel_size: (u64, u64), padding: Padding) -> Box<Conv2D> {
        Box::new(Conv2D {
            activation: Activation::ReLU,
            kernel_size,
            stride: (1, 1),
            padding,
            num_filters,
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            weights: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            biases: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dweights: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dbiases: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            z: None,
            a_prev: None,
            weights_initializer: Initializer::HeUniform,
            biases_initializer: Initializer::Zeros,
        })
    }

    pub fn with_param(num_filters: u64, kernel_size: (u64, u64), stride: (u64, u64), padding: Padding, activation: Activation, weights_initializer: Initializer, biases_initializer: Initializer) -> Box<Conv2D> {

        Box::new(Conv2D {
            activation,
            kernel_size,
            stride,
            padding,
            num_filters,
            output_shape: Dim4::new(&[0, 0, 0, 0]),
            weights: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            biases: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dweights: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            dbiases: Array::new_empty(Dim4::new(&[0, 0, 0, 0])),
            z: None,
            a_prev: None,
            weights_initializer,
            biases_initializer,
        })
    }

    fn compute_convolution(&self, input: &Array<f64>) -> Array<f64> {
        let height = input.dims().get()[0];
        let width = input.dims().get()[1];
        let num_channels = input.dims().get()[2];
        let mb_size = input.dims().get()[3];

        // Create padded input
        let mut h_out = 0;
        let mut w_out = 0;
        let mut pad = (0, 0, 0, 0); // top, right, bottom, left
        let padded = match self.padding {
            Padding::Same => {
                h_out = (height as f64 / self.stride.0 as f64).ceil() as u64;
                w_out = (width as f64 / self.stride.1 as f64).ceil() as u64;

                let pad_along_h = std::cmp::max((h_out - 1) * self.stride.0 + self.kernel_size.0 - height, 0);
                let pad_along_w = std::cmp::max((w_out - 1) * self.stride.1 + self.kernel_size.1 - width, 0);
                if pad_along_h != 0 {
                    if pad_along_h % 2 == 0 {
                        pad.0 = pad_along_h / 2;
                        pad.2 = pad_along_h / 2;
                    } else {
                        pad.0 = (pad_along_h - 1) / 2;
                        pad.2 = (pad_along_h + 1) / 2;
                    }
                }
                if pad_along_w != 0 {
                    if pad_along_w % 2 == 0 {
                        pad.1 = pad_along_w / 2;
                        pad.3 = pad_along_w / 2;
                    } else {
                        pad.1 = (pad_along_w + 1) / 2;
                        pad.3 = (pad_along_w - 1) / 2;
                    }
                }

                let pad_top = constant(0.0f64, Dim4::new(&[pad.0, width, num_channels, mb_size]));
                let pad_right = constant(0.0f64, Dim4::new(&[height + pad.0, pad.1, num_channels, mb_size]));
                let pad_bottom = constant(0.0f64, Dim4::new(&[pad.2, width + pad.1, num_channels, mb_size]));
                let pad_left = constant(0.0f64, Dim4::new(&[height + pad.0 + pad.2, pad.3, num_channels, mb_size]));
                let mut padded = join(0, &pad_top, input);
                padded = join(1, &padded, &pad_right);
                padded = join(0, &padded, &pad_bottom);
                padded = join(1, &pad_left, &padded);
                Some(padded)
            },
            Padding::Valid => {
                h_out = (((height - self.kernel_size.0 + 1) as f64) / self.stride.0 as f64).ceil() as u64;
                w_out = (((width - self.kernel_size.1 + 1) as f64) / self.stride.1 as f64).ceil() as u64;
                None
            }
        };

        let mut values = Array::new_empty(Dim4::new(&[1, 1, num_channels, mb_size]));

        // Generate indices to select values in the filtering window
        for i in 0..h_out {
            let lb_i = i * self.stride.0;
            let ub_i = lb_i + self.kernel_size.0 - 1;
            let seq_i = Seq::new(lb_i as f32, ub_i as f32, 1.0);

            for j in 0..w_out {
                let lb_j = j * self.stride.1;
                let ub_j = lb_j + self.kernel_size.1 - 1;
                let seq_j = Seq::new(lb_j as f32, ub_j as f32, 1.0);
                let seqs = &[seq_i, seq_j, Seq::default(), Seq::default()];

                // Select values in the filtering window and compute convolution
                let sub = match &padded {
                    Some(p) => index(p, seqs),
                    None => index(input, seqs),
                };
                let mut filter_values = Array::new_empty(Dim4::new(&[1, 1, num_channels, mb_size]));
                for k in 0..self.num_filters {
                    let value = sum(&sum(&sum(&mul(&index(&self.weights, &[Seq::default(), Seq::default(), Seq::default(), Seq::new(k as f64, k as f64, 1.)]), &sub, true), 0), 1), 2);
                    if k == 0 {
                        filter_values = value;
                    } else {
                        filter_values = join(2, &filter_values, &value);
                    }
                }

                // Rearrange values
                if i == 0 && j == 0 {
                    values = filter_values;
                } else {
                    values = join(0, &values, &filter_values);
                }
            }
        }

        // Reshape the array containing the max values and return
        transpose(&moddims(&values, Dim4::new(&[w_out, h_out, self.num_filters, mb_size])), false)
    }
}

impl Layer for Conv2D {
    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let height = input_shape.get()[0];
        let width = input_shape.get()[1];
        let num_channels = input_shape.get()[2];
        let mb_size = input_shape.get()[3];

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
        let fan_in = (height * width * num_channels) as f64;
        let fan_out = (h_out * w_out * self.num_filters) as f64;
        self.output_shape = Dim4::new(&[h_out, w_out, self.num_filters, 1]);
        self.weights = self.weights_initializer.new(Dim4::new(&[self.kernel_size.0, self.kernel_size.1, input_shape.get()[2], self.num_filters]), fan_in, fan_out);
        self.biases = self.biases_initializer.new(Dim4::new(&[1, 1, 1, self.num_filters]), fan_in, fan_out);
    }

    fn compute_activation(&self, input: &Array<f64>) -> Array<f64> {
        let conv = self.compute_convolution(input);
        self.activation.eval(&add(&conv, &self.biases, true))
    }

    fn compute_activation_mut(&mut self, input: &Array<f64>) -> Array<f64> {
        self.a_prev = Some(input.clone());

        let conv = self.compute_convolution(input);
        self.z = Some(add(&conv, &self.biases, true));

        match &self.z {
            Some(z) => self.activation.eval(z),
            None => panic!("The linear activations z have not been computed!")
        }
    }

    fn output_shape(&self) -> Dim4 {
        self.output_shape
    }

    fn compute_dactivation_mut(&mut self, dz: &Array<f64>) -> Array<f64> {
        unimplemented!()
    }

    fn parameters(&self) -> Vec<&Array<f64>> {
        vec![&self.weights, &self.biases]
    }

    fn dparameters(&self) -> Vec<&Array<f64>> {
        vec![&self.dweights, &self.dbiases]
    }

    fn set_parameters(&mut self, parameters: Vec<Array<f64>>) {
        self.weights = parameters[0].copy();
        self.biases = parameters[1].copy();
    }
}