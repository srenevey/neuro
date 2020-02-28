//! 2D convolution layer
use arrayfire::*;
use std::convert::{TryFrom};
use std::fmt;

use crate::activations::*;
use crate::errors::Error;
use crate::initializers::*;
use crate::regularizers::*;
use crate::tensor::*;
use super::Layer;

/// Defines the type of padding applied to the inputs.
///
/// * Same: a same convolution is such that the dimensions of the output of the convolution is the
/// same as the dimensions of the input, provided a stride of 1.
/// * Valid: a valid convolution is such that the kernel is moved as long as the shift results in a valid convolution operation. No padding is applied.
#[derive(hdf5::H5Type, Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum Padding {
    Same = 0,
    Valid = 1,
}

impl TryFrom<u8> for Padding {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            x if x == Padding::Same as u8 => Ok(Padding::Same),
            x if x == Padding::Valid as u8 => Ok(Padding::Valid),
            _ => Err(()),
        }
    }
}


/// Defines a 2D convolution layer.
pub struct Conv2D {
    activation: Activation,
    kernel_size: (u64, u64),
    stride: (u64, u64),
    padding: Padding,
    padding_size: (u64, u64, u64, u64), // top, right, bottom, left
    num_filters: u64,
    input_shape: Dim,
    output_shape: Dim,
    weights: Tensor,
    biases: Tensor,
    dweights: Tensor,
    dbiases: Tensor,
    linear_activation: Option<Tensor>,
    previous_activation: Option<Tensor>,
    reshaped_input: Tensor,
    weights_initializer: Initializer,
    biases_initializer: Initializer,
    regularizer: Option<Regularizer>,
}

impl Conv2D {

    pub(crate) const NAME: &'static str = "Conv2D";

    /// Creates a 2D convolution layer with the given parameters.
    ///
    /// By default, a ReLU activation is used and the parameters of the kernels are initialized
    /// using a HeNormal initializer and the biases of the layer a Zeros initializer.
    ///
    /// # Arguments
    ///
    /// * `num_filters` - The number of filters in the layer.
    /// * `kernel_size` - The height and width of the convolution kernels.
    /// * `stride` - The vertical and horizontal stride used for the convolution.
    /// * `padding` - The padding used for the convolution. Must be a variant of Padding.
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
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            weights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer: Initializer::HeNormal,
            biases_initializer: Initializer::Zeros,
            regularizer: None,
        })
    }

    /// Creates a 2D convolution layer with the given parameters.
    ///
    /// By default, the parameters of the kernels are initialized using a HeUniform initializer and the biases
    /// of the layer a Zeros initializer.
    ///
    /// # Arguments
    ///
    /// * `num_filters` - The number of filters in the layer.
    /// * `kernel_size` - The height and width of the convolution kernels.
    /// * `stride` - The vertical and horizontal stride used for the convolution.
    /// * `padding` - The padding used for the convolution. Must be a variant of Padding.
    /// * `activation` - The activation function used by the layer.
    /// * `weights_initializer` - The initializer used to initialize the weights of the layer.
    /// * `biases_initializer` - The initializer used to initialize the biases of the layer.
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
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
            weights: Tensor::new_empty_tensor(),
            biases: Tensor::new_empty_tensor(),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer,
            biases_initializer,
            regularizer: None,
        })
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<Conv2D> {
        let activation = group.dataset("activation").and_then(|ds| ds.read_raw::<Activation>()).expect("Could not retrieve the activation function.");
        let kernel_size = group.dataset("kernel_size").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the kernel size.");
        let stride = group.dataset("stride").and_then(|ds| ds.read_raw::<[u64; 2]>()).expect("Could not retrieve the stride.");
        let padding = group.dataset("padding").and_then(|ds| ds.read_raw::<Padding>()).expect("Could not retrieve the padding.");
        let padding_size = group.dataset("padding_size").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the pading size.");
        let num_filters = group.dataset("num_filters").and_then(|ds| ds.read_raw::<u64>()).expect("Could not retrieve the number of filters.");
        let input_shape = group.dataset("input_shape").and_then(|value| value.read_raw::<[u64; 4]>()).expect("Could not retrieve the input shape.");
        let output_shape = group.dataset("output_shape").and_then(|value| value.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape.");
        let weights = group.dataset("weights").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve the weights.");
        let biases = group.dataset("biases").and_then(|ds| ds.read_raw::<H5Tensor>()).expect("Could not retrieve the biases.");
        let weights_initializer = group.dataset("weights_initializer").and_then(|ds| ds.read_raw::<H5Initializer>()).expect("Could not retrieve the weights initializer.");
        let biases_initializer = group.dataset("biases_initializer").and_then(|ds| ds.read_raw::<H5Initializer>()).expect("Could not retrieve the biases initializer.");
        let regularizer = Regularizer::from_hdf5_group(group);

        Box::new(Conv2D {
            activation: activation[0],
            kernel_size: (kernel_size[0][0], kernel_size[0][1]),
            stride: (stride[0][0], stride[0][1]),
            padding: padding[0],
            padding_size: (padding_size[0][0], padding_size[0][1], padding_size[0][2], padding_size[0][3]),
            num_filters: num_filters[0],
            input_shape: Dim::new(&input_shape[0]),
            output_shape: Dim::new(&output_shape[0]),
            weights: Tensor::from(&weights[0]),
            biases: Tensor::from(&biases[0]),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer: Initializer::from(&weights_initializer[0]),
            biases_initializer: Initializer::from(&biases_initializer[0]),
            regularizer,
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
        let mut conv = add(&matmul(&self.weights, &input_values, MatProp::NONE, MatProp::NONE), &self.biases, true);

        // Reshape to have each mini-batch on the last dimension
        conv = moddims(&conv, Dim4::new(&[self.num_filters, h_out * w_out, 1, batch_size]));

        // Reshape to have correct output dimensions
        let linear_activation = moddims(&transpose(&conv, false), Dim4::new(&[h_out, w_out, self.num_filters, batch_size]));
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

    /// Converts the image into a columns representation.
    ///
    /// This is done for computation speed but there is a memory cost.
    fn img_to_col(&self, input: &Tensor) -> Tensor {
        let num_channels = input.dims().get()[2];
        let mut col = unwrap(input, self.kernel_size.0 as i64, self.kernel_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true);
        //col = reorder(&col, Dim4::new(&[0, 2, 1, 3]));
        col = reorder_v2(&col, 0, 2, Some(vec![1, 3]));
        moddims(&col, Dim4::new(&[col.dims().get()[0] * num_channels, col.elements() as u64/(col.dims().get()[0] * num_channels), 1, 1]))
    }

    /// Transforms a columns representation of an image into an image with dimensions height x width x channels.
    fn col_to_img(&self, input: &Tensor) -> Tensor {
        let num_channels = self.input_shape.get()[2];
        let h_out = self.output_shape.get()[0];
        let w_out = self.output_shape.get()[1];
        let num_cols = h_out * w_out;
        let batch_size = input.dims().get()[1] / num_cols;
        let height_padded = (h_out - 1) * self.stride.0 + self.kernel_size.0;
        let width_padded = (w_out - 1) * self.stride.1 + self.kernel_size.1;

        let mut img = moddims(&input, Dim4::new(&[input.dims().get()[0], h_out*w_out, 1, batch_size]));
        //img = reorder(&img, Dim4::new(&[1, 0, 2, 3]));
        img = reorder_v2(&img, 1, 0, Some(vec![2, 3]));
        img = moddims(&img, Dim4::new(&[img.dims().get()[0], self.kernel_size.0 * self.kernel_size.1, num_channels, batch_size]));
        img = transpose(&img, false);
        img = wrap(&img, height_padded as i64, width_padded as i64, self.kernel_size.0 as i64, self.kernel_size.1 as i64, self.stride.0 as i64, self.stride.1 as i64, 0, 0, true);

        // Remove padding
        index(&img, &[Seq::new(self.padding_size.0 as f32, (height_padded - self.padding_size.2 - 1) as f32, 1.0), Seq::new(self.padding_size.3 as f32, (width_padded - self.padding_size.1 - 1) as f32, 1.0), Seq::default(), Seq::default()])
    }
}

impl Layer for Conv2D {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn initialize_parameters(&mut self, input_shape: Dim4) {
        let height = input_shape.get()[0];
        let width = input_shape.get()[1];
        let num_channels = input_shape.get()[2];

        // Compute output dimensions and padding size
        let (h_out, w_out) = match self.padding {
            Padding::Same => {
                ((height as f64 / self.stride.0 as f64).ceil() as u64, (width as f64 / self.stride.1 as f64).ceil() as u64)
            },
            Padding::Valid => {
                ((((height - self.kernel_size.0 + 1) as f64) / self.stride.0 as f64).ceil() as u64, (((width - self.kernel_size.1 + 1) as f64) / self.stride.1 as f64).ceil() as u64)
            }
        };
        self.compute_padding_size(height, width, h_out, w_out);

        let receptive_field = self.kernel_size.0 * self.kernel_size.1;
        let fan_in = receptive_field * num_channels;
        let fan_out = receptive_field * self.num_filters;
        self.output_shape = Dim4::new(&[h_out, w_out, self.num_filters, 1]);
        self.input_shape = input_shape;

        // Initialize weights and biases
        self.weights = self.weights_initializer.new_tensor(Dim4::new(&[self.num_filters, receptive_field * num_channels, 1, 1]), fan_in, fan_out);
        self.biases = self.biases_initializer.new_tensor(Dim4::new(&[self.num_filters, 1, 1, 1]), fan_in, fan_out);
    }

    fn compute_activation(&self, input: &Tensor) -> Tensor {
        let (linear_activation, _) = self.compute_convolution(input);
        linear_activation.eval();
        let nonlinear_activation = self.activation.eval(&linear_activation);
        nonlinear_activation.eval();
        nonlinear_activation
    }

    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor {
        let (linear_activation, reshaped_input) = self.compute_convolution(input);
        linear_activation.eval();
        reshaped_input.eval();
        self.reshaped_input = reshaped_input;

        let nonlinear_activation = self.activation.eval(&linear_activation);

        self.linear_activation = Some(linear_activation);
        self.previous_activation = Some(input.copy());

        nonlinear_activation
    }


    fn compute_dactivation_mut(&mut self, input: &Tensor) -> Tensor {
        match &self.linear_activation {
            Some(linear_activation) => {
                let mut linear_activation_grad = mul(input, &self.activation.grad(linear_activation), true);
                //linear_activation_grad = reorder(&linear_activation_grad, Dim4::new(&[2, 0, 1, 3]));
                linear_activation_grad = reorder_v2(&linear_activation_grad, 2, 0, Some(vec![1, 3]));
                linear_activation_grad = moddims(&linear_activation_grad, Dim4::new(&[self.num_filters, linear_activation_grad.elements() as u64 / self.num_filters, 1, 1]));

                self.dbiases = sum(&linear_activation_grad, 1) / input.dims().get()[3];

                let weights_grad = matmul(&linear_activation_grad, &self.reshaped_input, MatProp::NONE, MatProp::TRANS);
                self.dweights = weights_grad / input.dims().get()[3];
                if let Some(regularizer) = self.regularizer {  self.dweights += regularizer.grad(&self.weights) }

                let input_grad = matmul(&self.weights, &linear_activation_grad, MatProp::TRANS, MatProp::NONE);
                self.col_to_img(&input_grad)
            },
            None => panic!("The linear activations have not been computed!"),
        }
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


    fn save(&self, group: &hdf5::Group, layer_number: usize) -> Result<(), Error> {
        let group_name = layer_number.to_string() + &String::from("_") + Self::NAME;
        let conv2d = group.create_group(&group_name)?;

        let activation = conv2d.new_dataset::<Activation>().create("activation", 1)?;
        activation.write(&[self.activation])?;

        let kernel_size = conv2d.new_dataset::<[u64; 2]>().create("kernel_size", 1)?;
        kernel_size.write(&[[self.kernel_size.0, self.kernel_size.1]])?;

        let stride = conv2d.new_dataset::<[u64; 2]>().create("stride", 1)?;
        stride.write(&[[self.stride.0, self.stride.1]])?;

        let padding = conv2d.new_dataset::<Padding>().create("padding", 1)?;
        padding.write(&[self.padding])?;

        let padding_size = conv2d.new_dataset::<[u64; 4]>().create("padding_size", 1)?;
        padding_size.write(&[[self.padding_size.0, self.padding_size.1, self.padding_size.2, self.padding_size.3]])?;

        let num_filters = conv2d.new_dataset::<u64>().create("num_filters", 1)?;
        num_filters.write(&[self.num_filters])?;

        let input_shape = conv2d.new_dataset::<[u64; 4]>().create("input_shape", 1)?;
        input_shape.write(&[*self.input_shape.get()])?;

        let output_shape = conv2d.new_dataset::<[u64; 4]>().create("output_shape", 1)?;
        output_shape.write(&[*self.output_shape.get()])?;

        let weights = conv2d.new_dataset::<H5Tensor>().create("weights", 1)?;
        weights.write(&[ H5Tensor::from(&self.weights) ])?;

        let biases = conv2d.new_dataset::<H5Tensor>().create("biases", 1)?;
        biases.write(&[ H5Tensor::from(&self.biases) ])?;

        let weights_initializer = conv2d.new_dataset::<H5Initializer>().create("weights_initializer", 1)?;
        let biases_initializer = conv2d.new_dataset::<H5Initializer>().create("biases_initializer", 1)?;
        self.weights_initializer.save(&weights_initializer)?;
        self.biases_initializer.save(&biases_initializer)?;
        if let Some(regularizer) = self.regularizer { regularizer.save(&conv2d)?; }

        Ok(())
    }


    fn set_regularizer(&mut self, regularizer: Option<Regularizer>) {
        self.regularizer = regularizer;
    }

}

impl fmt::Display for Conv2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num_parameters = self.weights.elements() + self.biases.elements();
        write!(f, "{} \t\t {} \t\t [{}, {}, {}]", Self::NAME, num_parameters, self.output_shape[0], self.output_shape[1], self.output_shape[2])
    }
}

#[cfg(test)]
mod tests {
    use crate::layers::{Conv2D, Layer};
    use crate::layers::Padding;
    use crate::activations::Activation;
    use crate::initializers::Initializer;
    use crate::tensor::*;
    use crate::assert_approx_eq;
    use arrayfire::*;

    fn create_test_layer() -> Conv2D {
        let weights = transpose(&Tensor::new(&[1., 1., 1., 1., 2., 1., 1., 2., -1., -1., -1., -1., 1., 2., 1., 2., -2., -2., -2., -2., 1., 3., 3., 1.], Dim::new(&[12, 2, 1, 1])), false);
        Conv2D {
            activation: Activation::Linear,
            kernel_size: (2, 2),
            stride: (1, 1),
            padding: Padding::Valid,
            padding_size: (0, 0, 0, 0), // top, right, bottom, left
            num_filters: 2,
            input_shape: Dim::new(&[3, 3, 3, 1]),
            output_shape: Dim::new(&[2, 2, 2, 1]),
            weights,
            biases: Tensor::new(&[0., 0.], Dim::new(&[2, 1, 1, 1])),
            dweights: Tensor::new_empty_tensor(),
            dbiases: Tensor::new_empty_tensor(),
            linear_activation: None,
            previous_activation: None,
            reshaped_input: Tensor::new_empty_tensor(),
            weights_initializer: Initializer::HeUniform,
            biases_initializer: Initializer::Zeros,
            regularizer: None,
        }
    }

    fn create_test_images() -> Tensor {
        let mut images_vec = (1u8..=27 as u8).map(PrimitiveType::from).collect::<Vec<PrimitiveType>>();
        let image2_vec: [PrimitiveType; 27] = [4., -1., 2., 2., -3., 1., 6., 9., -10., 7., 5., -3., 1., -2., 4., -12., -21., 1., 2., 9., 8., -4., -3., 7., 1., 1., -2.];
        images_vec.extend(&image2_vec);
        Tensor::new(&images_vec, Dim::new(&[3, 3, 3, 2]))
    }

    #[test]
    fn test_conv2d_forward() {
        let mut layer = create_test_layer();
        let images = create_test_images();

        let layer_output = layer.compute_activation_mut(&images);
        let mut output: [PrimitiveType; 16] = [0.; 16];
        layer_output.host(&mut output);
        let expected_output: [PrimitiveType; 16] = [0., 6., 18., 24., 91., 97., 109., 115., 14., -9., -35., -25., -10., 25., 79., 43.];

        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn test_conv2d_input_gradient() {
        let mut layer = create_test_layer();
        let images = create_test_images();

        let _ = layer.compute_activation_mut(&images);

        let input_vec = [1., -4., -2., 1., -1., -3., 1., 2., -2., 1., 3., -1., 4., -1., 2., -4.];
        let input = Tensor::new(&input_vec, Dim::new(&[2, 2, 2, 2]));

        let layer_output = layer.compute_dactivation_mut(&input);
        let mut output: [PrimitiveType; 54] = [0.; 54];
        layer_output.host(&mut output);
        let expected_output: [PrimitiveType; 54] = [0., -8., -10., -1., -5., -5., -1., 3., 5., 4., 1., 2., -3., 0., -5., -4., -9., -2., -2., -3., -5., -1., -1., 6., 5., 8., 1., 2., 6., -1., 7., 8., -10., 5., 2., -9., -12., -6., 3., -8., -4., 11., -1., 9., 6., 6., 12., -4., 13., 2., -13., 3., -12., -3.];

        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn test_conv2d_weights_gradient() {
        let mut layer = create_test_layer();
        let images = create_test_images();

        let _ = layer.compute_activation_mut(&images);

        let input_vec = [1., -4., -2., 1., -1., -3., 1., 2., -2., 1., 3., -1., 4., -1., 2., -4.];
        let input = Tensor::new(&input_vec, Dim::new(&[2, 2, 2, 2]));

        let _ = layer.compute_dactivation_mut(&input);
        let dweights = layer.dweights;
        let mut output: [PrimitiveType; 24] = [0.; 24];
        dweights.host(&mut output);
        // Filters are stored along the first dimension
        // Input 1 individually
        //let expected_output: [PrimitiveType; 24] = [-10., 7., -14., 6., -22., 4., -26., 3., -46., -2., -50., -3., -58., -5., -62., -6., -82., -11., -86., -12., -94., -14., -98., -15.];
        // Input 2 individually
        //let expected_output: [PrimitiveType; 24] = [0., 33., -6., -16., 2., -13., 44., 45., -4., 33., -23., 3., -19., 66., -56., -58., -4., 3., -26., -6., 7., -15., 18., -9.];
        // Average of both
        let expected_output: [PrimitiveType; 24] = [-5., 20., -10., -5., -10., -4.5, 9., 24., -25., 15.5, -36.5, 0., -38.5, 30.5, -59., -32., -43., -4., -56., -9., -43.5, -14.5, -40., -12.];
        assert_approx_eq!(output, expected_output);
    }

    #[test]
    fn test_conv2d_biases_gradient() {
        let mut layer = create_test_layer();
        let images = create_test_images();

        let _ = layer.compute_activation_mut(&images);

        let input_vec = [1., -4., -2., 1., -1., -3., 1., 2., -2., 1., 3., -1., 4., -1., 2., -4.];
        let input = Tensor::new(&input_vec, Dim::new(&[2, 2, 2, 2]));


        let _ = layer.compute_dactivation_mut(&input);
        let dbiases = layer.dbiases;
        let mut output: [PrimitiveType; 2] = [0.; 2];
        dbiases.host(&mut output);
        let expected_output: [PrimitiveType; 2] = [-1.5, 0.];

        assert_approx_eq!(output, expected_output);
    }
}