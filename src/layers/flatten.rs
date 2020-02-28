use hdf5::Group;
use std::fmt;

use crate::errors::Error;
use crate::layers::Layer;
use crate::tensor::*;

pub struct Flatten {
    input_shape: Dim,
    output_shape: Dim,
}

impl Flatten {
    pub(crate) const NAME: &'static str = "Flatten";

    pub fn new() -> Box<Flatten> {
        Box::new(Flatten {
            input_shape: Dim::new(&[0, 0, 0, 0]),
            output_shape: Dim::new(&[0, 0, 0, 0]),
        })
    }

    pub(crate) fn from_hdf5_group(group: &hdf5::Group) -> Box<Flatten> {
        let input_shape = group.dataset("input_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the input shape.");
        let output_shape = group.dataset("output_shape").and_then(|ds| ds.read_raw::<[u64; 4]>()).expect("Could not retrieve the output shape.");

        Box::new(Flatten {
            input_shape: Dim::new(&input_shape[0]),
            output_shape: Dim::new(&output_shape[0]),
        })
    }

}

impl Layer for Flatten {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn initialize_parameters(&mut self, input_shape: Dim) {
        self.input_shape = input_shape;
        self.output_shape = Dim::new(&[input_shape.get()[0] * input_shape.get()[1] * input_shape.get()[2], 1, 1, 1]);
    }

    fn compute_activation(&self, input: &Tensor) -> Tensor {
        input.flatten()
    }

    fn compute_activation_mut(&mut self, input: &Tensor) -> Tensor {
        input.flatten()
    }

    fn compute_dactivation_mut(&mut self, input: &Tensor) -> Tensor {
        input.reshape(Dim::new(&[self.input_shape.get()[0], self.input_shape.get()[1], self.input_shape.get()[2], input.dims().get()[3]]))
    }

    fn output_shape(&self) -> Dim {
        self.output_shape
    }

    fn save(&self, group: &Group, layer_number: usize) -> Result<(), Error> {
        let group_name = layer_number.to_string() + &String::from("_") + Self::NAME;
        let flatten = group.create_group(&group_name)?;

        let input_shape = flatten.new_dataset::<[u64; 4]>().create("input_shape", 1)?;
        input_shape.write(&[*self.input_shape.get()])?;

        let output_shape = flatten.new_dataset::<[u64; 4]>().create("output_shape", 1)?;
        output_shape.write(&[*self.output_shape.get()])?;

        Ok(())
    }
}

impl fmt::Display for Flatten {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} \t 0 \t\t [{}, {}, {}]", Self::NAME, self.output_shape[0], self.output_shape[1], self.output_shape[2])
    }
}