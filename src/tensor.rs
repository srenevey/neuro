//! Wrapper around ArrayFire's array with additional helper methods.
use arrayfire::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// This type is defined to easily change between f32 and f64 as the primitive type used by the crate.
/// It has important consequences on the memory footprint of the crate when running deep and/or wide neural networks.
pub type PrimitiveType = f32;

/// Type alias for ArrayFire's Array.
pub type Tensor = Array<PrimitiveType>;

/// Type alias for ArrayFire's Dim4.
pub type Dim = Dim4;

/// Defines reduction methods.
pub enum Reduction {
    SumBatches,
    MeanBatches,
}

const BATCH_AXIS: usize = 3;

/// Defines additional methods for the Tensor type.
pub trait TensorTrait {
    /// Creates a tensor of ones with the given dimensions.
    fn ones(dims: Dim4) -> Tensor;

    /// Creates a tensor of zeros with the given dimensions.
    fn zeros(dims: Dim4) -> Tensor;

    /// Creates an empty tensor with no dimensions.
    fn new_empty_tensor() -> Tensor;

    /// Returns the number of samples in a batch.
    fn batch_size(&self) -> u64;

    /// Shuffles two vectors with identical indices permutation along the last dimension.
    fn shuffle(tensor1: &Tensor, tensor2: &Tensor) -> (Tensor, Tensor);

    /// Shuffles two vectors with identical indices permutation along the last dimension inplace.
    fn shuffle_mut(tensor1: &mut Tensor, tensor2: &mut Tensor);

    /// Creates a tensor with the given dimensions where each entry is drawn from a uniform distribution.
    fn scaled_uniform(lower_bound: PrimitiveType, upper_bound: PrimitiveType, dims: Dim4) -> Tensor;

    /// Creates a tensor with the given dimensions where each entry is drawn from a normal distribution.
    fn scaled_normal(mean: PrimitiveType, standard_deviation: PrimitiveType, dims: Dim4) -> Tensor;

    /// Reduces the tensor.
    fn reduce(&self, reduction: Reduction) -> Tensor;

    /// Reshapes the tensor such that each sample is one-dimensional.
    ///
    /// For a tensor with dimensions [h, w, c, batch_size], the output tensor will have dimensions
    /// [hwc, 1, 1 batch_size].
    fn flatten(&self) -> Tensor;

    /// Reshapes the tensor inplace such that each sample is one-dimensional.
    ///
    /// For a tensor with dimensions [h, w, c, batch_size], the tensor will be modified to have
    /// dimensions [hwc, 1, 1 batch_size].
    fn flatten_mut(&mut self);

    /// Reshapes the tensor to the given dimensions.
    fn reshape(&self, dims: Dim4) -> Tensor;

    /// Reshapes the tensor to the given dimensions inplace.
    fn reshape_mut(&mut self, dims: Dim4);

    fn print_tensor(&self);
}

impl TensorTrait for Tensor {
    fn ones(dims: Dim4) -> Tensor {
        constant(1.0 as PrimitiveType, dims)
    }

    fn zeros(dims: Dim4) -> Tensor {
        constant(0.0 as PrimitiveType, dims)
    }

    fn new_empty_tensor() -> Tensor {
        Array::new_empty(Dim4::new(&[0, 0, 0, 0]))
    }

    fn batch_size(&self) -> u64 {
        self.dims().get()[BATCH_AXIS]
    }

    fn shuffle(x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        assert_eq!(x.batch_size(), y.batch_size());

        // Shuffle indices
        let mut indices: Vec<u64> = (0..x.batch_size()).collect();
        indices.shuffle(&mut thread_rng());

        let indices_arr = Array::new(&indices[..], Dim4::new(&[x.batch_size(), 1, 1, 1]));

        let x_shuffled = lookup(x, &indices_arr, BATCH_AXIS as i32);
        let y_shuffled = lookup(y, &indices_arr, BATCH_AXIS as i32);
        (x_shuffled, y_shuffled)
    }

    fn shuffle_mut(x: &mut Tensor, y: &mut Tensor) {
        assert_eq!(x.batch_size(), y.batch_size());

        // Shuffle indices
        let mut indices: Vec<u64> = (0..x.batch_size()).collect();
        indices.shuffle(&mut thread_rng());
        let indices_arr = Array::new(&indices[..], Dim4::new(&[x.batch_size(), 1, 1, 1]));

        *x = lookup(x, &indices_arr, BATCH_AXIS as i32);
        *y = lookup(y, &indices_arr, BATCH_AXIS as i32);
    }

    fn scaled_uniform(lower_bound: PrimitiveType, upper_bound: PrimitiveType, dims: Dim4) -> Tensor {
        constant(lower_bound, dims) + constant(upper_bound - lower_bound, dims) * randu::<PrimitiveType>(dims)
    }

    fn scaled_normal(mean: PrimitiveType, standard_deviation: PrimitiveType, dims: Dim4) -> Tensor {
        constant(standard_deviation, dims) * randn::<PrimitiveType>(dims) + constant(mean, dims)
    }

    fn reduce(&self, reduction: Reduction) -> Tensor
    {
        match reduction {
            Reduction::SumBatches => { sum(self, BATCH_AXIS as i32)},
            Reduction::MeanBatches => { mean(self, BATCH_AXIS as i64)},
        }
    }

    fn flatten(&self) -> Tensor {
        let dim0 = self.dims()[0];
        let dim1 = self.dims()[1];
        let dim2 = self.dims()[2];
        let dims = Dim4::new(&[dim0 * dim1 * dim2, 1, 1, self.batch_size()]);
        moddims(self, dims)
    }

    fn flatten_mut(&mut self) {
        let dim0 = self.dims()[0];
        let dim1 = self.dims()[1];
        let dim2 = self.dims()[2];
        let dims = Dim4::new(&[dim0 * dim1 * dim2, 1, 1, self.batch_size()]);
        self.reshape_mut(dims);
    }

    fn reshape(&self, dims: Dim4) -> Tensor {
        moddims(self, dims)
    }

    fn reshape_mut(&mut self, dims: Dim4) {
        *self = moddims(self, dims);
    }

    fn print_tensor(&self) {
        print(self);
    }
}



#[derive(hdf5::H5Type, Clone, Debug)]
#[repr(C)]
pub(crate) struct H5Tensor {
    dims: [u64; 4],
    values: hdf5::types::VarLenArray<PrimitiveType>,
}

impl From<Tensor> for H5Tensor {
    fn from(tensor: Tensor) -> Self {
        let mut buffer = vec![0.; tensor.elements()];
        tensor.host(&mut buffer);
        let dims = tensor.dims();

        H5Tensor {
            dims: [dims[0], dims[1], dims[2], dims[3]],
            values: hdf5::types::VarLenArray::from_slice(&buffer[..]),
        }
    }
}

impl From<&Tensor> for H5Tensor {
    fn from(tensor: &Tensor) -> Self {
        let mut buffer = vec![0.; tensor.elements()];
        tensor.host(&mut buffer);
        let dims = tensor.dims();

        H5Tensor {
            dims: [dims[0], dims[1], dims[2], dims[3]],
            values: hdf5::types::VarLenArray::from_slice(&buffer[..]),
        }
    }
}

impl From<H5Tensor> for Tensor {
    fn from(h5_tensor: H5Tensor) -> Self {
        Tensor::new(h5_tensor.values.as_slice(), Dim::new(&h5_tensor.dims))
    }
}

impl From<&H5Tensor> for Tensor {
    fn from(h5_tensor: &H5Tensor) -> Self {
        Tensor::new(h5_tensor.values.as_slice(), Dim::new(&h5_tensor.dims))
    }
}