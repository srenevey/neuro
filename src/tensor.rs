//! Wrapper around ArrayFire's array with additional helper methods.
use arrayfire::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// This type is defined to easily change between f32 and f64 as the primitive type used by the crate.
/// It has important consequences on the memory footprint of the crate when running deep and/or wide neural networks.
pub type PrimitiveType = f32;

/// Public type to hide ArrayFire from the user (so that they don't have to import it).
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
    /// Creates a tensor of ones.
    ///
    /// # Arguments
    /// * `dims`: dimensions of the tensor
    ///
    fn ones(dims: Dim4) -> Tensor;

    /// Creates a tensor of zeros.
    ///
    /// # Arguments
    /// * `dims`: dimensions of the tensor
    ///
    fn zeros(dims: Dim4) -> Tensor;

    /// Creates an empty tensor with no dimensions.
    fn new_empty_tensor() -> Tensor;

    /// Returns the number of samples in a batch.
    fn batch_size(&self) -> u64;

    /// Shuffles the last dimensions of two tensors.
    ///
    /// The indices permutation is identical for both tensors.
    ///
    /// # Arguments
    /// * `tensor1`: first tensor to shuffle
    /// * `tensor2`: second tensor to shuffle
    ///
    fn shuffle(tensor1: &Tensor, tensor2: &Tensor) -> (Tensor, Tensor);

    /// Shuffles the last dimensions of two tensors inplace.
    ///
    /// The indices permutation is identical for both tensors.
    ///
    /// # Arguments
    /// * `tensor1`: first tensor to shuffle
    /// * `tensor2`: second tensor to shuffle
    ///
    fn shuffle_mut(tensor1: &mut Tensor, tensor2: &mut Tensor);

    /// Creates a tensor with random entries drawn from a uniform distribution.
    ///
    /// # Arguments
    /// * `lower_bound`: lower bound of the distribution
    /// * `upper_bound`: upper bound of the distribution
    /// * `dims`: dimensions of the tensor
    ///
    fn scaled_uniform(lower_bound: PrimitiveType, upper_bound: PrimitiveType, dims: Dim4) -> Tensor;

    /// Creates a tensor with random entries drawn from a normal distribution.
    ///
    /// # Arguments
    /// * `mean`: mean of the distribution
    /// * `standard_deviation`: standard deviation of the distribution
    /// * `dims`: dimensions of the tensor
    ///
    fn scaled_normal(mean: PrimitiveType, standard_deviation: PrimitiveType, dims: Dim4) -> Tensor;

    /// Reduces the tensor.
    ///
    /// # Arguments
    /// * `reduction`: reduction method
    ///
    fn reduce(&self, reduction: Reduction) -> Tensor;

    /// Reshapes the tensor such that each sample is one-dimensional.
    ///
    /// For a tensor with dimensions [h, w, c, batch_size], the output tensor will have dimensions
    /// [hwc, 1, 1 batch_size].
    fn flatten(&self) -> Tensor;

    /// Reshapes the tensor inplace such that each sample is one-dimensional.
    ///
    /// For a tensor with dimensions [h, w, c, batch_size], the tensor will be modified to have
    ///  dimensions [hwc, 1, 1 batch_size].
    fn flatten_mut(&mut self);

    /// Reshapes the tensor to the given dimensions.
    ///
    /// # Arguments
    /// * `dims`: dimensions to reshape to
    ///
    fn reshape(&self, dims: Dim4) -> Tensor;

    /// Reshapes the tensor to the given dimensions inplace.
    ///
    /// # Arguments
    /// * `dims`: dimensions to reshape to
    ///
    fn reshape_mut(&mut self, dims: Dim4);

    // TODO: check if used
    fn print_tensor(&self);
    fn get_scalar(&self) -> PrimitiveType;
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
        self.reshape(dims)
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

    fn get_scalar(&self) -> PrimitiveType {
        let mut val: [PrimitiveType; 1] = [0.];
        self.host(&mut val);
        val[0]
    }
}