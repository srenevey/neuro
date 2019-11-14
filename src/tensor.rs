use arrayfire::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Public type definitions required to work with Tensor objects.
pub type PrimitiveType = f32;
pub type Tensor = Array<PrimitiveType>;
pub type Dim = Dim4;

pub enum Reduction {
    SumBatches,
    MeanBatches,
}

const BATCH_AXIS: usize = 3;

pub trait TensorTrait {
    fn ones(dims: Dim4) -> Tensor;
    fn zeros(dims: Dim4) -> Tensor;
    fn new_empty_tensor() -> Tensor;
    fn batch_size(&self) -> u64;
    fn shuffle(tensor1: &Tensor, tensor2: &Tensor) -> (Tensor, Tensor);
    fn shuffle_mut(tensor1: &mut Tensor, tensor2: &mut Tensor);
    fn scaled_uniform(lower_bound: PrimitiveType, upper_bound: PrimitiveType, dims: Dim4) -> Tensor;
    fn scaled_normal(mean: PrimitiveType, standard_deviation: PrimitiveType, dims: Dim4) -> Tensor;
    fn reduce(&self, reduction: Reduction) -> Tensor;
    fn flatten(&self) -> Tensor;
    fn flatten_mut(&mut self);
    fn reshape(&self, dims: Dim4) -> Tensor;
    fn reshape_mut(&mut self, dims: Dim4);
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