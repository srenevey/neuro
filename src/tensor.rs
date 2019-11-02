use arrayfire::*;
use rand::thread_rng;
use rand::seq::SliceRandom;

pub(crate) type PrimitiveType = f32;
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
    fn shuffle_mut(tensor1: &mut Tensor, tensor2: &mut Tensor);
    fn scaled_uniform(lower_bound: PrimitiveType, upper_bound: PrimitiveType, dims: Dim4) -> Tensor;
    fn scaled_normal(mean: PrimitiveType, standard_deviation: PrimitiveType, dims: Dim4) -> Tensor;
    fn reduce(&self, reduction: Reduction) -> Tensor;
    fn flatten(&self) -> Tensor;
    fn flatten_mut(&mut self);
    fn reshape(&self, dims: Dim4) -> Tensor;
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

    fn shuffle_mut(x: &mut Tensor, y: &mut Tensor) {
        assert_eq!(x.batch_size(), y.batch_size());

        // Shuffle indices
        let mut indices: Vec<u64> = (0..x.batch_size()).collect();
        indices.shuffle(&mut thread_rng());

        let indices_array = Array::new(&indices[..], Dim4::new(&[x.batch_size(), 1, 1, 1]));
        let seqs: Seq<f64> = Seq::default();

        let mut idxrs_x = Indexer::new();
        idxrs_x.set_index(&seqs, 0, Some(false));
        idxrs_x.set_index(&seqs, 1, Some(false)); // 3rd parameter indicates batch operation
        idxrs_x.set_index(&seqs, 2, Some(false));
        idxrs_x.set_index(&indices_array, BATCH_AXIS as u32, None); // 2nd parameter is indexing dimension
        *x = index_gen(&x, idxrs_x);

        let mut idxrs_y = Indexer::new();
        idxrs_y.set_index(&seqs, 0, Some(false));
        idxrs_y.set_index(&seqs, 1, Some(false)); // 3rd parameter indicates batch operation
        idxrs_y.set_index(&seqs, 2, Some(false));
        idxrs_y.set_index(&indices_array, BATCH_AXIS as u32, None); // 2nd parameter is indexing dimension
        *y = index_gen(&y, idxrs_y);
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
}



/*
pub(crate) type PrimitiveType = f32;
pub struct Tensor(Array<PrimitiveType>);
pub type Dim = [u64; 4];


impl Tensor
{
    pub fn new(values: &[PrimitiveType], dims: &Dim) -> Tensor {
        Tensor(Array::new(values, Dim4::new(dims)))
    }

    pub fn new_empty() -> Tensor {
        Tensor(Array::new_empty(Dim4::new(&[0, 0, 0, 0])))
    }

    pub fn from_array(array: Array<PrimitiveType>) -> Tensor {
        Tensor(array)
    }

    pub fn copy(&self) -> Tensor {
        Tensor(self.0.copy())
    }

    pub fn constant(values: PrimitiveType, dims: &Dim) -> Tensor {
        Tensor(constant(values, Dim4::new(dims)))
    }

    pub fn zeros(dims: &Dim) -> Tensor {
        Tensor(constant(<PrimitiveType>::from(0.), Dim4::new(dims)))
    }

    pub fn ones(dims: &Dim) -> Tensor {
        Tensor(constant::<PrimitiveType>(<PrimitiveType>::from(1.), Dim4::new(dims)))
    }

    pub fn randn(dims: &Dim) -> Tensor
    {
        Tensor(randn::<PrimitiveType>(Dim4::new(dims)))
    }

    pub fn randu(dims: &Dim) -> Tensor
    {
        Tensor(randu::<PrimitiveType>(Dim4::new(dims)))
    }

    pub fn scaled_uniform(lower_bound: PrimitiveType, upper_bound: PrimitiveType, dims: &Dim) -> Tensor {
        Tensor(constant(lower_bound, Dim4::new(dims)) + constant(2. * (upper_bound - lower_bound), Dim4::new(dims)) * randu::<PrimitiveType>(Dim4::new(dims)))
    }

    pub fn scaled_normal(mean: PrimitiveType, standard_deviation: PrimitiveType, dims: &Dim) -> Tensor {
        Tensor(constant(standard_deviation, Dim4::new(dims)) * randn(Dim4::new(dims)) + constant(mean, Dim4::new(dims)))
        //mul(&(6. / (fan_out + fan_in) as f64).sqrt(), &randn::<f64>(dims), false)
    }

    pub fn grad_relu(input: &Tensor) -> Tensor {
        let cond = ge(input, &0 as PrimitiveType, true);
        Tensor(selectr(&constant(1 as PrimitiveType, input.dims()), &cond, 0.0))
    }

    pub fn grad_leakyrelu(input: &Tensor) -> Tensor {
        let cond = ge(input, &0 as PrimitiveType, true);
        selectr(&constant(1 as PrimitiveType, input.dims()), &cond, 0.01)
    }

    pub fn sigmoid(input: &Tensor) -> Tensor {
        Tensor(sigmoid(input))
    }

    pub fn tanh(input: &Tensor) -> Tensor {
        Tensor(tanh(input))
    }

    pub fn exp(input: &Tensor) -> Tensor {
        Tensor(exp(input))
    }

    pub fn log(input: &Tensor) -> Tensor {
        Tensor(log(input))
    }

    pub fn reduce(&self, reduction: Reduction) -> Tensor
    {
        match reduction {
            Reduction::SumBatches => { Tensor(sum(self, BATCH_AXIS as i32))},
            Reduction::MeanBatches => { Tensor(mean(self, BATCH_AXIS as i64))},
        }
    }

    pub fn sum(input: &Tensor, dim: u64) -> Tensor {
        Tensor(sum(input, dim as i32))
    }

    pub fn sum_all(input: &Tensor) -> (f64, f64) {
        sum_all(input)
    }

    pub fn mean(input: &Tensor, dim: u64) -> Tensor {
        Tensor(mean(input, dim as i64))
    }

    pub fn mean_all(input: &Tensor) -> (f64, f64) {
        mean_all(input)
    }

    pub fn abs(input: &Tensor) -> Tensor {
        Tensor(abs(input))
    }

    pub fn grad_abs(input: &Tensor) -> Tensor {
        let cond = ge(input, 0., true);
        Tensor(selectr(&constant(&1.0,Dim4::new(input.dims())), &cond, -1.0f64))
    }

    pub fn pow(input: &Tensor, power: f64, batch: bool) -> Tensor {
        Tensor(pow(input, power, batch))
    }

    pub fn shuffle_batches(&mut self) {
        // Shuffle indices
        let mut indices: Vec<u64> = (0..self.batch_size()).collect();
        indices.shuffle(&mut thread_rng());

        let indices_array = Array::new(&indices[..], Dim4::new(&[self.batch_size(), 1, 1, 1]));
        let seqs: Seq<f64> = Seq::default();

        let mut idxrs = Indexer::new();
        idxrs.set_index(&indices_array, BATCH_AXIS as u32, None); // 2nd parameter is indexing dimension
        idxrs.set_index(&seqs, 1, Some(true));
        self.0 = index_gen(&self.0, idxrs);
    }

    pub fn shuffle_mut(x: &mut Tensor, y: &mut Tensor) {
        assert_eq!(x.batch_size(), y.batch_size());

        // Shuffle indices
        let mut indices: Vec<u64> = (0..x.batch_size()).collect();
        indices.shuffle(&mut thread_rng());

        let indices_array = Array::new(&indices[..], Dim4::new(&[x.batch_size(), 1, 1, 1]));
        let seqs: Seq<f64> = Seq::default();

        let mut idxrs_x = Indexer::new();
        idxrs_x.set_index(&seqs, 0, Some(false));
        idxrs_x.set_index(&seqs, 1, Some(false)); // 3rd parameter indicates batch operation
        idxrs_x.set_index(&seqs, 2, Some(false));
        idxrs_x.set_index(&indices_array, BATCH_AXIS as u32, None); // 2nd parameter is indexing dimension
        x.0 = index_gen(&x.0, idxrs_x);

        let mut idxrs_y = Indexer::new();
        idxrs_y.set_index(&seqs, 0, Some(false));
        idxrs_y.set_index(&seqs, 1, Some(false)); // 3rd parameter indicates batch operation
        idxrs_y.set_index(&seqs, 2, Some(false));
        idxrs_y.set_index(&indices_array, BATCH_AXIS as u32, None); // 2nd parameter is indexing dimension
        y.0 = index_gen(&y.0, idxrs_y);
    }

    pub fn batch_size(&self) -> u64 {
        self.dims()[BATCH_AXIS]
    }

    pub fn dims(&self) -> &Dim {
        self.0.dims().get()
    }

    pub fn flatten(&self) -> Tensor {
        let dim0 = self.dims()[0];
        let dim1 = self.dims()[1];
        let dim2 = self.dims()[2];
        let dims = [dim0 * dim1 * dim2, 1, 1, self.batch_size()];
        self.reshape(&dims)
        //self.0 = moddims(self, Dim4::new(&[dim0 * dim1 * dim2, 1, 1, self.batch_size()]));
    }

    pub fn reshape(&self, dims: &Dim) -> Tensor {
        Tensor(moddims(self, Dim4::new(dims)))
    }

    pub fn flatten_mut(&mut self) {
        let dim0 = self.dims()[0];
        let dim1 = self.dims()[1];
        let dim2 = self.dims()[2];
        let dims = [dim0 * dim1 * dim2, 1, 1, self.batch_size()];
        self.reshape(&dims);
        //self.0 = moddims(self, Dim4::new(&[dim0 * dim1 * dim2, 1, 1, self.batch_size()]));
    }

    pub fn reshape_mut(&mut self, dims: &Dim) {
        self.0 = moddims(self, Dim4::new(dims));
    }

    pub fn add(tensor1: &Tensor, tensor2: &Tensor, batch: bool) -> Tensor {
        Tensor(add(tensor1, tensor2, batch))
    }

    pub fn sub(tensor1: &Tensor, tensor2: &Tensor, batch: bool) -> Tensor {
        Tensor(sub(tensor1, tensor2, batch))
    }

    pub fn mul(tensor1: &Tensor, tensor2: &Tensor, batch: bool) -> Tensor {
        Tensor(mul(&tensor1, &tensor2, batch))
    }

    pub fn matmul(tensor1: &Tensor, tensor2: &Tensor, transpose_t1: bool, transpose_t2: bool) -> Tensor {
        let transpose1 = if transpose_t1 { MatProp::TRANS } else { MatProp::NONE};
        let transpose2 = if transpose_t2 { MatProp::TRANS } else { MatProp::NONE};
        Tensor(matmul(tensor1, tensor2, transpose1, transpose2))
    }

    pub fn div(tensor1: &Tensor, tensor2: &Tensor, batch: bool) -> Tensor {
        Tensor(div(tensor1, tensor2, batch))
    }

    pub fn max(input: &Tensor, dim: u64) -> Tensor {
        Tensor(max(input, dim as i32))
    }

    pub fn maxof(tensor1: &Tensor, tensor2: &Tensor, batch: bool) -> Tensor {
        Tensor(maxof(tensor1, tensor1, batch))
    }
}


impl Deref for Tensor
{
    type Target = Array<PrimitiveType>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        sub(self, rhs, true)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        sub(self, rhs, true)
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        mul(self, rhs, true)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        mul(self, rhs, true)
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        div(self, rhs, true)
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        div(self, rhs, true)
    }
}

*/

/*
impl<T> Tensor<T>
    where T: HasAfEnum + ConstGenerator<OutType=T> + FromPrimitive
{
    pub fn new(values: &[T], dims: &[u64; 4]) -> Tensor<T> {
        Tensor(Array::new(values, Dim4::new(dims)))
    }

    pub fn from_array(array: Array<T>) -> Tensor<T> {
        Tensor(array)
    }

    pub fn constant(values: T, dims: &[u64; 4]) -> Tensor<T> {
        Tensor(constant::<T>(values, Dim4::new(dims)))
    }

    pub fn zeros(dims: &[u64; 4]) -> Tensor<T> {
        Tensor(constant::<T>(T::from_i32(0).unwrap(), Dim4::new(dims)))
    }

    pub fn ones(dims: &[u64; 4]) -> Tensor<T> {
        Tensor(constant::<T>(T::from_i32(1).unwrap(), Dim4::new(dims)))
    }

    pub fn randn(dims: &[u64; 4]) -> Tensor<T>
        where T: FloatingPoint
    {
        Tensor(randn::<T>(Dim4::new(dims)))
    }

    pub fn randu(dims: &[u64; 4]) -> Tensor<T>
        where T: FloatingPoint
    {
        Tensor(randu::<T>(Dim4::new(dims)))
    }

    pub fn reduce(&self, reduction: Reduction) -> Tensor<T>
        where T: HasAfEnum<AggregateOutType=T, MeanOutType=T>
    {
        match reduction {
            Reduction::SumBatches => { Tensor(sum(self, 0))},
            Reduction::MeanBatches => { Tensor(mean(self, 0))},
        }
    }

    pub fn shuffle_batches(&mut self) {
        // Shuffle indices
        let mut indices: Vec<u64> = (0..self.batch_size()).collect();
        indices.shuffle(&mut thread_rng());

        let indices_array = Array::new(&indices[..], Dim4::new(&[self.batch_size(), 1, 1, 1]));
        let seqs: Seq<f64> = Seq::default();

        let mut idxrs = Indexer::new();
        idxrs.set_index(&indices_array, BATCH_AXIS as u32, None); // 2nd parameter is indexing dimension
        idxrs.set_index(&seqs, 1, Some(true));
        self.0 = index_gen(&self.0, idxrs);
    }

    pub fn shuffle_mut(x: &mut Tensor<T>, y: &mut Tensor<T>) {
        assert_eq!(x.batch_size(), y.batch_size());

        // Shuffle indices
        let mut indices: Vec<u64> = (0..x.batch_size()).collect();
        indices.shuffle(&mut thread_rng());

        let indices_array = Array::new(&indices[..], Dim4::new(&[x.batch_size(), 1, 1, 1]));
        let seqs: Seq<f64> = Seq::default();

        let mut idxrs_x = Indexer::new();
        idxrs_x.set_index(&indices_array, BATCH_AXIS as u32, None); // 2nd parameter is indexing dimension
        idxrs_x.set_index(&seqs, 1, Some(true));
        x.0 = index_gen(&x.0, idxrs_x);

        let mut idxrs_y = Indexer::new();
        idxrs_y.set_index(&indices_array, BATCH_AXIS as u32, None); // 2nd parameter is indexing dimension
        idxrs_y.set_index(&seqs, 1, Some(true));
        y.0 = index_gen(&y.0, idxrs_y);
    }

    pub fn batch_size(&self) -> u64 {
        self.dims().get()[BATCH_AXIS]
    }
}


impl<T> Deref for Tensor<T>
    where T: HasAfEnum
{
    type Target = Array<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
*/
