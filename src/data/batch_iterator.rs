use arrayfire::*;

use crate::tensor::*;

pub struct BatchIterator<'a> {
    data: (&'a Tensor, &'a Tensor),
    num_samples: u64,
    batch_size: u64,
    batch: u64,
    num_batches: u64
}

impl<'a> BatchIterator<'a> {

    /// Creates a batch iterator of given size for the two Tensors.
    ///
    /// # Arguments
    /// * `data` - tuple of reference to the Tensors.
    /// * `batch_size` - size of the mini-batches
    ///
    pub fn new(data: (&'a Tensor, &'a Tensor), batch_size: u64) -> BatchIterator<'a> {
        // Check that both tensors have the same number of samples
        assert_eq!(data.0.dims().get()[3], data.1.dims().get()[3]);
        let num_samples = data.0.dims().get()[3];

        let (batch_size, num_batches) = if batch_size < num_samples {
            let num_batches = (num_samples as f64 / batch_size as f64).ceil() as u64;
            (batch_size, num_batches)
        } else {
            (num_samples, 1)
        };

        BatchIterator {
            data,
            num_samples,
            batch_size,
            batch: 0,
            num_batches
        }
    }

    /// Returns the number of batches that the iterator will produce.
    pub(crate) fn num_batches(&self) -> u64 {
        self.num_batches
    }
}

impl<'a> std::iter::Iterator for BatchIterator<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch < self.num_batches {
            // Compute lower and upper bounds to retrieve samples
            let lb = (self.batch * self.batch_size) as usize;
            let mut ub = ((self.batch + 1) * self.batch_size - 1) as usize;
            if ub >= self.num_samples as usize {
                ub = (self.num_samples - 1) as usize;
            }

            // Create mini-batches
            let seqs = &[Seq::default(), Seq::default(), Seq::default(), Seq::new(lb as f64, ub as f64, 1.0)];
            let mini_batch_x = index(&self.data.0, seqs);
            let mini_batch_y = index(&self.data.1, seqs);

            self.batch += 1;

            Some((mini_batch_x, mini_batch_y))
        } else {
            None
        }
    }
}