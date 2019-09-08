use arrayfire::*;

pub struct BatchIterator {
    x_train: Vec<Array<f64>>,
    y_train: Vec<Array<f64>>,
    batch_size: u64,
    num_batches: u64,
    count: u64,
}


impl BatchIterator {
    pub fn new<L: super::DataSet>(data: &L, batch_size: u64) -> BatchIterator {
        let mut batch_size = batch_size;
        let mut num_batches= 0;
        if batch_size < data.num_train_samples() {
            num_batches = (data.num_train_samples() as f64 / batch_size as f64).ceil() as u64;
        } else {
            num_batches = 1;
            batch_size = data.num_train_samples();
        }

        BatchIterator {
            x_train: data.x_train().clone(),
            y_train: data.y_train().clone(),
            batch_size,
            num_batches,
            count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.count = 0;
    }
}


impl Iterator for BatchIterator {
    type Item = (Array<f64>, Array<f64>);

    fn next(&mut self) -> Option<Self::Item> {

        if self.count < self.num_batches {

            // Compute lower and upper bounds to retrieve samples
            let lb = (self.count * self.batch_size) as usize;
            let mut ub = ((self.count + 1) * self.batch_size) as usize;
            if ub >= self.x_train.len() {
                ub = self.x_train.len() - 1;
            }

            // Create mini-batch
            let mut mb_x = self.x_train[lb].copy();
            let mut mb_y = self.y_train[lb].copy();
            for i in (lb + 1)..ub as usize {
                mb_x = join(3, &mb_x, &self.x_train[i]);
                mb_y = join(3, &mb_y, &self.y_train[i]);
            }

            self.count += 1;
            Some((mb_x, mb_y))
        } else {
            None
        }
    }
}