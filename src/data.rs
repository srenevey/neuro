use arrayfire::*;
use std::path::Path;
use csv::Reader;
use std::io;
use std::error;
use std::fmt;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Debug)]
pub enum DataSetError {
    Io(io::Error),
    Csv(csv::Error),
    DimensionMismatch,
}

impl fmt::Display for DataSetError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DataSetError::Io(ref err) => write!(f, "IO error: {}", err),
            DataSetError::Csv(ref err) => write!(f, "CSV error: {}", err),
            DataSetError::DimensionMismatch => write!(f, "The number of input and output samples differ.")
        }
    }
}

pub(crate) trait DataSet {
    fn num_features(&self) -> u64;
    fn set_batch_size(&mut self, batch_size: u64);
    fn mini_batch(&self) -> &BatchIterator;
}

struct BatchIterator {
    batch_size: u64,
    num_batches: u64,
    count: u64,
}

impl BatchIterator {
    fn new() -> BatchIterator {
        BatchIterator {
            batch_size: 0,
            num_batches: 0,
            count: 0,
        }
    }

    fn set_batch_size(&mut self, batch_size: u64, num_batches: u64) {
        self.batch_size = batch_size;
        self.num_batches = num_batches;
    }
}

impl Iterator for BatchIterator {
    type Item = (Array<f64>, Array<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;

        if self.count < self.num_batches {
            Some()
        } else {
            None
        }
    }
}

pub struct TabularDataSet {
    num_features: u64,
    num_training_samples: u64,
    x_train: Vec<f64>,
    y_train: Vec<f64>,
    x_valid: Vec<f64>,
    y_valid: Vec<f64>,
    x_test: Option<Vec<f64>>,
    y_test: Option<Vec<f64>>,
    x_train_stats: Option<(Vec<f64>, Vec<f64>)>,
    y_train_stats: Option<(Vec<f64>, Vec<f64>)>,
    batch_iterator: BatchIterator,
}

impl TabularDataSet {
    pub fn from_csv(inputs: &Path, outputs: &Path, valid_frac: f64) -> Result<TabularDataSet, DataSetError> {
        let (in_shape, num_in_samples, in_values) = DataSet::load_data_from_path(&inputs)?;
        let (out_shape, num_out_samples, out_values) = DataSet::load_data_from_path(&outputs)?;

        if num_in_samples != num_out_samples {
            Err(DataSetError::DimensionMismatch)
        } else {
            // Shuffle the samples
            let mut rng = thread_rng();
            let mut sample_id: Vec<u64> = (0..num_in_samples).collect();
            sample_id.shuffle(&mut rng);

            let mut x = Vec::<f64>::new();
            let mut y = Vec::<f64>::new();
            for i in sample_id.iter() {
                for j in 0..in_shape {
                    x.push(in_values[(i+j) as usize]);
                    y.push(out_values[(i+j) as usize]);
                }
            }

            // Compute number of samples in training set and in validation set
            let num_valid_samples = (valid_frac * num_in_samples as f64).floor() as u64;
            let num_train_samples = num_in_samples - num_valid_samples;
            let idx_in = (num_train_samples * in_shape) as usize;
            let idx_out = (num_train_samples * out_shape) as usize;

            // Create the data set
            Ok(TabularDataSet {
                num_training_samples: num_train_samples,
                num_features: in_shape,
                x_train: x[0..idx_in].to_vec(),
                y_train: y[0..idx_out].to_vec(),
                x_valid: x[idx_in..].to_vec(),
                y_valid: y[idx_out..].to_vec(),
                x_test: None,
                y_test: None,
                x_train_stats: None,
                y_train_stats: None,
                batch_iterator: BatchIterator::new(),
            })
        }
    }

    fn load_data_from_path(path: &Path) -> Result<(u64, u64, Vec<f64>), DataSetError> {
        let mut reader = Reader::from_path(path);
        match reader {
            Ok(mut rdr) => {
                let mut values = Vec::<f64>::new();
                let mut input_shape = 0;
                for (i, result) in rdr.records().enumerate() {
                    let record = result.unwrap();
                    if i == 0 {
                        input_shape = record.len() as u64;
                    }
                    for entry in record.iter() {
                        values.push((*entry).parse::<f64>().unwrap());
                    }
                }

                let num_samples = values.len() as u64 / input_shape;
                Ok((input_shape, num_samples, values))
            },
            Err(e) => Err(DataSetError::Csv(e))
        }
    }
}

impl DataSet for TabularDataSet {
    fn num_features(&self) -> u64 {
        self.num_features
    }

    fn set_batch_size(&mut self, batch_size: u64) {
        let num_batches = (self.num_training_samples / batch_size).floor() as u64;
        self.batch_iterator.set_batch_size(batch_size, num_batches);
    }

    fn mini_batch(&self) -> &BatchIterator {
        &self.batch_iterator
    }
}

pub struct ImageDataSet {

}

/*
pub struct DataSet {
    num_features: Dim4,
    num_training_samples: u64,
    x_train: Array<f64>,
    y_train: Array<f64>,
    x_valid: Array<f64>,
    y_valid: Array<f64>,
    x_test: Option<Array<f64>>,
    y_test: Option<Array<f64>>,
    x_train_stats: Option<(Array<f64>, Array<f64>)>,
    y_train_stats: Option<(Array<f64>, Array<f64>)>,
}

impl DataSet {

    /// Create a dataset from features and labels.
    ///
    /// # Arguments
    /// `features: array containing the features. The last dimension corresponds to the number of samples.
    /// `labels: array containing the labels. The last dimension corresponds to the number of samples.
    /// `valid_frac: fraction of the dataset to be used for validation.
    ///
    pub fn new(features: Array<f64>, labels: Array<f64>, valid_frac: f64) -> Result<DataSet, DataSetError> {
        if features.dims()[3] == labels.dims()[3] {
            Ok(
            DataSet {
                num_features: Dim4::new(&[features.dims()[0], features.dims()[1], features.dims()[2], 1]),
                num_training_samples: features.dims()[3],
                x_train: features,
                y_train: labels,
                x_valid: constant(0.0f64, Dim4::new(&[0, 0, 0, 0])),
                y_valid: constant(0.0f64, Dim4::new(&[0, 0, 0, 0])),
                x_test: None,
                y_test: None,
                x_train_stats: None,
                y_train_stats: None,
            })
        } else {
            Err(DataSetError::DimensionMismatch)
        }

    }

    /// Create a dataset from a folder architecture.
    ///
    /// # Arguments
    /// `path: the path to a folder containing the data. The folder must contain the subfolders train, valid, and optionally test.
    ///
    pub fn from_folder(path: &Path) -> DataSet {

        // Look if folder contains train, valid, and optionally test
        let train = path.join("train");
        let valid = path.join("valid");
        let test = path.join("test");

        if train.exists() && valid.exists() {

            let x_test = None;
            let y_test = None;

            if test.exists() {

            }

            DataSet {
                num_features: Dim4::new(&[0, 0, 0, 0]),
                num_training_samples: 0,
                x_train: constant(0.0f64, Dim4::new(&[0, 0, 0, 0])),
                y_train: constant(0.0f64, Dim4::new(&[0, 0, 0, 0])),
                x_valid: constant(0.0f64, Dim4::new(&[0, 0, 0, 0])),
                y_valid: constant(0.0f64, Dim4::new(&[0, 0, 0, 0])),
                x_test,
                y_test,
                x_train_stats: None,
                y_train_stats: None,
            }

        } else {
            panic!("The 'train' and/or 'valid' subfolders don't exist.");
        }
    }

    /// Create a dataset from csv files.
    ///
    /// # Arguments
    /// `inputs: path to a csv file containing the inputs
    /// `outputs: path to a csv file containing the outputs
    /// `valid_frac: fraction of the dataset to be used for validation.
    ///
    pub fn from_csv(inputs: &Path, outputs: &Path, valid_frac: f64) -> Result<DataSet, DataSetError> {

        let (in_shape, num_in_samples, in_values) = DataSet::load_data_from_path(&inputs)?;
        let (out_shape, num_out_samples, out_values) = DataSet::load_data_from_path(&outputs)?;

        if num_in_samples != num_out_samples {
            Err(DataSetError::DimensionMismatch)
        } else {
            let num_valid_samples = (valid_frac * num_in_samples as f64).floor() as u64;
            let num_train_samples = num_in_samples - num_valid_samples;
            let idx_in = (num_train_samples * in_shape) as usize;
            let idx_out = (num_train_samples * out_shape) as usize;

            Ok(DataSet {
                num_training_samples: num_train_samples,
                x_train: Array::new(&in_values[0..idx_in], Dim4::new(&[in_shape, 1, 1, num_train_samples])),
                y_train: Array::new(&out_values[0..idx_out], Dim4::new(&[out_shape, 1, 1, num_train_samples])),
                x_valid: Array::new(&in_values[idx_in..], Dim4::new(&[in_shape, 1, 1, num_valid_samples])),
                y_valid: Array::new(&out_values[idx_out..], Dim4::new(&[out_shape, 1, 1, num_valid_samples])),
                x_test: None,
                y_test: None,
                x_train_stats: None,
                y_train_stats: None,
                num_features: Dim4::new(&[in_shape, 1, 1, 1]),
            })
        }
    }

    fn load_data_from_path(path: &Path) -> Result<(u64, u64, Vec<f64>), DataSetError> {
        let mut reader = Reader::from_path(path);
        match reader {
            Ok(mut rdr) => {
                let mut values = Vec::<f64>::new();
                let mut input_shape = 0;
                for (i, result) in rdr.records().enumerate() {
                    let record = result.unwrap();
                    if i == 0 {
                        input_shape = record.len() as u64;
                    }
                    for entry in record.iter() {
                        values.push((*entry).parse::<f64>().unwrap());
                    }
                }

                let num_samples = values.len() as u64 / input_shape;
                Ok((input_shape, num_samples, values))
            },
            Err(e) => Err(DataSetError::Csv(e))
        }
    }

    pub fn normalize(&mut self) {
        self.x_train_stats = Some((mean(&self.x_train, 3), stdev(&self.x_train, 3)));
        //self.x_train_mean = Some(mean(&self.x_train, 3));
        //self.x_train_std = Some(stdev(&self.x_train, 3));
        match &self.x_train_stats {
            Some((mean, std)) => {
                self.x_train = div(&sub(&self.x_train, mean, true), std, true);
                self.x_train.eval();

                self.x_valid = div(&sub(&self.x_valid, mean, true), std, true);
                self.x_valid.eval();
            },
            None => (),
        }
        //self.x_train = div(&sub(&self.x_train, &(self.x_train_mean.unwrap()), true), &self.x_train_std.unwrap(), true);
        //self.x_train.eval();

        //let y_train_mean = mean(&self.y_train, 3);
        //let y_train_std = stdev(&self.y_train, 3);
        self.y_train_stats = Some((mean(&self.y_train, 3), stdev(&self.y_train, 3)));
        match &self.y_train_stats {
            Some((mean, std)) => {
                self.y_train = div(&sub(&self.y_train, mean, true), std, true);
                self.y_train.eval();

                self.y_valid = div(&sub(&self.y_valid, mean, true), std, true);
                self.y_valid.eval();
            },
            None => (),
        }


        //self.y_train = div(&sub(&self.y_train, &mean(&self.y_train, 3), true), &stdev(&self.y_train, 3), true);
        //self.y_train.eval();
    }

    pub fn x_train(&self) -> &Array<f64> {
        &self.x_train
    }

    pub fn y_train(&self) -> &Array<f64> {
        &self.y_train
    }

    pub fn x_valid(&self) -> &Array<f64> {
        &self.x_valid
    }

    pub fn y_valid(&self) -> &Array<f64> {
        &self.y_valid
    }

    pub fn x_test(&self) -> Option<&Array<f64>> {
        match &self.x_test {
            Some(x_test) => Some(&x_test),
            None => None
        }
    }

    pub fn y_test(&self) -> Option<&Array<f64>> {
        match &self.y_test {
            Some(y_test) => Some(&y_test),
            None => None
        }
    }

    pub fn batch_iterator(&self, batch_size: u64) {
        let mut rng = thread_rng();
        let mut y: Vec<u64> = (0..self.num_training_samples).collect();
        println!("Unshuffled: {:?}", y);
        y.shuffle(&mut rng);
        println!("Shuffled:   {:?}", y);

        println!("Batch idxs: {:?}", &y[0..batch_size as usize]);


        let idx = Array::new(&y[0..batch_size as usize], Dim4::new(&[1, 1, 1, batch_size]));
        af_print!("idx", idx);

        let mut idxr = Indexer::new();

        idxr.set_index(&idx, 3, Some(false));

        let mini_batch = index_gen(&self.x_train, idxr);
        af_print!("after:", mini_batch);
    }

}
*/

#[cfg(test)]
mod tests {
    use arrayfire::*;
    use crate::data::DataSet;
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    #[test]
    fn test_batches() {
        let num_features = 2;
        let num_samples = 4;
        let batch_size = 2;
        let in_values = [1.0f64, -2.0, 4.1, 3.5, 6.7, -2.3, -8.9, 1.1];
        //let dims = Dim4::new(&[num_features, 1, 1, num_samples]);
        //let out_values = [3.1f64, -2.1, 4.6, 0.3];

        // Shuffle indices
        let mut rng = thread_rng();
        let mut y: Vec<u64> = (0..num_samples).collect();
        y.shuffle(&mut rng);

        // Select values
        let mut mb_values: Vec<f64> = Vec::new();
        for i1 in &y[0..batch_size as usize] {
            for i2 in 0..num_features {
                let idx = (i1 + i2) as usize;
                mb_values.push(in_values[idx]);
            }
        }

        // Create mini-batch
        let dims = Dim4::new(&[num_features, 1, 1, batch_size]);
        let mb_x_train = Array::new(&mb_values, dims);
        af_print!("mb_x_train", mb_x_train);


        //let x = Array::new(&in_values, dims);
        //let y = Array::new(&out_values, Dim4::new(&[1, 1, 1, 4]));
        //let data = DataSet::new(x, y, 0.0).unwrap();

        //data.batch_iterator(2);
    }
}