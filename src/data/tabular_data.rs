use arrayfire::*;
use super::{DataSet, DataSetError, Scaling};
use std::path::Path;
use csv::Reader;
use rand::thread_rng;
use rand::Rng;


pub struct TabularDataSet {
    num_features: u64,
    num_outputs: u64,
    num_train_samples: u64,
    x_train: Vec<Array<f64>>,
    y_train: Vec<Array<f64>>,
    x_valid: Array<f64>,
    y_valid: Array<f64>,
    x_test: Option<Array<f64>>,
    y_test: Option<Array<f64>>,
    x_train_stats: Option<(Scaling, Vec<f64>, Vec<f64>)>,
    y_train_stats: Option<(Scaling, Array<f64>, Array<f64>)>,
}

impl TabularDataSet {
    pub fn from_csv(inputs: &Path, outputs: &Path, valid_frac: f64) -> Result<TabularDataSet, DataSetError> {
        let (in_shape, num_in_samples, in_values) = TabularDataSet::load_data_from_path(&inputs)?;
        let (out_shape, num_out_samples, out_values) = TabularDataSet::load_data_from_path(&outputs)?;

        if num_in_samples != num_out_samples {
            Err(DataSetError::DimensionMismatch)
        } else {
            let num_samples = num_in_samples;

            // Create a vector of arrays containing the data
            let mut x: Vec<Array<f64>> = Vec::with_capacity(num_samples as usize);
            let mut y: Vec<Array<f64>> = Vec::with_capacity(num_samples as usize);

            for i in 0..num_samples {
                let lb_x = (i * in_shape) as usize;
                let ub_x = ((i + 1) * in_shape) as usize;
                x.push(Array::new(&in_values[lb_x..ub_x], Dim4::new(&[in_shape, 1, 1, 1])));

                let lb_y = (i * out_shape) as usize;
                let ub_y = ((i + 1) * out_shape) as usize;
                y.push(Array::new(&out_values[lb_y..ub_y], Dim4::new(&[out_shape, 1, 1, 1])));
            }

            // Shuffle the data before we create the training and validation sets
            let mut rng = thread_rng();
            for i in (1..num_samples as usize).rev() {
                let idx = rng.gen_range(0, i + 1);
                x.swap(i, idx);
                y.swap(i, idx);
            }

            // Compute number of samples in training set and validation set
            let num_valid_samples = (valid_frac * num_samples as f64).floor() as u64;
            let num_train_samples = num_samples - num_valid_samples;

            // Create validation set
            let mut x_valid = x[num_train_samples as usize].copy();
            let mut y_valid = y[num_train_samples as usize].copy();
            for i in 1..(num_valid_samples as usize) {
                x_valid = join(3, &x_valid, &x[num_train_samples as usize + i]);
                y_valid = join(3, &y_valid, &y[num_train_samples as usize + i]);
            }

            // Create the data set
            Ok(TabularDataSet {
                num_train_samples,
                num_features: in_shape,
                num_outputs: out_shape,
                x_train: x[0..num_train_samples as usize].to_vec(),
                y_train: y[0..num_train_samples as usize].to_vec(),
                x_valid,
                y_valid,
                x_test: None,
                y_test: None,
                x_train_stats: None,
                y_train_stats: None,
            })
        }
    }

    fn load_data_from_path(path: &Path) -> Result<(u64, u64, Vec<f64>), DataSetError> {
        let reader = Reader::from_path(path);
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

    /// Normalize the output data.
    ///
    /// The values in y_train are rescaled to within 0 and 1.
    pub fn normalize_output(&mut self) {
        // Normalize y_train
        let mut y_train = self.y_train[0].copy();
        for i in 1..self.y_train.len() {
            y_train = join(3, &y_train, &self.y_train[i]);
        }

        let y_max = max(&y_train, 3);
        let y_min = min(&y_train, 3);

        let mut y_train_normalized = Vec::with_capacity(self.num_train_samples as usize);
        for sample in &self.y_train {
            y_train_normalized.push(div(&sub(sample, &y_max, false), &sub(&y_max, &y_min, false), false));
        }
        self.y_train = y_train_normalized;

        // Normalize y_valid and y_test
        self.y_valid = div(&sub(&self.y_valid, &y_max, true), &sub(&y_max, &y_min, true), true);

        match &mut self.y_test {
            Some(y_test) => {
                self.y_test = Some(div(&sub(y_test, &y_max, true), &sub(&y_max, &y_min, true), true));
            },
            None => (),
        }

        // Save normalization parameters
        self.y_train_stats = Some((Scaling::Normalized, y_min, y_max));
    }

    /// Standarize the output data.
    ///
    /// The values in y_train are rescaled such that the mean is 0 and standard deviation 1.
    pub fn standarize_output(&mut self) {
        // Normalize y_train
        let mut y_train = self.y_train[0].copy();
        for i in 1..self.y_train.len() {
            y_train = join(3, &y_train, &self.y_train[i]);
        }

        let y_mean = mean(&y_train, 3);
        let y_std = stdev(&y_train, 3);

        let mut y_train_standarized = Vec::with_capacity(self.num_train_samples as usize);
        for sample in &self.y_train {
            y_train_standarized.push(div(&sub(sample, &y_mean, false), &y_std, false));
        }
        self.y_train = y_train_standarized;

        // Standarize y_valid and y_test
        self.y_valid = div(&sub(&self.y_valid, &y_mean, true), &y_std, true);

        match &mut self.y_test {
            Some(y_test) => {
                self.y_test = Some(div(&sub(y_test, &y_mean, true), &y_std, true));
            },
            None => (),
        }

        // Save standardization parameters
        self.y_train_stats = Some((Scaling::Standarized, y_mean, y_std));
    }

    /// Print the scaling coefficients if any.
    pub fn print_stats(&self) {
        match &self.y_train_stats {
            Some((scaling, c1, c2)) => {
                match scaling {
                    Scaling::Normalized => {
                        println!("The output data have been normalized with:");
                        af_print!("y_min:", c1);
                        af_print!("y_max:", c2);
                    },
                    Scaling::Standarized => {
                        println!("The output data have been standarized with:");
                        af_print!("mean:", c1);
                        af_print!("std:", c2);
                    }
                }
            },
            None => println!("The output data have not been rescaled."),
        }
    }

}

impl DataSet for TabularDataSet {
    fn num_features(&self) -> u64 {
        self.num_features
    }

    fn num_outputs(&self) -> u64 {
        self.num_outputs
    }

    fn num_train_samples(&self) -> u64 {
        self.num_train_samples
    }

    fn shuffle(&mut self) {
        let mut rng = thread_rng();

        for i in (1..self.num_train_samples as usize).rev() {
            let idx = rng.gen_range(0, i + 1);
            self.x_train.swap(i, idx);
            self.y_train.swap(i, idx);
        }
    }

    fn x_train(&self) -> &Vec<Array<f64>> {
        &self.x_train
    }

    fn y_train(&self) -> &Vec<Array<f64>> {
        &self.y_train
    }

    fn x_valid(&self) -> &Array<f64> {
        &self.x_valid
    }

    fn y_valid(&self) -> &Array<f64> {
        &self.y_valid
    }

}

