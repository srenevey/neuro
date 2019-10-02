use arrayfire::*;
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use image;
use walkdir::{DirEntry, WalkDir};

use super::{Scaling, DataSet, DataSetError};
use image::{GenericImageView, ImageDecoder, load, ImageFormat};
use image::jpeg::JPEGDecoder;
use std::convert::TryInto;
use std::fs::FileType;
use rand::thread_rng;
use rand::Rng;

pub struct ImageDataSet {
    input_shape: Dim4,
    output_shape: Dim4,
    num_train_samples: u64,
    num_valid_samples: u64,
    classes: Vec<String>,
    x_train: Vec<Array<f64>>,
    y_train: Vec<Array<f64>>,
    x_valid: Array<f64>,
    y_valid: Array<f64>,
    x_test: Option<Array<f64>>,
    y_test: Option<Array<f64>>,
    //x_train_stats: Option<(Scaling, Vec<f64>, Vec<f64>)>,
    //y_train_stats: Option<(Scaling, Array<f64>, Array<f64>)>,
}

impl ImageDataSet {

    /// Construct an ImageDataSet from a path. All images in the data set will have the same size. This method loads all images in memory.
    ///
    /// # Arguments
    /// * `path`: path to load the images from. Must contain a train, valid, and (optional) test subdirectories
    /// * `image_size`: size the images are resized to
    ///
    pub fn from_path(path: &Path, image_size: u32) -> Result<ImageDataSet, DataSetError> {
        if path.exists() {

            // Create the paths to the training, validation, and test samples
            let train_path = path.join("train");
            if !train_path.exists() {
                return Err(DataSetError::TrainPathDoesNotExist);
            }
            let valid_path = path.join("valid");
            if !valid_path.exists() {
                return Err(DataSetError::ValidPathDoesNotExist);
            }
            let test_path = path.join("test");
            let (x_test, y_test) = if test_path.exists() {
                let (mut x_test_vec, mut y_test_vec, _) = ImageDataSet::load_images_from_dir(&test_path, image_size)?;
                let (x_test_arr, y_test_arr) = ImageDataSet::join_vec(&x_test_vec, &y_test_vec);
                (Some(x_test_arr), Some(y_test_arr))
            } else {
                (None, None)
            };

            // Load the images in vectors of Array<f64> and shuffle
            let (mut x_train, mut y_train, classes) = ImageDataSet::load_images_from_dir(&train_path, image_size)?;
            let (mut x_valid_vec, mut y_valid_vec, _) = ImageDataSet::load_images_from_dir(&valid_path, image_size)?;
            ImageDataSet::shuffle_vec(&mut x_train, &mut y_train);
            ImageDataSet::shuffle_vec(&mut x_valid_vec, &mut y_valid_vec);

            let (x_valid, y_valid) = ImageDataSet::join_vec(&x_valid_vec, &y_valid_vec);


            let input_shape = x_train[0].dims();
            let output_shape = y_train[0].dims();

            let num_train_samples = x_train.len() as u64;
            let num_valid_samples = x_valid_vec.len() as u64;

            Ok(ImageDataSet {
                input_shape,
                output_shape,
                num_train_samples,
                num_valid_samples,
                classes,
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_test,
                y_test,
            })
        } else {
            Err(DataSetError::PathDoesNotExist)
        }
    }

    /// Print some info on the data set
    pub fn print_stats(&self) {
        println!("Number of training samples: {}", self.num_train_samples);
        println!("Number of validation samples: {}", self.num_valid_samples);
        println!("Input shape: {:?}", self.input_shape);
        println!("Output shape: {:?}", self.output_shape);
    }

    /// Load the images in vectors of arrays and create one hot encoded classes.
    ///
    /// The method returns a Result containing a tuple of a vector containing the samples, a vector containing the labels, and a vector containing the one hot encoding dictionary.
    ///
    /// # Arguments
    /// * `path`: the path to load the images from
    /// * `image_size`: size the images are resized to
    ///
    /// # Panic
    /// The method panics if the image format is not supported
    ///
    fn load_images_from_dir(path: &Path, image_size: u32) -> Result<(Vec<Array<f64>>, Vec<Array<f64>>, Vec<String>), DataSetError> {
        // Each subdirectory corresponds to a class
        let walker = WalkDir::new(&path).min_depth(1).max_depth(1).into_iter();
        let num_classes = walker.filter_entry(|e| !Self::is_hidden(e)).count();

        let mut classes = Vec::<String>::with_capacity(num_classes);
        let mut x: Vec<Array<f64>> = Vec::new();
        let mut y: Vec<Array<f64>> = Vec::with_capacity(num_classes);

        // Iterate through the subdirectories and load the images
        for class in fs::read_dir(&path)? {
            let class = class?;
            if class.path().is_dir() {

                // One hot encoding of the classes
                classes.push(class.path().file_name().unwrap().to_str().unwrap().to_string());
                let mut one_hot_encoded_class = vec![0.; num_classes];
                one_hot_encoded_class[classes.len() - 1] = 1.;

                // Load the images in memory
                for image in fs::read_dir(&class.path())? {
                    let image = image?;

                    let img = image::open(image.path()).unwrap().resize_exact(image_size, image_size, image::FilterType::Nearest);
                    let img_vec = img.raw_pixels().iter().map(|it| *it as f64 / 255.0).collect::<Vec<f64>>();
                    let (width, height) = img.dimensions();

                    let num_channels = match img.color() {
                        image::Gray(_)      => { 1u64 },
                        image::GrayA(_)     => { 2u64 },
                        image::RGB(_)       => { 3u64 },
                        image::RGBA(_)      => { 4u64 },
                        image::BGR(_)       => { 3u64 },
                        image::BGRA(_)      => { 4u64 } ,
                        image::Palette(_)   => { panic!("The image could not be read.")},
                    };
                    x.push(Array::new(&img_vec[..], Dim4::new(&[height as u64, width as u64, num_channels, 1])));

                    // Create label
                    y.push(Array::new(&one_hot_encoded_class[..], Dim4::new(&[num_classes as u64, 1, 1, 1])));
                }
            }
        }
        Ok((x, y, classes))
    }

    /// Filter out hidden directories (typically .DS_Store on macos)
    ///
    /// The filtering is performed by testing if the directory name starts by '.'.
    ///
    /// # Arguments
    /// * `entry`: DirEntry to test
    ///
    fn is_hidden(entry: &DirEntry) -> bool {
        entry.file_name()
            .to_str()
            .map(|s| s.starts_with("."))
            .unwrap_or(false)
    }

    /// Shuffle the content of two vectors
    ///
    /// # Arguments
    /// * `x`: first vector to shuffle
    /// * `y`: second vector to shuffle
    ///
    fn shuffle_vec(x: &mut Vec<Array<f64>>, y: &mut Vec<Array<f64>>) {
        let mut rng = thread_rng();

        if x.len() != y.len() {
            panic!("The length of the two vectors must be the same.");
        }

        for i in (1..x.len()).rev() {
            let idx = rng.gen_range(0, i + 1);
            x.swap(i, idx);
            y.swap(i, idx);
        }
    }

    /// Convert vectors of arrays into arrays
    ///
    /// # Arguments
    /// * `x`: first vector to convert
    /// * `y`: second vector to convert
    ///
    /// # Panic
    /// The method panics if the length of the two vectors is different.
    ///
    fn join_vec(x: &Vec<Array<f64>>, y: &Vec<Array<f64>>) -> (Array<f64>, Array<f64>) {
        if x.len() != y.len() {
            panic!("The length of the two vectors must be the same.");
        }

        let mut x_arr = x[0].copy();
        let mut y_arr = y[0].copy();
        for i in 1..x.len() {
            x_arr = join(3, &x_arr, &x[i]);
            y_arr = join(3, &y_arr, &y[i]);
        }

        (x_arr, y_arr)
    }
}

impl DataSet for ImageDataSet {
    fn input_shape(&self) -> Dim4 { self.input_shape }

    fn output_shape(&self) -> Dim4 { self.output_shape }

    fn num_train_samples(&self) -> u64 { self.num_train_samples }

    fn shuffle(&mut self) {
        let mut rng = thread_rng();

        for i in (1..self.num_train_samples as usize).rev() {
            let idx = rng.gen_range(0, i + 1);
            self.x_train.swap(i, idx);
            self.y_train.swap(i, idx);
        }
    }

    fn x_train(&self) -> &Vec<Array<f64>> {
        unimplemented!()
    }

    fn y_train(&self) -> &Vec<Array<f64>> {
        unimplemented!()
    }

    fn x_valid(&self) -> &Array<f64> {
        unimplemented!()
    }

    fn y_valid(&self) -> &Array<f64> {
        unimplemented!()
    }
}

