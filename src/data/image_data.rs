//! Helper methods to work with image data sets.
use super::{Scaling, DataSet, DataSetError};
use crate::errors::*;
use crate::Tensor;
use crate::tensor::*;

use std::fmt;
use std::fs;
use std::path::Path;

use arrayfire::*;
use image;
use image::DynamicImage;
use walkdir::{DirEntry, WalkDir};


/// Structure representing a collection of images.
pub struct ImageDataSet {
    input_shape: Dim4,
    output_shape: Dim4,
    image_size: (u32, u32),
    num_train_samples: u64,
    num_valid_samples: u64,
    classes: Vec<String>,
    x_train: Tensor,
    y_train: Tensor,
    x_valid: Tensor,
    y_valid: Tensor,
    x_test: Option<Tensor>,
    y_test: Option<Tensor>,
}

impl ImageDataSet {

    /// Constructs an ImageDataSet from a path.
    ///
    /// The images must be in folders named after the corresponding class in a *train* top-level directory.
    /// Optionally, if a *test* directory exists, its content will be used to test the trained model.
    /// For instance:
    /// ```
    /// train/
    ///   cats/
    ///     img1.jpg
    ///     img2.jpg
    ///     ...
    ///   dogs/
    ///     img1.jpg
    ///     img2.jpg
    ///     ...
    /// test/
    ///   cats/
    ///     img1.jpg
    ///     img2.jpg
    ///     ...
    ///   dogs/
    ///     img1.jpg
    ///     img2.jpg
    ///     ...
    /// ```
    /// The method resizes the images and load them all in memory.
    ///
    /// # Arguments
    /// * `path`: path to load the images from. Must contain a *train* and optionally a *test* subdirectory.
    /// * `image_size`: height and width the images are resized to
    /// * `valid_frac`: fraction of the data used for validation. Must be between 0 and 1.
    ///
    pub fn from_path(path: &Path, image_size: (u32, u32), valid_frac: f64) -> Result<ImageDataSet, NeuroError> {

        if valid_frac < 0. || valid_frac > 1. {
            return Err(std::convert::From::from(DataSetError::InvalidValidationFraction));
        }

        print!("Loading the data...");

        if path.exists() {

            // Create the paths to the training and test samples
            let train_path = path.join("train");
            if !train_path.exists() {
                return Err(std::convert::From::from(DataSetError::TrainPathDoesNotExist));
            }

            let test_path = path.join("test");
            let (x_test, y_test) = if test_path.exists() {
                let (x_test, y_test, _, _, _) = ImageDataSet::load_images_from_dir(&test_path, image_size, None)?;
                (Some(x_test), Some(y_test))
            } else {
                (None, None)
            };

            // Load the images in tensors
            let valid_f = if valid_frac != 0. { Some(valid_frac) } else { None };
            let (x_train, y_train, x_valid, y_valid, classes) = ImageDataSet::load_images_from_dir(&train_path, image_size, valid_f)?;

            let input_shape = x_train.dims();
            let output_shape = y_train.dims();

            let num_train_samples = x_train.dims().get()[3];
            let num_valid_samples = x_valid.dims().get()[3];

            println!("done.");

            Ok(ImageDataSet {
                input_shape,
                output_shape,
                image_size,
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
            Err(std::convert::From::from(DataSetError::PathDoesNotExist))
        }
    }


    /// Loads the images in vectors of arrays and create one hot encoded classes.
    ///
    /// The images are shuffled before being split into training and validation sets.
    ///
    /// # Arguments
    /// * `path`: the path to load the images from
    /// * `image_size`: height and width the images are resized to
    /// * `valid_frac`: optional fraction of the data used for validation. Must be between 0 and 1.
    ///
    /// # Returns
    /// Returns a Result containing a tuple of tensors containing the training samples, training labels,
    /// validation samples, validation labels, and a vector containing the classes of the data set.
    ///
    fn load_images_from_dir(path: &Path, image_size: (u32, u32), valid_frac: Option<f64>) -> Result<(Tensor, Tensor, Tensor, Tensor, Vec<String>), DataSetError> {
        // Each subdirectory corresponds to a class
        let walker = WalkDir::new(&path).min_depth(1).max_depth(1).into_iter();
        let num_classes = walker.filter_entry(|e| !Self::is_hidden(e)).count();
        let mut classes = Vec::<String>::with_capacity(num_classes);

        let mut x: Vec<PrimitiveType> = Vec::new();
        let mut y: Vec<PrimitiveType> = Vec::new();

        // Iterate through the subdirectories and load the images
        let mut num_images = 0;
        let mut num_channels: Option<u64> = None;
        let mut class_id = 0;
        for class in fs::read_dir(&path)? {
            let class = class?;
            if class.path().is_dir() {

                // One hot encoding of the classes
                classes.push(class.path().file_name().unwrap().to_str().unwrap().to_string());
                let one_hot_encoded_class = if num_classes < 3 {
                    let mut ohe = vec![0.; 1];
                    ohe[0] = class_id as PrimitiveType;
                    ohe
                } else {
                    let mut ohe = vec![0.; num_classes];
                    ohe[class_id] = 1.;
                    ohe
                };
                class_id += 1;

                // Load the images in memory
                for image in fs::read_dir(&class.path())? {
                    let image = image?;

                    let img = image::open(image.path()).unwrap().resize_exact(image_size.1, image_size.0, image::FilterType::Nearest);
                    let img_vec = img.raw_pixels().iter().map(|it| *it as PrimitiveType / 255.0).collect::<Vec<PrimitiveType>>();
                    let img_channels = Self::get_num_channels(&img)?;

                    match num_channels {
                        Some(n) => {
                            if img_channels != n {
                                return Err(DataSetError::DifferentNumbersOfChannels);
                            }
                        },
                        None => num_channels = Some(img_channels),
                    }
                    x.extend(img_vec);

                    // Create label
                    y.extend(one_hot_encoded_class.clone());

                    num_images += 1;
                }
            }
        }

        let mut x_arr = Tensor::new(&x[..], Dim4::new(&[image_size.0 as u64, image_size.1 as u64, num_channels.unwrap(), num_images as u64]));
        let class_dim = if num_classes < 3 { 1 } else { num_classes as u64 };
        let mut y_arr = Tensor::new(&y[..], Dim4::new(&[class_dim, 1, 1, num_images as u64]));

        Tensor::shuffle_mut(&mut x_arr, &mut y_arr);

        // Split data into training and validation sets
        match valid_frac {
            Some(valid_frac) => {
                let num_valid_samples = (valid_frac * num_images as f64).floor() as u64;
                let num_train_samples = num_images - num_valid_samples;

                let seqs_train = &[Seq::default(), Seq::default(), Seq::default(), Seq::new(0.0, (num_train_samples - 1) as f64, 1.0)];
                let seqs_valid = &[Seq::default(), Seq::default(), Seq::default(), Seq::new(num_train_samples as f64, (num_images - 1) as f64, 1.0)];
                let x_train = index(&x_arr, seqs_train);
                let x_valid = index(&x_arr, seqs_valid);
                let y_train = index(&y_arr, seqs_train);
                let y_valid = index(&y_arr, seqs_valid);
                Ok((x_train, y_train, x_valid, y_valid, classes))
            },
            None => {
                let x_train = x_arr;
                let y_train = y_arr;
                let x_valid = Tensor::new_empty_tensor();
                let y_valid = Tensor::new_empty_tensor();
                Ok((x_train, y_train, x_valid, y_valid, classes))
            }
        }


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
            .map(|s| s.starts_with('.'))
            .unwrap_or(false)
    }


    /// Prints the classes in the dataset.
    pub fn print_classes(&self) {
        print!("Classes: ");
        for class in &self.classes {
            print!("{} ", class);
        }
        println!();
    }

    /// Retrieves the number of channels for the image.
    fn get_num_channels(image: &DynamicImage) -> Result<u64, DataSetError> {
        match image.color() {
            image::Gray(_) => Ok(1u64),
            image::GrayA(_) => Ok(2u64),
            image::RGB(_) => Ok(3u64),
            image::RGBA(_) => Ok(4u64),
            image::BGR(_) => Ok(3u64),
            image::BGRA(_) => Ok(4u64),
            image::Palette(_) => Err(DataSetError::ImageFormatNotSupported),
        }
    }


    /// Loads a single image from a path.
    ///
    /// The image is resized to the size defined at initialization.
    ///
    /// # Arguments
    /// * `path`: path to the image
    ///
    pub fn load_img(&self, path: &Path) -> Result<Tensor, NeuroError> {
        let img = image::open(path).unwrap().resize_exact(self.image_size.1, self.image_size.0, image::FilterType::Nearest);
        let img_vec = img.raw_pixels().iter().map(|it| *it as PrimitiveType / 255.0).collect::<Vec<PrimitiveType>>();
        let num_channels = Self::get_num_channels(&img)?;

        Ok(Tensor::new(&img_vec[..], Dim4::new(&[self.image_size.0 as u64, self.image_size.1 as u64, num_channels, 1])))
    }

    /// Loads the images from the paths.
    ///
    /// # Arguments
    /// * `paths`: a slice of paths to the images. Each path must point to an individual image.
    ///
    pub fn load_img_vec(&self, paths: &[&Path]) -> Result<Tensor, NeuroError> {
        let mut images = Vec::new();
        let mut num_channels = None;

        for path in paths {
            let img = image::open(path).unwrap().resize_exact(self.image_size.1, self.image_size.0, image::FilterType::Nearest);
            let img_vec = img.raw_pixels().iter().map(|it| *it as PrimitiveType / 255.0).collect::<Vec<PrimitiveType>>();
            let img_channels = Self::get_num_channels(&img)?;

            match num_channels {
                Some(n) => {
                    if img_channels != n {
                        return Err(std::convert::From::from(DataSetError::DifferentNumbersOfChannels));
                    }
                },
                None => num_channels = Some(img_channels),
            }

            images.extend(img_vec);
        }

        Ok(Tensor::new(&images[..], Dim4::new(&[self.image_size.0 as u64, self.image_size.1 as u64, num_channels.unwrap(), paths.len() as u64])))
    }
}

impl DataSet for ImageDataSet {
    fn input_shape(&self) -> Dim4 { self.input_shape }

    fn output_shape(&self) -> Dim4 { self.output_shape }

    fn num_train_samples(&self) -> u64 { self.num_train_samples }

    fn num_valid_samples(&self) -> u64 { self.num_valid_samples }

    fn classes(&self) -> Option<Vec<String>> {
        Some(self.classes.clone())
    }

    fn x_train(&self) -> &Tensor {
        &self.x_train
    }

    fn y_train(&self) -> &Tensor {
        &self.y_train
    }

    fn x_valid(&self) -> &Tensor {
        &self.x_valid
    }

    fn y_valid(&self) -> &Tensor {
        &self.y_valid
    }

    fn x_test(&self) -> Option<&Tensor> {
        match &self.x_test {
            Some(values) => Some(values),
            None => None,
        }
    }

    fn y_test(&self) -> Option<&Tensor> {
        match &self.y_test {
            Some(values) => Some(values),
            None => None,
        }
    }

    fn x_train_stats(&self) -> &Option<(Scaling, Tensor, Tensor)> {
        &None
    }

    fn y_train_stats(&self) -> &Option<(Scaling, Tensor, Tensor)> {
        &None
    }

}


impl fmt::Display for ImageDataSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Input shape: \t [{} {} {}]", self.input_shape.get()[0], self.input_shape.get()[1], self.input_shape.get()[2],)?;
        writeln!(f, "Output shape: \t [{} {} {}]", self.output_shape.get()[0], self.output_shape.get()[1], self.output_shape.get()[2])?;
        writeln!(f, "Number of training samples: \t {}", self.num_train_samples)?;
        writeln!(f, "Number of validation samples: \t {}", self.num_valid_samples)?;
        match &self.x_test {
            Some(values) => writeln!(f, "Number of test samples: \t {}", values.dims().get()[3])?,
            None => {},
        }
        writeln!(f, "Number of classes: \t {}", self.classes.len())?;

        Ok(())
    }
}