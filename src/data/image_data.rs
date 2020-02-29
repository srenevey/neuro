//! Helper methods to work with image data sets.
use arrayfire::*;
use image;
use image::DynamicImage;
use walkdir::{DirEntry, WalkDir};
use rand::{thread_rng, Rng};
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::io::Write;

use super::{Scaling, DataSet, DataSetError};
use crate::errors::*;
use crate::tensor::*;

/// Structure representing a collection of images.
///
/// A builder class is provided for ease of creation: [ImageDataSetBuilder](struct.ImageDataSetBuilder.html).
pub struct ImageDataSet {
    input_shape: Dim,
    output_shape: Dim,
    image_size: (u32, u32),
    image_ops: ImageOps,
    num_train_samples: u64,
    num_valid_samples: u64,
    classes: Vec<String>,
    x_train: Tensor,
    y_train: Tensor,
    x_valid: Option<Tensor>,
    y_valid: Option<Tensor>,
    x_test: Option<Tensor>,
    y_test: Option<Tensor>,
}

impl ImageDataSet {
    /// Constructs an ImageDataSet from a directory tree.
    ///
    /// The images must be in folders named after the corresponding class in a *train* top-level directory.
    /// Optionally, if a *test* directory exists, its content will be used to create a test set.
    /// For instance:
    /// ```
    /// pets/
    ///   train/
    ///     cats/
    ///       img1.jpg
    ///       img2.jpg
    ///       ...
    ///     dogs/
    ///       img1.jpg
    ///       img2.jpg
    ///       ...
    ///   test/
    ///     cats/
    ///       img1.jpg
    ///       img2.jpg
    ///       ...
    ///     dogs/
    ///       img1.jpg
    ///       img2.jpg
    ///       ...
    /// ```
    ///
    /// The images are resized to the given size using nearest-neighbor interpolation. The aspect ratio is not conserved.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the top level directory containing the images.
    /// * `image_size` - The height and width of the images.
    /// * `one_hot_encode` - Flag indicating whether the labels are one hot encoded.
    /// * `valid_frac` - The fraction of the data used for validation.
    /// * `image_ops` - The collection of operations applied on the images.
    pub fn from_dir(path: &Path,
                    image_size: (u32, u32),
                    one_hot_encode: bool,
                    valid_frac: Option<f64>,
                    image_ops: ImageOps,
    ) -> Result<ImageDataSet, Error> {

        if let Some(valid_frac) = valid_frac {
            if valid_frac <= 0. || valid_frac >= 1. {
                return Err(std::convert::From::from(DataSetError::InvalidValidationFraction));
            }
        }

        print!("Loading the data...");
        io::stdout().flush().map_err(DataSetError::Io)?;

        if path.exists() {

            // Create the paths to the training samples and load the images
            let train_path = path.join("train");
            if !train_path.exists() {
                return Err(std::convert::From::from(DataSetError::TrainPathDoesNotExist));
            }
            let (x, y, classes) = Self::load_images_from_dir(&train_path, image_size, one_hot_encode, &image_ops)?;

            // Create the path to the test samples and load the images
            let test_path = path.join("test");
            let (x_test, y_test) = if test_path.exists() {
                let mut image_test_ops = ImageOps::default();
                image_test_ops.scale = image_ops.scale;
                let (x_test, y_test, _) = Self::load_images_from_dir(&test_path, image_size, one_hot_encode, &image_test_ops)?;
                (Some(x_test), Some(y_test))
            } else {
                (None, None)
            };

            // Split into train / validation sets
            let (x_train, y_train, x_valid, y_valid) = match valid_frac {
                Some(valid_frac) => Self::split_data(x, y, valid_frac),
                None => (x, y, None, None),
            };

            let input_shape = x_train.dims();
            let output_shape = y_train.dims();

            let num_train_samples = x_train.dims().get()[3];
            let num_valid_samples = match &x_valid {
                Some(x) => x.dims().get()[3],
                None => 0
            };

            println!("done.");

            Ok(ImageDataSet {
                input_shape,
                output_shape,
                image_size,
                image_ops,
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


    fn load_images_from_dir(path: &Path,
                            size: (u32, u32),
                            one_hot_encode: bool,
                            image_ops: &ImageOps,
    ) -> Result<(Tensor, Tensor, Vec<String>), DataSetError> {
        // Each subdirectory corresponds to a class
        let walker = WalkDir::new(&path).min_depth(1).max_depth(1).into_iter();
        let num_classes = walker.filter_entry(|e| !Self::is_hidden(e)).count();
        let mut classes = Vec::<String>::with_capacity(num_classes);

        let mut x_vec: Vec<PrimitiveType> = Vec::new();
        let mut y_vec: Vec<PrimitiveType> = Vec::new();

        // Iterate through the subdirectories and load the images
        let mut class_id: usize = 0;
        let mut num_channels = 0;
        let mut num_images = 0;
        for class in fs::read_dir(&path)? {
            let class = class?;
            if class.path().is_dir() {

                // Store name of the class
                classes.push(class.path().file_name().unwrap().to_str().unwrap().to_string());

                // Load the images
                for image in fs::read_dir(&class.path())? {
                    let dir_entry = image?;
                    let image = Self::load_image(dir_entry.path().as_path(), size, image_ops)?;

                    let label = if one_hot_encode {
                        Self::one_hot_encode(class_id, num_classes)
                    } else {
                        vec![class_id as PrimitiveType]
                    };

                    x_vec.extend(image.0);
                    y_vec.extend(label);

                    num_channels = image.1;
                    num_images += 1;
                }
                class_id += 1;
            }
        }

        let mut x = Tensor::new(&x_vec[..], Dim::new(&[num_channels as u64, size.1 as u64, size.0 as u64, num_images as u64]));
        //x = reorder(&x, Dim::new(&[2, 1, 0, 3]));
        x = reorder_v2(&x, 2, 1, Some(vec![0, 3]));
        let mut y = if one_hot_encode {
            Tensor::new(&y_vec[..], Dim::new(&[num_classes as u64, 1, 1, num_images as u64]))
        } else {
            Tensor::new(&y_vec[..], Dim::new(&[1, 1, 1, num_images as u64]))
        };

        Tensor::shuffle_mut(&mut x, &mut y);
        Ok((x, y, classes))
    }


    /// One hot encodes the label.
    ///
    /// # Arguments
    ///
    /// * `class_id` - The unique identifier of the class.
    /// * `num_classes` - The number of classes present in the dataset.
    fn one_hot_encode(class_id: usize, num_classes: usize) -> Vec<PrimitiveType> {
        if num_classes < 3 {
            let mut ohe = vec![0.; 1];
            ohe[0] = class_id as PrimitiveType;
            ohe
        } else {
            let mut ohe = vec![0.; num_classes];
            ohe[class_id] = 1.;
            ohe
        }
    }


    /// Splits the samples and labels into training and validation sets.
    ///
    /// # Return values
    ///
    /// Tuple containing the training samples, training labels, validation samples, and validation labels.
    fn split_data(x: Tensor, y: Tensor, valid_frac: f64) -> (Tensor, Tensor, Option<Tensor>, Option<Tensor>) {
        let num_samples = x.dims().get()[3];
        let num_valid_samples = (valid_frac * num_samples as f64).floor() as u64;
        let num_train_samples = num_samples - num_valid_samples;

        let seqs_train = &[Seq::default(), Seq::default(), Seq::default(), Seq::new(0.0, (num_train_samples - 1) as f64, 1.0)];
        let seqs_valid = &[Seq::default(), Seq::default(), Seq::default(), Seq::new(num_train_samples as f64, (num_samples - 1) as f64, 1.0)];
        let x_train = index(&x, seqs_train);
        let y_train = index(&y, seqs_train);
        let x_valid = index(&x, seqs_valid);
        let y_valid = index(&y, seqs_valid);
        (x_train, y_train, Some(x_valid), Some(y_valid))
    }

    /// Filters out hidden directories (typically .DS_Store on macOS).
    ///
    /// The filtering is performed by testing if the directory name starts by '.'.
    ///
    /// # Arguments
    ///
    /// * `entry` - DirEntry to test.
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

    /// Loads a single image from a path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the image.
    /// * `size` - The height and width of the image.
    /// * `image_ops` - The collection of operations to apply on the image.
    ///
    /// # Return value
    ///
    /// Tuple containing a vector with the image and the number of channels.
    pub fn load_image(path: &Path,
                      size: (u32, u32),
                      image_ops: &ImageOps,
    ) -> Result<(Vec<PrimitiveType>, u8), DataSetError> {

        match image::open(path) {
            Ok(mut image) => {
                image = image.resize_exact(size.1, size.0, image::imageops::FilterType::Nearest);
                let image_vec = image_ops.process(&mut image);
                let num_channels = image.color().channel_count();
                Ok((image_vec, num_channels))
            },
            Err(_) => Err(DataSetError::InvalidImagePath),
        }
    }

    /// Loads the images from the paths.
    ///
    /// The images are scaled by the factor used to create the dataset.
    ///
    /// # Arguments
    ///
    /// * `paths` - A slice of paths to the images. Each path must point to an individual image.
    ///
    pub fn load_image_vec(paths: &[&Path], image_size: (u32, u32), image_ops: &ImageOps) -> Result<Tensor, Error> {
        let mut images = Vec::new();
        let mut num_channels = None;

        for path in paths {
            let image = Self::load_image(path, image_size, image_ops)?;

            match num_channels {
                Some(n) => {
                    if image.1 != n {
                        return Err(std::convert::From::from(DataSetError::DifferentNumbersOfChannels));
                    }
                },
                None => num_channels = Some(image.1),
            }
            images.extend(image.0);
        }

        let mut x = Tensor::new(&images[..], Dim::new(&[num_channels.unwrap() as u64, image_size.1 as u64, image_size.0 as u64, paths.len() as u64]));
        x = reorder_v2(&x, 2, 1, Some(vec![0, 3]));
        Ok(x)
    }
    
    pub fn image_ops(&self) -> &ImageOps {
        &self.image_ops
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

    fn x_valid(&self) -> Option<&Tensor> {
        match &self.x_valid {
            Some(x) => Some(x),
            None => None
        }
    }

    fn y_valid(&self) -> Option<&Tensor> {
        match &self.y_valid {
            Some(y) => Some(y),
            None => None
        }
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
        writeln!(f, "=======")?;
        writeln!(f, "Dataset")?;
        writeln!(f, "=======")?;
        writeln!(f, "Samples shape: [{} {} {}]", self.input_shape.get()[0], self.input_shape.get()[1], self.input_shape.get()[2],)?;
        writeln!(f, "Labels shape: [{} {} {}]", self.output_shape.get()[0], self.output_shape.get()[1], self.output_shape.get()[2])?;
        writeln!(f, "Number of training samples: {}", self.num_train_samples)?;
        writeln!(f, "Number of validation samples: {}", self.num_valid_samples)?;
        match &self.x_test {
            Some(values) => writeln!(f, "Number of test samples: {}", values.dims().get()[3])?,
            None => {},
        }
        writeln!(f, "Number of classes: {}", self.classes.len())?;

        Ok(())
    }
}

/// Contains the parameters of the different operations applied on the images.
#[derive(Clone, Debug, Default)]
pub struct ImageOps {
    rotation: Option<(i32, f64)>,
    hflip: Option<f64>,
    vflip: Option<f64>,
    scale: Option<PrimitiveType>,
}

impl ImageOps {
    /// Creates a new image operations structure.
    ///
    /// # Arguments
    ///
    /// * `rotation` - A tuple containing the absolute value of the maximum angle in degrees and the probability that a rotation is applied.
    /// * `hflip` - The probability that the image is flipped horizontally.
    /// * `vflip` - The probability that the image is flipped vertically.
    /// * `scale` - A factor applied to each pixel of the image.
    pub fn new(rotation: Option<(i32, f64)>, hflip: Option<f64>, vflip: Option<f64>, scale: Option<PrimitiveType>) -> ImageOps {
        ImageOps {
            rotation,
            hflip,
            vflip,
            scale,
        }
    }

    fn process(&self, image: &mut DynamicImage) -> Vec<PrimitiveType> {
        self.rotate(image);
        self.hflip(image);
        self.vflip(image);
        self.scale(image)
    }

    fn rotate(&self, image: &mut DynamicImage) {
        if let Some((angle, prob)) = self.rotation {
            if prob >= thread_rng().gen() {
                // Generate angle between ±angle
                let alpha: i32 = thread_rng().gen_range(-angle, angle);
                image.huerotate(alpha);
            }
        }
    }

    fn hflip(&self, image: &mut DynamicImage) {
        if let Some(prob) = self.hflip {
            if prob >= thread_rng().gen() {
                image.fliph();
            }
        }
    }

    fn vflip(&self, image: &mut DynamicImage) {
        if let Some(prob) = self.vflip {
            if prob >= thread_rng().gen() {
                image.flipv();
            }
        }
    }

    fn scale(&self, image: &mut DynamicImage) -> Vec<PrimitiveType> {
       image.to_bytes().iter_mut().map(|pixel| {
            if let Some(factor) = self.scale {
                *pixel as PrimitiveType * factor
            } else {
                *pixel as PrimitiveType
            }
        }).collect::<Vec<PrimitiveType>>()

    }
}

enum Source {
    //CSV,
    Dir,
}

pub struct ImageDataSetBuilder {
    source: Source,
    path: &'static Path,
    image_size: (u32, u32),
    valid_frac: Option<f64>,
    one_hot_encode: bool,
    image_ops: ImageOps,
}

impl ImageDataSetBuilder {

    /// Creates a dataset builder from a directory tree.
    ///
    /// The images must be in folders named after the corresponding class in a *train* top-level directory.
    /// Optionally, if a *test* directory exists, its content will be used to create a test set.
    /// For instance:
    /// ```
    /// pets/
    ///   train/
    ///     cats/
    ///       img1.jpg
    ///       img2.jpg
    ///       ...
    ///     dogs/
    ///       img1.jpg
    ///       img2.jpg
    ///       ...
    ///   test/
    ///     cats/
    ///       img1.jpg
    ///       img2.jpg
    ///       ...
    ///     dogs/
    ///       img1.jpg
    ///       img2.jpg
    ///       ...
    /// ```
    ///
    /// The images are resized to the given size using nearest-neighbor interpolation. The aspect ratio is not conserved.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::path::Path;
    /// # use neuro::data::ImageDataSetBuilder;
    /// # use neuro::errors::NeuroError;
    /// # fn main() -> Result<(), NeuroError> {
    /// let path = Path::new("dataset/pets");
    /// let data = ImageDataSetBuilder::from_dir(&path, (32, 32))
    ///     .one_hot_encode()
    ///     .valid_split(0.2)
    ///     .scale(1./255.)
    ///     .rotate(10, 0.1)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_dir(path: &'static Path, image_size: (u32, u32)) -> ImageDataSetBuilder {
        ImageDataSetBuilder {
            source: Source::Dir,
            path,
            image_size,
            valid_frac: None,
            one_hot_encode: false,
            image_ops: ImageOps::default(),
        }
    }

    /*
    pub fn from_csv(path: &'static Path, image_size: (u32, u32)) -> ImageDataSetBuilder {
        unimplemented!();
    }
    */

    /// Builds an ImageDataSet from the image dataset builder.
    pub fn build(self) -> Result<ImageDataSet, Error> {
        match self.source {
            /*
            Source::CSV => {
                Ok(ImageDataSet {
                    input_shape: Dim::new(&[1, 1, 1, 1]),
                    output_shape: Dim::new(&[1, 1, 1, 1]),
                    image_size: self.image_size,
                    image_ops: ImageOps::default(),
                    num_train_samples: 0,
                    num_valid_samples: 0,
                    classes: Vec::new(),
                    x_train: Tensor::new_empty_tensor(),
                    y_train: Tensor::new_empty_tensor(),
                    x_valid: None,
                    y_valid: None,
                    x_test: None,
                    y_test: None,
                })
            }, */
            Source::Dir => {
                ImageDataSet::from_dir(self.path, self.image_size, self.one_hot_encode, self.valid_frac, self.image_ops)
            }
        }
    }

    /// Flips the images horizontally with the given probability.
    pub fn hflip(mut self, prob: f64) -> ImageDataSetBuilder {
        if prob < 0. || prob > 1. {
            panic!("The probability must be between 0 and 1.")
        }
        self.image_ops.hflip = Some(prob);
        self
    }

    /// Flips the images vertically with the given probability.
    pub fn vflip(mut self, prob: f64) -> ImageDataSetBuilder {
        if prob < 0. || prob > 1. {
            panic!("The probability must be between 0 and 1.")
        }

        self.image_ops.vflip = Some(prob);
        self
    }

    /// One hot encodes the labels.
    pub fn one_hot_encode(mut self) -> ImageDataSetBuilder {
        self.one_hot_encode = true;
        self
    }

    /// Rotates the images by an angle drawn from a uniform distribution with bounds ±`angle` (in degrees). A rotation is applied with the given probability.
    pub fn rotate(mut self, angle: i32, prob: f64) -> ImageDataSetBuilder {
        if prob < 0. || prob > 1. {
            panic!("The probability must be between 0 and 1.")
        }
        self.image_ops.rotation = Some((angle, prob));
        self
    }

    /// Splits the data into a training and validation sets.
    pub fn valid_split(mut self, valid_frac: f64) -> ImageDataSetBuilder {
        if valid_frac <= 0. || valid_frac >= 1. {
            panic!("The validation fraction must be between 0 and 1 (excluded).")
        }
        self.valid_frac = Some(valid_frac);
        self
    }

    /// Scales the images by multiplying each pixel by the given factor.
    pub fn scale(mut self, factor: PrimitiveType) -> ImageDataSetBuilder {
        self.image_ops.scale = Some(factor);
        self
    }
}