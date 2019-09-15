use arrayfire::*;
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use image;

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
    x_train: Vec<Array<u8>>,
    y_train: Vec<Array<u8>>,
    x_valid: Array<u8>,
    y_valid: Array<u8>,
    //x_test: Option<Array<f64>>,
    //y_test: Option<Array<f64>>,
    //x_train_stats: Option<(Scaling, Vec<f64>, Vec<f64>)>,
    //y_train_stats: Option<(Scaling, Array<f64>, Array<f64>)>,
}

impl ImageDataSet {
    pub fn from_path(path: &Path, image_size: u32) -> Result<ImageDataSet, DataSetError> {
        if path.exists() {

            // Create the paths to the training and validation samples
            let train_path = path.join("train");
            if !train_path.exists() {
                return Err(DataSetError::TrainPathDoesNotExist);
            }
            let valid_path = path.join("valid");
            if !valid_path.exists() {
                return Err(DataSetError::ValidPathDoesNotExist);
            }

            // Load the images in vectors of Array<u8> and shuffle
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
            })
        } else {
            Err(DataSetError::PathDoesNotExist)
        }
    }

    pub fn print_stats(&self) {
        println!("Number of training samples: {}", self.num_train_samples);
        println!("Number of validation samples: {}", self.num_valid_samples);
        println!("Input shape: {:?}", self.input_shape);
        println!("Output shape: {:?}", self.output_shape);
    }

    fn load_images_from_dir(path: &Path, image_size: u32) -> Result<(Vec<Array<u8>>, Vec<Array<u8>>, Vec<String>), DataSetError> {
        // Each subdirectory corresponds to a class
        let num_classes = fs::read_dir(&path).unwrap().count();
        let mut classes = Vec::<String>::with_capacity(num_classes);
        let mut x: Vec<Array<u8>> = Vec::new();
        let mut y: Vec<Array<u8>> = Vec::with_capacity(num_classes);

        // Iterate through the subdirectories and load the images
        for class in fs::read_dir(&path)? {
            let class = class?;
            if class.path().is_dir() {

                // One hot encoding of the classes
                classes.push(class.path().file_name().unwrap().to_str().unwrap().to_string());
                let mut one_hot_encoded_class = vec![0u8; num_classes];
                one_hot_encoded_class[classes.len() - 1] = 1;

                for image in fs::read_dir(&class.path())? {
                    let image = image?;

                    let img = image::open(image.path()).unwrap().resize_exact(image_size, image_size, image::FilterType::Nearest);
                    match img.color() {
                        image::Gray(_) => {
                            let luma_img = img.as_luma8().unwrap();
                            let img_vec = luma_img.to_vec();
                            let (width, height) = luma_img.dimensions();
                            x.push(Array::new(&luma_img.to_vec(), Dim4::new(&[width as u64, height as u64, 1, 1])));
                        },
                        image::GrayA(_) => {
                            let lumaa_img = img.as_luma_alpha8().unwrap();
                            let img_vec = lumaa_img.to_vec();
                            let (width, height) = lumaa_img.dimensions();
                            x.push(Array::new(&lumaa_img.to_vec(), Dim4::new(&[width as u64, height as u64, 2, 1])));
                        },
                        image::RGB(_) => {
                            let rgb_img = img.as_rgb8().unwrap();
                            let img_vec = rgb_img.to_vec();
                            let (width, height) = rgb_img.dimensions();
                            x.push(Array::new(&rgb_img.to_vec(), Dim4::new(&[width as u64, height as u64, 3, 1])));
                        },
                        image::RGBA(_) => {
                            let rgba_img = img.as_rgba8().unwrap();
                            let img_vec = rgba_img.to_vec();
                            let (width, height) = rgba_img.dimensions();
                            x.push(Array::new(&rgba_img.to_vec(), Dim4::new(&[width as u64, height as u64, 4, 1])));
                        },
                        image::BGR(_) => {
                            let bgr_img = img.as_bgr8().unwrap();
                            let img_vec = bgr_img.to_vec();
                            let (width, height) = bgr_img.dimensions();
                            x.push(Array::new(&bgr_img.to_vec(), Dim4::new(&[width as u64, height as u64, 3, 1])));
                        },
                        image::BGRA(_) => {
                            let bgra_img = img.as_bgra8().unwrap();
                            let img_vec = bgra_img.to_vec();
                            let (width, height) = bgra_img.dimensions();
                            x.push(Array::new(&bgra_img.to_vec(), Dim4::new(&[width as u64, height as u64, 4, 1])));
                        } ,
                        image::Palette(_) => { panic!("The image could not be read.")},
                    };

                    // Create label
                    y.push(Array::new(&one_hot_encoded_class, Dim4::new(&[num_classes as u64, 1, 1, 1])));
                }
            }
        }
        Ok((x, y, classes))
    }

    fn shuffle_vec(x: &mut Vec<Array<u8>>, y: &mut Vec<Array<u8>>) {
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

    fn join_vec(x: &Vec<Array<u8>>, y: &Vec<Array<u8>>) -> (Array<u8>, Array<u8>) {
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

