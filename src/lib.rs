//! # Neuro
//!
//! Neuro is a deep learning library that runs on the GPU. The architecture of the library is inspired by Keras and works by stacking layers.
//! Currently supported layers are:
//! * BatchNorm
//! * Conv2D
//! * Dense
//! * Dropout
//! * MaxPooling2D
//!
//! allowing the creation of feedforward and convolutional neural networks.
//!
//! ### Note
//! The crate is under heavy development and several features have not yet been implemented. Among them
//! is the ability to save a trained network.
//!
//!
//! ## Installation
//! The crate is powered by ArrayFire to perform all operations on the GPU. The first step is therefore to
//! [install ArrayFire](https://crates.io/crates/arrayfire). Once the library is installed, clone this repository
//! and start using neuro by importing it in your project:
//! ```
//! use neuro::*;
//! ```
//! When building your project, make sure that the path to the ArrayFire library is in the path environment
//! variables. For instance for a typical install (on Unix):
//! ```
//! export DYLD_LIBRARY_PATH=/opt/arrayfire/lib
//! ```
//! In order to quickly get started, check out the [examples](https://srenevey.github.io/neuro/examples).


pub use self::tensor::Tensor;

pub mod activations;
pub mod data;
pub mod errors;
pub mod initializers;
pub mod layers;
pub mod losses;
pub mod metrics;
pub mod models;
pub mod optimizers;
pub mod regularizers;
pub mod tensor;



/// Asserts if two expressions are approximately equal.
#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let eps = 1e-6;
        let (a, b) = ($a, $b);
        for (i,_) in a.iter().enumerate() {
            assert!(
            (a[i] - b[i]).abs() < eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            a[i],
            b[i],
            eps,
            (a[i] - b[i]).abs()
        );
        }
    }};
}