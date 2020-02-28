//! # Neuro
//!
//! Neuro is a deep learning library that runs on the GPU. The library is designed to be very modular and allow users
//! to easily add custom activation functions, loss functions, layers, and optimizers.
//! The library presently supports:
//! * Layers: BatchNorm, Conv2D, Dense, Dropout, Flatten, MaxPool2D.
//! * Optimizers: Adadelta, Adam, RMSprop, SGD.
//! * Activations: LeakyReLU, Linear, ReLU, Sigmoid, Softmax, Tanh.
//! * Loss functions: BinaryCrossEntropy, CrossEntropy, MeanAbsoluteError, MeanSquaredError, SoftmaxCrossEntropy.
//!
//! Additionaly, many initialization schemes are available. The current implementation allows the creation
//! of feedforward and convolutional neural networks. It is planned to add recurrent neural networks in the future.
//!
//! # Installation
//! The crate is powered by ArrayFire to perform all operations on the GPU. The first step is therefore to
//! [install this library](https://crates.io/crates/arrayfire).
//! When building a project, the path to the ArrayFire library must be in the path environment
//! variables. For instance for a typical install (on Unix):
//! ```bash
//! export DYLD_LIBRARY_PATH=/opt/arrayfire/lib
//! ```
//!
//! The models trained with neuro can be saved in the Hierarchical Data Format (HDF5). In order to do so, HDF5 1.8.4 or
//! newer must be installed. Installation files can be found on the [HDF Group website](https://www.hdfgroup.org/downloads/hdf5).
//! macOS users can install it with homebrew:
//! ```bash
//! brew install hdf5
//! ````
//!
//! To start using the library, add the following line to the project's Cargo.toml file:
//! ```rust
//! [dependencies]
//! neuro = "0.1.0"
//! ```
//!
//! It is highly recommended to build the project in release mode for considerable speedup (e.g. `cargo run my_project --release`).
//! In order to quickly get started, check out the [examples](https://srenevey.github.io/neuro/examples).

pub use self::tensor::Tensor;

pub mod activations;
pub mod data;
pub mod errors;
pub mod initializers;
pub(crate) mod io;
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