//! # Neuro
//!
//! A library to do deep learning on the GPU. The architecture of the library is inspired by Keras and works by stacking layers.
//! Currently supported layers are:
//! * BatchNorm
//! * Conv2D
//! * Dense
//! * Dropout
//! * MaxPooling2D
//!
//! allowing the creation of feedforward and convolutional neural networks.

pub use self::tensor::Tensor;

pub mod activations;
pub mod data;
pub mod layers;
pub mod losses;
pub mod metrics;
pub mod models;
pub mod optimizers;
pub mod regularizers;
pub mod tensor;



/// Assert if two expressions are approximately equal.
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