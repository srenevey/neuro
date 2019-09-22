//! # Neuro
//!
//! A library to do deep learning on the GPU. The architecture of the library is inspired by Keras and works by stacking layers.
//! Currently supported layers are:
//! * BatchNorm
//! * Conv2D
//! * Dense
//! * MaxPooling2D
//!
//! allowing the creation of feedforward and convolutional neural networks.

pub mod activations;
pub mod layers;
pub mod models;
pub mod losses;
pub mod optimizers;
pub mod data;


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