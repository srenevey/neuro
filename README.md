# neuro 
[![Crates.io](https://img.shields.io/crates/v/neuro.svg)](https://crates.io/crates/neuro/) [![Crates.io](https://img.shields.io/crates/l/neuro.svg)](https://opensource.org/licenses/BSD-3-Clause)

[Homepage](https://srenevey.github.io/neuro) • [API Documentation](https://srenevey.github.io/neuro/api/neuro) • [Examples](https://srenevey.github.io/neuro/examples)

Neuro is a deep learning library that runs on the GPU. The library is designed to be very modular and allow users to easily add custom activation functions, loss functions, layers, and optimizers.
The library presently supports:

  * Layers: BatchNorm, Conv2D, Dense, Dropout, Flatten, MaxPool2D.
  * Optimizers: Adadelta, Adam, RMSprop, SGD.
  * Activations: LeakyReLU, Linear, ReLU, Sigmoid, Softmax, Tanh.
  * Loss functions: BinaryCrossEntropy, CrossEntropy, MeanAbsoluteError, MeanSquaredError, SoftmaxCrossEntropy.

Additionaly, many initialization schemes are available. The current implementation allows the creation of feedforward and convolutional neural networks. It is planned to add recurrent neural networks in the future.

# Installation
The crate is powered by ArrayFire to perform all operations on the GPU. The first step is therefore to [install this library](https://crates.io/crates/arrayfire). When building a project, the path to the ArrayFire library must be in the path environment variables. For instance for a typical install (on Unix):
```bash
export DYLD_LIBRARY_PATH=/opt/arrayfire/lib
```

The models trained with neuro can be saved in the Hierarchical Data Format (HDF5). In order to do so, HDF5 1.8.4 or newer must be installed. Installation files can be found on the [HDF Group website](https://www.hdfgroup.org/downloads/hdf5). macOS users can install it with homebrew:

```shell
brew install hdf5
````

To start using the library, add the following line to the project's Cargo.toml file:
```toml
[dependencies]
neuro = "0.1.0"
```

It is highly recommended to build the project in release mode for considerable speedup (e.g. `cargo run my_project --release`).
In order to quickly get started, check out the [examples](https://srenevey.github.io/neuro/examples).


