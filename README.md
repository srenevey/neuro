# neuro 
[Homepage](https://srenevey.github.io/neuro)  [API Documentation](https://srenevey.github.io/neuro/api)  [Examples](https://srenevey.github.io/neuro/examples)

Neuro is a deep learning library that runs on the GPU. The architecture of the library is inspired by Keras and works by stacking layers. Currently supported layers are:

* BatchNorm
* Conv2D (currently broken, see note 2 below)
* Dense
* Dropout
* MaxPooling2D

allowing the creation of feedforward and convolutional neural networks.

### Note 1
The crate is under heavy development and several features have not yet been implemented. Among them is the ability to save a trained network.

### Note 2
When creating convolutional neural networks, there is a bug when trying to retrieve the values from the GPU (the process freezes). This means that the loss and metrics cannot be printed and therefore there is no way to assess if the network is actually learning something.

## Installation
The crate is powered by ArrayFire to perform all operations on the GPU. The first step is therefore to install ArrayFire. Once the library is installed, clone this repository and start using neuro by importing it in your project:
```
use neuro::*;
```
When building your project, make sure that the path to the ArrayFire library is in the path environment variables. For instance for a typical install (on Unix):
```
export DYLD_LIBRARY_PATH=/opt/arrayfire/lib
```
In order to quickly get started, check out the [examples](https://srenevey.github.io/neuro/examples).
