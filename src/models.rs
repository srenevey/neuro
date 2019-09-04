use arrayfire::*;
use crate::layers::*;
use crate::activations::*;
use crate::losses::*;
use crate::optimizers::*;
use crate::data::*;
use std::marker::PhantomData;

enum NetworkError {
    NoLayers
}

pub struct Network<'a, L, O>
where L: DataSet, O: Optimizer
{
    data: &'a L,
    layers: Vec<Box<dyn Layer>>,
    loss_function: Loss,
    optimizer: O,
}


impl<'a, L, O> Network<'a, L, O>
where L: DataSet, O: Optimizer
{
    pub fn new(data: &'a L, loss_function: Loss, optimizer: O) -> Network<L, O> {
        Network {
            data,
            layers: Vec::new(),
            loss_function,
            optimizer,
        }
    }

    /// Add layer to the network
    pub fn add(&mut self, layer: Box<dyn Layer>)
    {
        let fan_in = match self.layers.last() {
            Some(layer) => layer.fan_out(),
            None => self.data.num_features(),
        };
        self.layers.push(layer);
        self.layers.last_mut().unwrap().initialize_parameters(fan_in);
    }

    /// Compute a forward pass of the network
    fn forward(&mut self, x: &Array<f64>) -> Array<f64> {
        self.layers.iter_mut().fold(
            x.copy(),
            |a_prev, layer| layer.compute_activation_mut(&a_prev)
        )
        /*
        let mut a_prev = x.copy();
        for layer in self.layers.iter_mut() {
            a_prev = layer.compute_activation_mut(&a_prev);
        }
        a_prev
        */
    }

    /// Compute a backward pass of the network
    fn backward(&mut self, y: &Array<f64>, y_expected: &Array<f64>) {
        self.layers.iter_mut().rev().fold(
            self.loss_function.grad(y, y_expected),
            |da_prev, layer| layer.compute_da_prev_mut(&da_prev)
        );

        /*
        let mut da_prev = self.loss_function.grad(y, y_expected);
        for layer in self.layers.iter_mut().rev() {
            da_prev = layer.compute_da_prev_mut(&da_prev);
        }
        */
    }

    pub fn fit(&mut self, batch_size: u64, epochs: u64) {
        let (name, platform, _, _) = device_info();
        println!("Running on {} using {}.", name, platform);

        self.initialize_optimizer();

        self.data.set_batch_size(batch_size);

        //let x = self.data.x_train();
        //let y = self.data.y_train();



        for epoch in 0..epochs {

            for (mini_batch_x, mini_batch_y) in *self.data.mini_batch() {

                // Compute activation of the last layer
                let al = self.forward(&mini_batch_x);

                // Print loss every 10 epoch
                if (epoch + 1) % 10 == 0 {
                    let train_loss = self.compute_loss(&al, &mini_batch_y);
                    let valid_loss = self.evaluate();
                    println!("epoch: {}, train_loss: {}, valid_loss: {}", epoch + 1, train_loss, valid_loss)
                }

                self.backward(&al, &mini_batch_y);
                self.update_parameters();
            }
            //mem_info!("Memory used by Arrayfire");
        }
    }

    fn initialize_optimizer(&mut self) {
        let mut dims = Vec::<(Dim4, Dim4)>::new();
        for layer in self.layers.iter() {
            dims.push((layer.parameters()[0].dims(), layer.parameters()[1].dims()));
        }
        self.optimizer.initialize_opt_params(dims);
    }

    fn compute_loss(&self, last_activation: &Array<f64>, y_expected: &Array<f64>) -> f64 {
        self.loss_function.eval(last_activation, y_expected)
    }

    fn update_parameters(&mut self) {
        // TODO: Parallelize
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            self.optimizer.update_parameters(layer, idx);
        }
    }

    /// Evaluate the model with the validation data
    ///
    /// Returns the loss computed on the validation set
    pub fn evaluate(&self) -> f64 {
        // Compute activation of last layer
        let mut a_prev = self.data.x_valid().copy();
        for layer in self.layers.iter() {
            a_prev = layer.compute_activation(&a_prev);
        }

        match self.layers.last() {
            Some(last_layer) => {
                self.loss_function.eval(&a_prev, self.data.y_valid())
            },
            None => 0.,
        }
    }
}