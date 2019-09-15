use arrayfire::*;

#[derive(Debug, Copy, Clone)]
pub enum Initializer {
    Zeros,
    Ones,
    Constant(f64),
    RandomUniform,
    RandomNormal,
    XavierNormal,
    XavierUniform,
    HeNormal,
    HeUniform,
}

impl Initializer {
    pub(crate) fn new(self, dims: Dim4, fan_in: f64, fan_out: f64) -> Array<f64> {
        match self {
            Initializer::Zeros  => constant(0.0f64, dims),
            Initializer::Ones   => constant(1.0f64, dims),
            Initializer::Constant(x) => constant(x, dims),
            Initializer::RandomUniform  => randu::<f64>(dims),
            Initializer::RandomNormal   => mul(&0.01, &randn::<f64>(dims), false),
            Initializer::XavierNormal => mul(&(2. / (fan_out + fan_in)).sqrt(), &randn::<f64>(dims), false),
            Initializer::XavierUniform => {
                let lim = (6. / (fan_out + fan_in)).sqrt();
                constant(-lim, dims) + constant(2. * lim, dims) * randu::<f64>(dims)
            },
            Initializer::HeNormal     => mul(&(2. / fan_in).sqrt(), &randn::<f64>(dims), false),
            Initializer::HeUniform     => {
                let lim = (6. / fan_in).sqrt();
                constant(-lim, dims) + constant(2. * lim, dims) * randu::<f64>(dims)
            }
        }
    }
}