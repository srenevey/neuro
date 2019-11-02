use arrayfire::*;
use crate::tensor::*;

#[derive(Debug)]
pub enum Metrics {
    Accuracy,
    F2Score,
    LogLoss,
    MeanAbsoluteError,
    MeanSquaredError,
    RSquared,
}

impl Metrics {
    pub(crate) fn eval(&self, y_pred: &Tensor, y_true: &Tensor) -> PrimitiveType {
        match self {
            Metrics::Accuracy => {
                let mb_size = y_true.dims().get()[3];
                let num_classes = y_true.dims().get()[0];
                let (predicted_class, true_class) = if num_classes == 1 {
                    let predicted_class = select(&constant(1u32, y_pred.dims()), &ge(y_pred, &0.5, true), &constant(0u32, y_pred.dims()));
                    let true_class = select(&constant(1u32, y_true.dims()), &ge(y_true, &0.5, true), &constant(0u32, y_true.dims()));;
                    (predicted_class, true_class)
                } else {
                    let predicted_class = imax(y_pred, 0).1;
                    let true_class = imax(y_true, 0).1;
                    (predicted_class, true_class)
                };
                //af_print!("y_pred", y_pred);
                //af_print!("y_true", y_true);
                //af_print!("predicted class", predicted_class);
                //af_print!("true class", true_class);
                let num_correctly_classified = eq(&predicted_class, &true_class, true);
                sum_all(&num_correctly_classified).0 as PrimitiveType / mb_size as PrimitiveType
            },
            Metrics::F2Score => { unimplemented!() },
            Metrics::LogLoss => { unimplemented!() },
            Metrics::MeanAbsoluteError => { unimplemented!() },
            Metrics::MeanSquaredError => { unimplemented!() },
            Metrics::RSquared => { unimplemented!() }
        }
    }
}


#[cfg(test)]
mod tests {
    use arrayfire::*;
    use crate::metrics::Metrics;
    use crate::*;

    #[test]
    fn test_accuracy() {
        let predictions = [0.1, 0.3, 0.6, 0.15, 0.8, 0.05, 0.6, 0.3, 0.1];
        let true_values = [0., 0., 1., 0., 1., 0., 0., 1., 0.];
        let y_pred = Array::new(&predictions, Dim4::new(&[3, 1, 1, 3]));
        let y_true = Array::new(&true_values, Dim4::new(&[3, 1, 1, 3]));

        let metrics = Metrics::Accuracy;
        let score = metrics.eval(&y_pred, &y_true);
        assert_approx_eq!([score], [0.6666666]);
    }
}