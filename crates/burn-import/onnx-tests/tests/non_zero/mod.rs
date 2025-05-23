use crate::include_models;
include_models!(non_zero);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData, ops::FloatElem};

    type Backend = burn_ndarray::NdArray<f32>;
    type FT = FloatElem<Backend>;

    #[test]
    fn non_zero() {
        // Test for NonZero model
        let device = Default::default();
        let model = non_zero::Model::<Backend>::new(&device);
        let input: Tensor<Backend, 1, _> = Tensor::from_floats([-2, 0, -1, 0, 1, 2], &device);
        let expected: Tensor<Backend, 1, burn::prelude::Float> =
            Tensor::from_data(TensorData::from([]), &device);
        let output: Tensor<Backend, 2, Int> = model.forward(input);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), burn::tensor::Tolerance::default());
    }
}
