// Import the shared macro
use crate::include_models;
include_models!(deform_conv2d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    type Backend = burn_ndarray::NdArray<f32>;
    type FT = FloatElem<Backend>;

    #[test]
    fn deform_conv2d() {
        let device = Default::default();
        let model: deform_conv2d::Model<Backend> = deform_conv2d::Model::new(&device);

        let input =
            Tensor::<Backend, 4>::arange(0..(2 * 4 * 10 * 15), &device).reshape([2, 4, 10, 15]);

        let output = model.forward(input);
        let expected = TensorData::from([[[[0.5403f32, -0.6536, -0.9111, 0.9912]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
