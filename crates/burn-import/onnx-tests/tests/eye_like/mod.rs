use crate::include_models;
include_models!(eye_like_1d, eye_like_2d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Shape, Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn eye_like_1d() {
        let device = Default::default();
        let model: eye_like_1d::Model<Backend> = eye_like_1d::Model::new(&device);

        let input_shape = Shape::from([2, 3]);
        let input = Tensor::random(input_shape, Distribution::Default, &device);
        let expected = TensorData::from([
            [0.0f32, 1.0, 0.0], //
            [0.0, 0.0, 1.0],
        ]);

        let output = model.forward(input);
        output.to_data().assert_eq(&expected, true);
    }
}
