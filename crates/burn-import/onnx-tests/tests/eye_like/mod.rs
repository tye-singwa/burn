use crate::include_models;
include_models!(eye_like_up, eye_like_down, eye_like_0);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Shape, Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn eye_like_up() {
        let device = Default::default();
        let model: eye_like_up::Model<Backend> = eye_like_up::Model::new(&device);

        let input_shape = Shape::from([2, 3]);
        let input = Tensor::random(input_shape, Distribution::Default, &device);
        let expected = TensorData::from([
            [0.0f32, 1.0, 0.0], //
            [0.0, 0.0, 1.0],
        ]);

        let output = model.forward(input);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn eye_like_down() {
        let device = Default::default();
        let model: eye_like_down::Model<Backend> = eye_like_down::Model::new(&device);

        let input_shape = Shape::from([2, 3]);
        let input = Tensor::random(input_shape, Distribution::Default, &device);
        let expected = TensorData::from([
            [0.0f32, 0.0, 0.0], //
            [1.0, 0.0, 0.0],
        ]);

        let output = model.forward(input);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn eye_like_0() {
        let device = Default::default();
        let model: eye_like_0::Model<Backend> = eye_like_0::Model::new(&device);

        let input_shape = Shape::from([2, 3]);
        let input = Tensor::random(input_shape, Distribution::Default, &device);
        let expected = TensorData::from([
            [1, 0, 0], //
            [0, 1, 0],
        ]);

        let output = model.forward(input);
        output.to_data().assert_eq(&expected, true);
    }
}
