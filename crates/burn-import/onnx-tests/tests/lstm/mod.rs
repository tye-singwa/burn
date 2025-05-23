use crate::include_models;
include_models!(lstm);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn lstm() {
        // Initialize the model with weights (loaded from the exported file)
        let model: lstm::Model<Backend> = lstm::Model::default();

        let device = Default::default();

        let test_input = Tensor::<Backend, 3>::from_floats([[[-0.1335, 0.3415, -0.0716]]], &device)
            .reshape([1, 1, 3]);

        let test_hidden1 =
            Tensor::<Backend, 3>::from_floats([[[-0.0909, -1.3297, -0.5426, 0.5471]]], &device)
                .reshape([1, 1, 4]);

        let test_hidden2 =
            Tensor::<Backend, 3>::from_floats([[[0.6431, -0.7905, -0.9058, -0.2607]]], &device)
                .reshape([1, 1, 4]);

        let _ = model.forward(test_input, test_hidden1, test_hidden2);
        // // matrix-matrix `a @ b`
        // let expected_mm = TensorData::from([[
        //     [[28f32, 34.], [76., 98.], [124., 162.]],
        //     [[604., 658.], [780., 850.], [956., 1042.]],
        // ]]);
        // // matrix-vector `c @ d` where the lhs vector is expanded and broadcasted to the correct dims
        // let expected_mv = TensorData::from([
        //     [
        //         [14f32, 38., 62., 86.],
        //         [110., 134., 158., 182.],
        //         [206., 230., 254., 278.],
        //     ],
        //     [
        //         [302., 326., 350., 374.],
        //         [398., 422., 446., 470.],
        //         [494., 518., 542., 566.],
        //     ],
        // ]);
        // // vector-matrix `d @ c` where the rhs vector is expanded and broadcasted to the correct dims
        // let expected_vm = TensorData::from([
        //     [
        //         [56f32, 62., 68., 74.],
        //         [152., 158., 164., 170.],
        //         [248., 254., 260., 266.],
        //     ],
        //     [
        //         [344., 350., 356., 362.],
        //         [440., 446., 452., 458.],
        //         [536., 542., 548., 554.],
        //     ],
        // ]);

        // output_mm.to_data().assert_eq(&expected_mm, true);
        // output_vm.to_data().assert_eq(&expected_vm, true);
        // output_mv.to_data().assert_eq(&expected_mv, true);
    }
}
