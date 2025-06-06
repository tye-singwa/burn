use crate::include_models;
include_models!(lstm);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn lstm() {
        let model: lstm::Model<Backend> = lstm::Model::default();
        let device = Default::default();

        let test_input = Tensor::<Backend, 3>::from_floats(
            [
                [[0.5, -0.14, 0.65], [1.52, -0.23, -0.23]],
                [[1.58, 0.77, -0.47], [0.54, -0.46, -0.47]],
                [[0.24, -1.91, -1.72], [-0.56, -1.01, 0.31]],
                [[-0.91, -1.41, 1.47], [-0.23, 0.07, -1.42]],
                [[-0.54, 0.11, -1.15], [0.38, -0.6, -0.29]],
            ],
            &device,
        );
        let test_initial_hidden = Tensor::<Backend, 3>::from_floats(
            [[[-0.6, 1.85, -0.01], [-1.06, 0.82, -1.22]]],
            &device,
        );
        let test_initial_cell =
            Tensor::<Backend, 3>::from_floats([[[0.21, -1.96, -1.33], [0.2, 0.74, 0.17]]], &device);

        let (output, output_hidden, output_cell) =
            model.forward(test_input, test_initial_hidden, test_initial_cell);

        let expected_output = Tensor::<Backend, 3>::from_floats(
            [
                [[
                    [0.62440413, -0.44544125, -0.09581377], //
                    [0.05938361, 0.16412814, 0.27383617],
                ]],
                [[
                    [0.725525, 0.14052059, 0.40293327], //
                    [0.40060508, 0.3772577, 0.45591256],
                ]],
                [[
                    [0.34914458, -0.03509847, 0.22759908], //
                    [0.6250826, 0.4965896, 0.6060403],
                ]],
                [[
                    [0.723569, 0.19432248, 0.5564013], //
                    [0.50790846, 0.5127549, 0.4036172],
                ]],
                [[
                    [0.5121318, 0.28977823, 0.3770225], //
                    [0.73861325, 0.564144, 0.64455897],
                ]],
            ],
            &device,
        );
        let expected_output_hidden = Tensor::<Backend, 3>::from_floats(
            [[
                [0.5121318, 0.28977823, 0.3770225], //
                [0.73861325, 0.564144, 0.64455897],
            ]],
            &device,
        );
        let expected_output_cell = Tensor::<Backend, 3>::from_floats(
            [[
                [1.9854021, 0.5002765, 1.0150967], //
                [2.4118917, 1.8665679, 1.8459547],
            ]],
            &device,
        );

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected_output, Tolerance::default());
        output_hidden
            .to_data()
            .assert_approx_eq::<FT>(&expected_output_hidden, Tolerance::default());
        output_cell
            .to_data()
            .assert_approx_eq::<FT>(&expected_output_cell, Tolerance::default());
    }
}
