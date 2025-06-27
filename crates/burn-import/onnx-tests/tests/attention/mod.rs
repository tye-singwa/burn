// Import the shared macro
use crate::include_models;
include_models!(attention_mha_3d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData, Tolerance, ops::FloatElem};

    type Backend = burn_ndarray::NdArray<f32>;
    type FT = FloatElem<Backend>;

    #[test]
    fn attention_mha_3d() {
        let device = Default::default();
        let model: attention_mha_3d::Model<Backend> = attention_mha_3d::Model::new(&device);
    }
}