use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Update output rank for DeformConv (same as input rank).
pub fn deform_conv_update_outputs(node: &mut Node) {
    log::debug!("DeformConv rank inference for node {}", node.name);

    log::debug!("Input tensors: {:?}", node.inputs);
    log::debug!("Output tensors: {:?}", node.outputs);

    node.outputs[0].ty = ArgType::Tensor(
        TensorType {
            elem_type: tensor.elem_type
        }
    )
}