use crate::{ArgType, TensorType, ir::Node};

pub fn lstm_config(node: &Node) -> (usize, usize) {
    let mut hidden_size: Option<i64> = None;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "hidden_size" => {
                hidden_size = Some(value.clone().into_i64());
            }
            _ => panic!("Unexpected attribute for LSTM: {key}"),
        }
    }

    if hidden_size.is_none() {
        panic!("LSTM: hidden_size attribute must be provided");
    }

    let weight_shape = node.inputs[1]
        .value
        .as_ref()
        .expect("LSTM: weight tensor must be present")
        .shape
        .clone();

    let d_input = weight_shape[0];

    (hidden_size.unwrap() as usize, d_input)
}

pub fn lstm_update_outputs(node: &mut Node) {
    log::debug!("LSTM rank inference for node {}", node.name);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: 3,
        static_shape: None, // shape is tracked and calculated at runtime
        elem_type: node.inputs[1].ty.elem_type().clone(),
    });

    if node.outputs.len() > 1 {
        node.outputs[1].ty = ArgType::Tensor(TensorType {
            rank: 2,
            static_shape: None, // shape is tracked and calculated at runtime
            elem_type: node.inputs[2].ty.elem_type().clone(),
        });
    }

    if node.outputs.len() > 2 {
        node.outputs[2].ty = ArgType::Tensor(TensorType {
            rank: 2,
            static_shape: None, // shape is tracked and calculated at runtime
            elem_type: node.inputs[2].ty.elem_type().clone(),
        });
    }
}
