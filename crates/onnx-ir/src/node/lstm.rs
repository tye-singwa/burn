use crate::ir::{ArgType, Node, TensorType};

/// Configuration for the Lstm operation.
#[derive(Debug, Clone, PartialEq)]
pub struct LstmConfig {
    pub hidden_size: usize,
    pub input_size: usize,
}

pub fn lstm_config(node: &Node) -> LstmConfig {
    let mut hidden_size: Option<i64> = None;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "activation_alpha" => log::warn!("LSTM: activation_alpha is ignored"),
            "activation_beta" => log::warn!("LSTM: activation_beta is ignored"),
            "activations" => log::warn!("LSTM: activations is ignored"),
            "clip" => log::warn!("LSTM: clip is ignored"),
            "direction" => assert_eq!(
                value.clone().into_string().to_lowercase(),
                "forward",
                "LSTM: direction other than 'forward' is not supported"
            ),
            "hidden_size" => hidden_size = Some(value.clone().into_i64()),
            "input_forget" => assert_eq!(
                value.clone().into_i32(),
                0,
                "LSTM: input_forget other than 0 is not supported"
            ),
            "output_sequence" => assert_eq!(
                value.clone().into_i32(),
                1,
                "LSTM: output_sequence other than 1 is not supported"
            ),
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

    let input_size = weight_shape[0];

    LstmConfig {
        hidden_size: hidden_size.unwrap() as usize,
        input_size,
    }
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
