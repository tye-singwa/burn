use protobuf::Enum;

use crate::{ArgType, ElementType, Node, protos::tensor_proto::DataType};

/// Update output rank for EyeLike operations based on input rank.
pub fn eye_like_update_output(node: &mut Node) {
    log::debug!("EyeLike rank inference for node {}", node.name);

    let mut elem_type: ElementType = node.inputs.first().unwrap().ty.elem_type().clone();

    let dtype = node
        .attrs
        .get("dtype")
        .map(|val| DataType::from_i32(val.clone().into_i32()).unwrap())
        .unwrap_or(node.inputs.first());
    log::debug!("RandomLike dtype for {}: {:?}", node.name, dtype);
}

/// EyeLike (offset, elem_type) from the attributes of the node
pub fn eye_like_config(node: &Node) -> (i64, ElementType) {
    let mut offset: i64 = 0;
    let mut elem_type: ElementType = node.inputs.first().unwrap().ty.elem_type().clone();

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "k" => offset = value.clone().into_i64(),
            "dtype" => {
                let dtype = DataType::from_i32(value.clone().into_i32()).unwrap();
                elem_type = elem_type_from_dtype(dtype)
            }
            _ => panic!("Unexpected attribute for EyeLike: {key}"),
        }
    }

    (offset, elem_type)
}

fn elem_type_from_dtype(dtype: DataType) -> ElementType {
    match dtype {
        DataType::FLOAT16 => ElementType::Float16,
        DataType::FLOAT => ElementType::Float32,
        DataType::DOUBLE => ElementType::Float64,
        DataType::INT8 | DataType::INT16 => {
            log::warn!("EyeLike: Tensor with type {dtype:?} not supported output, assuming int32");
            ElementType::Int32
        }
        DataType::INT32 => ElementType::Int32,
        DataType::INT64 => ElementType::Int64,
        DataType::UINT8 | DataType::UINT16 | DataType::UINT32 => {
            log::warn!("EyeLike: Tensor with type {dtype:?} not supported output, assuming int32");
            ElementType::Int32
        }
        DataType::UINT64 => {
            log::warn!("EyeLike: Tensor with type {dtype:?} not supported output, assuming int64");
            ElementType::Int64
        }
        _ => panic!("EyeLike: Tensor with type {dtype:?} not supported output"),
    }
}
