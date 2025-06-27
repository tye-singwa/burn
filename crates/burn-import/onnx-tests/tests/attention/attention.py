#!/usr/bin/env python3

# used to generate model: attention_*.onnx

import math
from typing import Any
import numpy as np
import onnx
import onnx.helper
from onnx import ModelProto, TensorProto, ValueInfoProto
from onnx.helper import make_tensor_value_info
from onnx.reference import ReferenceEvaluator


def build_and_save_onnx_model(
    *,
    name: str,
    file_name: str,
    inputs: list[ValueInfoProto],
    outputs: list[ValueInfoProto],
    attributes: dict[str, Any] = {},
):
    node_inputs = [input.name for input in inputs]
    node_outputs = [output.name for output in outputs]

    attention = onnx.helper.make_node(
        "Attention",
        inputs=node_inputs,
        outputs=node_outputs,
        name=f"{name}Node",
        **attributes,
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        nodes=[attention],
        name='AttentionModel',
        inputs=inputs,
        outputs=outputs,
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 23)],
        graph=graph,
        producer_name='ONNX_Generator',
    )

    # Ensure valid ONNX and save
    # onnx.checker.check_model(model)
    onnx.save(model, file_name)

    return model


# Build model for 3d MHA
def build_and_save_attention_mha_3d():
    # attributes for MHA
    is_causal = '0'
    kv_num_heads = q_num_heads = 3
    qk_matmul_output_mode = 0
    batch_size = 2
    head_size = 2
    v_head_size = 3
    q_sequence_length = 4
    q_hidden_size = q_num_heads * head_size
    kv_sequence_length = 5
    k_hidden_size = kv_num_heads * head_size
    v_hidden_size = kv_num_heads * v_head_size
    hidden_size = q_num_heads * v_head_size

    # shapes
    q_shape = (batch_size, q_sequence_length, q_hidden_size)
    k_shape = (batch_size, kv_sequence_length, k_hidden_size)
    v_shape = (batch_size, kv_sequence_length, v_hidden_size)

    # inputs
    q = make_tensor_value_info("q", TensorProto.FLOAT, q_shape)
    k = make_tensor_value_info("k", TensorProto.FLOAT, k_shape)
    v = make_tensor_value_info("v", TensorProto.FLOAT, v_shape)

    # outputs
    y = make_tensor_value_info(
        "y", TensorProto.FLOAT, (batch_size, q_sequence_length, hidden_size))

    # build model
    file_name = 'attention_mha_3d.onnx'
    onnx_model = build_and_save_onnx_model(
        name='MHAttention3d',
        file_name=file_name,
        inputs=[q, k, v],
        outputs=[y],
        attributes={
            'is_causal': is_causal,
            'kv_num_heads': kv_num_heads,
            'q_num_heads': q_num_heads,
            'qk_matmul_output_mode': qk_matmul_output_mode,
        }
    )

    # prepare test inputs
    test_q = np.random.randn(*q_shape).round(2)
    test_k = np.random.randn(*k_shape).round(2)
    test_v = np.random.randn(*v_shape).round(2)

    # output some test data for use in the test
    print(f"Test q:\n{repr(test_q)}")
    print(f"Test q shape: {test_q.shape}")
    print(f"Test k:\n{repr(test_k)}")
    print(f"Test k shape: {test_k.shape}")
    print(f"Test v:\n{repr(test_v)}")
    print(f"Test v shape: {test_v.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    test_y, = session.run(None, {
        "k": test_k, "v": test_v, "q": test_q
    })
    print(f"Test y:\n{repr(test_y)}")
    print(f"Test y shape: {test_y.shape}")


if __name__ == "__main__":
    # Set seed and precision
    np.random.seed(42)
    np.set_printoptions(precision=8)

    # build and save MHA 3d
    build_and_save_attention_mha_3d()