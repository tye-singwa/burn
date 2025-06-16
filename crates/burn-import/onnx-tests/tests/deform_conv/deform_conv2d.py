#!/usr/bin/env python3

# used to generate model: deform_conv2d.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model():
    # Define the graph inputs and outputs
    input = onnx.helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [2, 4, 10, 15])
    output = onnx.helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [2, 3, 2, 11])

    scale = onnx.helper.make_tensor(
        'scale', TensorProto.FLOAT, [2, 2, 3, 5], np.random.rand(2, 2, 3, 5))
    offset = onnx.helper.make_tensor(
        'offset', TensorProto.FLOAT, [2, 30, 2, 11], np.random.rand(2, 30, 2, 11))
    # bias = onnx.helper.make_tensor(
    #     'bias', TensorProto.FLOAT, [11], np.random.rand(11))
    # mask = onnx.helper.make_tensor(
    #     'mask', TensorProto.FLOAT, [], np.random.rand(1, 1, 1, 1))

    # Create the DeformConv node
    deform_conv2d = onnx.helper.make_node(
        "DeformConv",
        inputs=["input", "scale", "offset"],
        outputs=["output"],
        name="DeformConv2dNode",
        dilations=[3, 1],
        group=2,
        kernel_shape=[3, 5],
        offset_group=1,
        strides=[2, 1],
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [deform_conv2d],
        'DeformConvModel',
        [input],
        [output],
        [scale, offset],
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 19)],
        graph=graph,
        producer_name='ONNX_Generator',
    )

    return model


if __name__ == "__main__":
    # Set seed and precision
    np.random.seed(42)
    np.set_printoptions(precision=8)

    # Build model
    test_input = np.arange(2 * 4 * 10 * 15, dtype=np.float32).reshape(2, 4, 10, 15)
    onnx_model = build_model()
    file_name = "deform_conv2d.onnx"

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator("deform_conv2d.onnx", verbose=1)
    test_output, = session.run(None, {"input": test_input})
    print(f"Test output:\n{repr(test_output)}")
    print(f"Test output shape: {test_output.shape}")
