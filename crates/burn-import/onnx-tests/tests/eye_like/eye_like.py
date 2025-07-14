#!/usr/bin/env python3

# used to generate model: eye_like.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def build_model(shape, **args):
    # Define the graph inputs and outputs
    input = onnx.helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, shape)
    output = onnx.helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, shape)

    # Create the EyeLike node
    eye_like = onnx.helper.make_node(
        "EyeLike",
        inputs=["input"],
        outputs=["output"],
        name="EyeLikeNode",
        **args
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        [eye_like],
        'EyeLikeModel',
        [input],
        [output],
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 16)],
        graph=graph,
        producer_name='ONNX_Generator',
    )

    return model


def export_onnx_model(file_name, test_input, **args):
    # Build model
    onnx_model = build_model(test_input.shape, **args)

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data:\n{repr(test_input)}")
    print(f"Test input data shape: {test_input.shape}")
    session = ReferenceEvaluator(file_name, verbose=1)
    test_output, = session.run(None, {"input": test_input})
    print(f"Test output:\n{repr(test_output)}")
    print(f"Test output shape: {test_output.shape}")


if __name__ == "__main__":
    # Set seed and precision
    np.random.seed(42)
    np.set_printoptions(precision=8)

    test_input = np.random.randn(2, 3).round(2)

    # Test k>0
    export_onnx_model(
        "eye_like_up.onnx", test_input,
        k=1, dtype=TensorProto.FLOAT
    )

    # Test k<0
    export_onnx_model(
        "eye_like_down.onnx", test_input,
        k=-1, dtype=TensorProto.FLOAT
    )

    # Test k=0
    export_onnx_model(
        "eye_like_0.onnx", test_input,
        k=0, dtype=TensorProto.INT32
    )
