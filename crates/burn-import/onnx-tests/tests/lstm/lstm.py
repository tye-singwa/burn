#!/usr/bin/env python3

# used to generate model: lstm.onnx

import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from onnxruntime import InferenceSession


def build_model():
    # Define dimensions
    seq_length = 5
    hidden_size = 3
    batch = 2
    input_size = 3

    # Define the graph inputs
    input = onnx.helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [seq_length, batch, input_size])
    initial_h = onnx.helper.make_tensor_value_info(
        'initial_h', TensorProto.FLOAT, [1, batch, hidden_size])
    initial_c = onnx.helper.make_tensor_value_info(
        'initial_c', TensorProto.FLOAT, [1, batch, hidden_size])

    # Define the graph outputs
    output = onnx.helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [seq_length, 1, batch, hidden_size])
    output_h = onnx.helper.make_tensor_value_info(
        'output_h', TensorProto.FLOAT, [1, batch, hidden_size])
    output_c = onnx.helper.make_tensor_value_info(
        'output_c', TensorProto.FLOAT, [1, batch, hidden_size])

    # Define the graph initialized
    weights_shape = [1, 4*hidden_size, input_size]
    weights = onnx.helper.make_tensor(
        'weights', TensorProto.FLOAT, weights_shape, np.random.rand(*weights_shape))
    recurrent_weights = onnx.helper.make_tensor(
        'recurrent_weights', TensorProto.FLOAT, weights_shape, np.random.rand(*weights_shape))
    bias_shape = [1, 8*hidden_size]
    bias = onnx.helper.make_tensor(
        'bias', TensorProto.FLOAT, bias_shape, np.random.rand(*bias_shape))

    # Define LstmNode attributes
    direction = 'forward'
    input_forget = 0
    layout = 0

    # Create the LstmNode node
    lstm = onnx.helper.make_node(
        "LSTM",
        inputs=["input", "weights", "recurrent_weights",
                "bias", "", "initial_h", "initial_c"],
        outputs=["output", "output_h", "output_c"],
        name="LstmNode",
        direction=direction,
        input_forget=input_forget,
        layout=layout,
        hidden_size=hidden_size,
    )

    # Create the graph
    graph = onnx.helper.make_graph(
        nodes=[lstm],
        name='LstmModel',
        inputs=[input, initial_h, initial_c],
        outputs=[output, output_h, output_c],
        initializer=[weights, recurrent_weights, bias],
    )

    # Create the model
    model = onnx.helper.make_model(
        opset_imports=[onnx.helper.make_operatorsetid("", 21)],
        graph=graph,
        producer_name='ONNX_Generator',
    )

    return model


if __name__ == "__main__":
    # Set seed and precision
    np.random.seed(42)
    np.set_printoptions(precision=8)

    # Build model
    test_input = np.random.randn(5, 2, 3).astype(np.float32).round(2)
    test_initial_h = np.random.randn(1, 2, 3).astype(np.float32).round(2)
    test_initial_c = np.random.randn(1, 2, 3).astype(np.float32).round(2)
    onnx_model = build_model()
    file_name = "lstm.onnx"

    # Ensure valid ONNX and save
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)
    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data:\n{repr(test_input)}")
    print(f"Test input data shape: {test_input.shape}")
    print(f"Test input initial hidden data:\n{repr(test_initial_h)}")
    print(f"Test input initial hidden data shape: {test_initial_h.shape}")
    print(f"Test input initial cell data:\n{repr(test_initial_c)}")
    print(f"Test input initial cell data shape: {test_initial_c.shape}")
    session = InferenceSession(onnx.load("lstm.onnx").SerializePartialToString(), verbose=1)
    test_output, test_output_h, test_output_c = session.run(['output', 'output_h', 'output_c'], {
        "input": test_input,
        "initial_h": test_initial_h,
        "initial_c": test_initial_c
    })
    print(f"Test output:\n{repr(test_output)}")
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output hidden:\n{repr(test_output_h)}")
    print(f"Test output hidden shape: {test_output_h.shape}")
    print(f"Test output cell:\n{repr(test_output_c)}")
    print(f"Test output cell shape: {test_output_c.shape}")
