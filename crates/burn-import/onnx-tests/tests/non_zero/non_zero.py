#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/nonzero/nonzero.onnx
import torch
import torch.nn as nn
import onnx


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return x.nonzero().T


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Create test model
    model = Model()
    model.eval()
    device = torch.device("cpu")
    test_input = torch.tensor([-2, 0, -1, 0, 1, 2], device=device, dtype=torch.float)

    # Export to ONNX
    onnx_name = "non_zero.onnx"
    torch.onnx.export(
        model,
        test_input,
        onnx_name,
        opset_version=16,
        input_names=["input"],
        output_names=["non_zero_output"],
    )
    print(f"Finished exporting model to {onnx_name}")
    
    # Output some test data for use in the test
    print(f"Test input data of ones: {test_input}")
    print(f"Test input data shape of ones: {test_input.shape}")
    output = model.forward(test_input)
    print(f"Test output data shape: {output.shape}")
    print(f"Test output: {output}")


if __name__ == "__main__":
    main()
