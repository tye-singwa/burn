
#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/lstm/lstm.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(3, 4) 

    def forward(self, x, hidden):
        out, next_hidden = self.lstm(x, hidden)
        out = out.unsqueeze(1)
        return out, next_hidden


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "lstm.onnx"
    test_input = torch.randn(1, 1, 3, device=device)
    test_hidden1 = torch.randn(1, 1, 4, device=device)
    test_hidden2 = torch.randn(1, 1, 4, device=device)
    # LayerNormalization only appeared in opset 17
    torch.onnx.export(model, (test_input, (test_hidden1, test_hidden2)), onnx_name, verbose=False, opset_version=17)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}, {test_hidden1}, {test_hidden2}")
    output = model.forward(test_input, (test_hidden1, test_hidden2))
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
