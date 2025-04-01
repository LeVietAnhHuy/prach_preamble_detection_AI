import torch
from torch import nn
from complexPyTorch.complexLayers import ComplexBatchNorm1d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu
from modified_complexFunctions import ComplexDropout

class ComplexNN_v1(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(ComplexNN_v1, self).__init__()
        self.complex_dropout1 = ComplexDropout(device, 0.2)
        self.complex_linear1 = ComplexLinear(input_size, input_size * 10)

        self.complex_dropout2 = ComplexDropout(device, 0.2)
        self.complex_linear2 = ComplexLinear(input_size * 10, input_size * 20)

        self.complex_dropout3 = ComplexDropout(device, 0.2)
        self.complex_linear3 = ComplexLinear(input_size * 20, output_size)

    def forward(self, x):

        x = self.complex_dropout1(x)
        x = complex_relu(self.complex_linear1(x))

        x = self.complex_dropout2(x)
        x = complex_relu(self.complex_linear2(x))

        x = self.complex_dropout3(x)
        x = self.complex_linear3(x)

        return x



