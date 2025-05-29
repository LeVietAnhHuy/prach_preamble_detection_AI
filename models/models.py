from torch import nn
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu
from modified_complexFunctions import ComplexDropout
from complextorch.nn.modules.linear import CVLinear
from complextorch.nn.modules.dropout import CVDropout
from complextorch.nn.modules.batchnorm import CVBatchNorm1d
from complextorch.nn.modules.activation.split_type_A import GeneralizedSplitActivation
from complextorch.nn.modules.activation.complex_relu import CPReLU

from complextorch.nn.modules.conv import Conv1d


class ComplexNN_v2(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNN_v2, self).__init__()
        self.complex_dropout1 = CVDropout(0.2)
        self.complex_linear1 = CVLinear(input_size, input_size * 10)
        self.complex_relu1 = CPReLU()

        self.complex_dropout2 = CVDropout(0.2)
        self.complex_linear2 = CVLinear(input_size * 10, input_size * 20)
        self.complex_relu2 = CPReLU()

        self.complex_dropout3 = CVDropout(0.2)
        self.complex_linear3 = CVLinear(input_size * 20, output_size)

    def forward(self, x):
        x = self.complex_dropout1(x)
        x = self.complex_linear1(x)
        x = self.complex_relu1(x)

        x = self.complex_dropout2(x)
        x = self.complex_linear2(x)
        x = self.complex_relu2(x)

        x = self.complex_dropout3(x)
        x = self.complex_linear3(x)

        return x

class Complex_VGG11(nn.Module):
    def __init__(self, in_channels=1):
        super(Complex_VGG11, self).__init__()
        self.conv1d_1 = Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        





