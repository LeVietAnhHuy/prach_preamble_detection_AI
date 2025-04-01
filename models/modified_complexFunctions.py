from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate
import torch
from torch.nn import Module

class ComplexDropout(Module):
    def __init__(self, device, p=0.5):
        super().__init__()
        self.p = p
        self.device = device

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.device, self.p)
        else:
            return input

def complex_dropout(input, device, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    #mask = torch.ones_like(input).type(torch.float32)
    mask = torch.ones(*input.shape, dtype = torch.float32).to(device)
    mask = dropout(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    return mask*input