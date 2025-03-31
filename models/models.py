import torch
from torch import nn

class NN_v1(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN_v1, self).__init__()
        # self.dropout1 = nn.Dropout(0.2)
        self.mask = nn.Dropout(0.2)
        self.linear1 = nn.Linear(input_size, input_size * input_size)
        self.act1 = nn.ReLU()
        # self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(input_size * input_size, input_size * input_size)
        self.act2 = nn.ReLU()
        # self.dropout3 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(input_size * input_size, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x * self.mask(torch.ones_like(x.real))
        # x = self.dropout1(x)
        x = self.act1(self.linear1(x))
        x = x * self.mask(torch.ones_like(x.real))
        # x = self.dropout2(x)
        x = self.act2(self.linear2(x))
        x = x * self.mask(torch.ones_like(x.real))
        # x = self.dropout3(x)
        output = self.linear3(x)
        return output



