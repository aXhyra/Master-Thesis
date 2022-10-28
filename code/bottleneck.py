import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, input_size, bottleneck_size, bottleneck_activation):
        super(Bottleneck, self).__init__()
        self.input = int(np.prod(input_size))
        self.output = input_size
        self.bottleneck_size = bottleneck_size
        self.bottleneck_activation = bottleneck_activation
        self.fc1 = nn.Linear(self.input, bottleneck_size)
        self.fc2 = nn.Linear(bottleneck_size, self.input)

    def forward(self, x):
        x = x.view(-1, self.input)
        if self.bottleneck_activation is not None:
            encoded = self.bottleneck_activation(self.fc1(x))
        else:
            encoded = self.fc1(x)
        x = self.fc2(encoded)
        x = x.view(-1, self.output[0], self.output[1], self.output[2])
        return x, encoded
