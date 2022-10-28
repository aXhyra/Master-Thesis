import torch
from torch import nn
import torch.nn.functional as F


class ConvolutionalEncoder(nn.Module):
    def __init__(self, dropout_rate=0.5, in_channels=1):
        super(ConvolutionalEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv0 = nn.Conv2d(in_channels, 16, 3)
        self.conv1 = nn.Conv2d(16, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 6, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv0(x))
        x = self.dropout(F.leaky_relu(self.conv1(x)))
        x = self.dropout(F.leaky_relu(self.conv2(x)))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x
