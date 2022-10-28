import torch
from torch import nn
import torch.nn.functional as F


class ConvolutionalDecoder(nn.Module):
    def __init__(self, output_size, dropout_rate=0.5, out_channels=1, last_conv_size=(4, 4)):
        super(ConvolutionalDecoder, self).__init__()
        self.output_size = output_size
        self.last_conv_size = last_conv_size
        self.dropout = nn.Dropout(dropout_rate)
        self.deconv0 = nn.ConvTranspose2d(128, 64, 6, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3)
        self.deconv3 = nn.ConvTranspose2d(16, out_channels, 3)

    def forward(self, x):
        x = x.view(-1, 128, self.last_conv_size[0], self.last_conv_size[1])
        x = F.leaky_relu(self.deconv0(x))
        x = self.dropout(F.leaky_relu(self.deconv1(x)))
        x = self.dropout(F.leaky_relu(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x[:, :, :self.output_size[0], :self.output_size[1]]
