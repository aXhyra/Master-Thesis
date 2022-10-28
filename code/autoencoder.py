import torch
import torch.nn as nn
import sklearn.metrics as metrics
from tqdm.auto import tqdm
from utils import compute_auc_score

import numpy as np
from torchinfo import summary

from .bottleneck import Bottleneck
from .convolutional_decoder import ConvolutionalDecoder
from .convolutional_encoder import ConvolutionalEncoder
from .decoder import Decoder
from .encoder import Encoder


class Autoencoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size=128,
                 convolutional=True, dropout_rate=0.5, config=None, device=None,
                 bottleneck_activation=None):
        super(Autoencoder, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu") if device is None else device
        self.hidden_size = hidden_size
        self.in_channels = input_size[0]
        self.input_size = input_size[1:]
        self.batch_size = batch_size
        self.last_conv_size = (4, 4) if self.input_size == (28, 28) else (
            13, 13)

        if config is not None:
            input_size = config["input_size"]
            hidden_size = config["hidden_size"]
            output_size = config["output_size"]
            convolutional = config["convolutional"]
            dropout_rate = config["dropout_rate"]
        self.convolutional = convolutional
        if convolutional:
            self.encoder = ConvolutionalEncoder(dropout_rate=dropout_rate,
                                                in_channels=self.in_channels)
            if self.hidden_size > 0:
                self.bottleneck = Bottleneck((self.batch_size,
                                              self.last_conv_size[0],
                                              self.last_conv_size[1]),
                                             hidden_size,
                                             bottleneck_activation)
            self.decoder = ConvolutionalDecoder(
                self.input_size,
                dropout_rate=dropout_rate,
                out_channels=self.in_channels,
                last_conv_size=self.last_conv_size)
        else:
            self.encoder = Encoder(input_size, hidden_size, output_size,
                                   device=self.device)
            self.decoder = Decoder(input_size, hidden_size, output_size,
                                   device=self.device)

        self.threshold = 0
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "convolutional": convolutional,
            "dropout_rate": dropout_rate
        }

    def summary(self, input_size):
        return summary(self, input_size=input_size, device=self.device)

    def forward(self, x):
        encoder_out = self.encoder(x)
        if self.convolutional:
            if self.hidden_size > 0:
                bottleneck_out, encoded = self.bottleneck(encoder_out)
            else:
                bottleneck_out = torch.ones_like(
                    encoder_out.view(-1,
                                     int(np.prod((
                                         self.batch_size,
                                         4,
                                         4))))).to(self.device)
                encoded = torch.zeros(self.batch_size, 1).to(self.device)
            decoded = self.decoder(bottleneck_out)
        else:
            encoded = encoder_out
            decoded = self.decoder(encoder_out)
        return encoded, decoded

    def set_threshold(self, threshold_data_loader):
        self.eval()
        with torch.no_grad():
            for x, _ in tqdm(threshold_data_loader):
                x = x.to(self.device)
                encoded, y, _ = self(x)
                y_test = (x > 0.5).reshape(-1).cpu().detach().numpy().astype(
                    np.uint8).tolist()
                y_score = y.reshape(-1).cpu().detach().numpy().astype(
                    float).tolist()

                _, _, thresholds = metrics.roc_curve(y_test, y_score)
                self.threshold = thresholds[1]
