"""
A 3D-CNN Framework for Hyperspectral Unmixing with Spectral Variability (3DCNN-var)
simple PyTorch implementation
based on https://github.com/zhaomin0101/3DCNN-var
"""

import logging
import time

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .base import BlindUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ThreeDCNNvar(nn.Module, BlindUnmixingModel):
    def __init__(self):
        super().__init__()

        # Hyperparameters
        self.L = hsi.L  # number of channels
        self.p = hsi.p  # number of endmembers

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available else "cpu",
        )

        def init_architecture(
            self,
            seed,
        ):
            # Set random seed
            torch.manual_seed(seed)

            self.encoder_cnn = nn.Sequential(
                nn.Conv3d(
                    in_channels=1,
                    out_channels=128,
                    kernel_size=(3, 3, 6),
                    stride=(1, 1, 2),
                    padding=(1, 1, 0),
                    bias=False,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=(3, 3, 4),
                    stride=(1, 1, 2),
                    padding=(1, 1, 0),
                    bias=False,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=(3, 3, 5),
                    stride=(1, 1, 2),
                    padding=(0, 0, 0),
                    bias=False,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=32,
                    out_channels=16,
                    kernel_size=(1, 1, 3),
                    stride=(1, 1, 2),
                    padding=(0, 0, 0),
                    bias=False,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=16,
                    out_channels=8,
                    kernel_size=(1, 1, 4),
                    stride=(1, 1, 2),
                    padding=(0, 0, 0),
                    bias=False,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels=8,
                    out_channels=self.p,
                    kernel_size=(1, 1, 3),  # self.p here?
                    stride=(1, 1, 1),
                    padding=(0, 0, 0),
                    bias=False,
                ),
            )

            self.decoder_linear = nn.Linear(self.p, self.p * self.L, bias=False)

            self.decoder_nonlinear = nn.Sequential(
                nn.Linear(self.p * self.L, self.L, bias=True),
                nn.Sigmoid(),
                nn.Linear(self.L, self.L, bias=True),
                nn.Sigmoid(),
                nn.Linear(self.L, self.L, bias=True),
            )

        def forward(self, x):
            x = torch.reshape(x, (-1, 1, 3, 3, self.L))
            out_encoder = self.encoder_cnn(x)
            out_encoder = torch.reshape(out_encoder, (-1, self.p))
            out_encoder = out_encoder.abs()
            out_encoder = out_encoder.t() / out_encoder.sum(1)
            out_encoder = out_encoder.t()
            out_linear = self.decoder_linear(out_encoder)
            out_nonlinear = self.decoder_nonlinear(out_linear)

            return out_linear, out_nonlinear, out_encoder

        def get_endmember(self, x):
            endmember = self.decoder_linear(x)
            return endmember

        def get_abundance(self, x):
            x = self.encoder_cnn(x)
            x = torch.reshape(x, (-1, self.p))
            weights = self.encoder_cnn(x)
            weights = torch.reshape(weights, (-1, self.L))
            weights = weights.abs()
            weights = weights.t() / weights.sum(1)
            weights = weights.t()
            return weights
