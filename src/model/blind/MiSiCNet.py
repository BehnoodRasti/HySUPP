"""
MiSiCNet simple PyTorch implementation
"""

import logging
import time

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .base import BlindUnmixingModel
from src.model.extractors import SiVM

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MiSiCNet(nn.Module, BlindUnmixingModel):
    def __init__(
        self,
        niters=8000,
        lr=0.001,
        exp_weight=0.99,
        lambd=100.0,
        kernel_size=3,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.kernel_sizes = [kernel_size] * 4 + [1]
        self.strides = [1, 1, 1, 1, 1]
        self.padding = [(k - 1) // 2 for k in self.kernel_sizes]

        self.lrelu_params = {
            "negative_slope": 0.1,
            "inplace": True,
        }

        self.niters = niters
        self.lr = lr
        self.exp_weight = exp_weight
        self.lambd = lambd

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)
        # MiSiCNet-like architecture
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[0]),
            nn.Conv2d(self.L, 256, self.kernel_sizes[0], stride=self.strides[0]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[1]),
            nn.Conv2d(256, 256, self.kernel_sizes[1], stride=self.strides[1]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layerskip = nn.Sequential(
            nn.ReflectionPad2d(self.padding[-1]),
            nn.Conv2d(self.L, 4, self.kernel_sizes[-1], stride=self.strides[-1]),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer3 = nn.Sequential(
            # nn.BatchNorm2d(260),
            nn.ReflectionPad2d(self.padding[2]),
            nn.Conv2d(260, 256, self.kernel_sizes[2], stride=self.strides[2]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer4 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[3]),
            nn.Conv2d(256, self.p, self.kernel_sizes[3], stride=self.strides[3]),
            nn.BatchNorm2d(self.p),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.decoder = nn.Linear(
            self.p,
            self.L,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.layer1(x)
        xskip = self.layerskip(x)
        xcat = torch.cat([x1, xskip], dim=1)
        abund = self.softmax(self.layer4(self.layer3(xcat)))
        abund_reshape = torch.transpose(abund.squeeze().view(-1, self.H * self.W), 0, 1)
        img = self.decoder(abund_reshape)
        return abund_reshape, img

    def loss(self, target, output):
        N, L = output.shape

        target_reshape = target.squeeze().reshape(L, N)
        fit_term = 0.5 * torch.linalg.norm(target_reshape.t() - output, "fro") ** 2

        O = target_reshape.mean(1, keepdims=True)
        reg_term = torch.linalg.norm(self.decoder.weight - O, "fro") ** 2

        return fit_term + self.lambd * reg_term

    def compute_endmembers_and_abundances(self, Y, p, H, W, seed=0, *args, **kwargs):
        tic = time.time()
        logger.debug("Solving started...")

        L, N = Y.shape

        # Hyperparameters
        self.L = L  # number of channels
        self.p = p  # number of endmembers
        self.H = H  # number of lines
        self.W = W  # number of samples per line

        self.init_architecture(seed=seed)

        # Initialize endmembers using SiVM extractor
        extractor = SiVM()
        Ehat = extractor.extract_endmembers(
            Y,
            p,
            seed=seed,
        )
        self.decoder.weight.data = torch.Tensor(Ehat)

        l, h, w = self.L, self.H, self.W

        Y = torch.Tensor(Y)
        Y = Y.view(1, l, h, w)

        self = self.to(self.device)
        Y = Y.to(self.device)

        noisy_input = torch.rand_like(Y)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        progress = tqdm(range(self.niters))
        for ii in progress:
            optimizer.zero_grad()

            abund, output = self(noisy_input)

            if ii == 0:
                out_avg = abund.detach()
            else:
                out_avg = out_avg * self.exp_weight + abund.detach() * (
                    1 - self.exp_weight
                )

            # Reshape data
            loss = self.loss(Y, output)

            progress.set_postfix_str(f"loss={loss.item():.3e}")

            loss.backward()
            optimizer.step()
            # Enforce physical constraints on endmembers
            self.decoder.weight.data[self.decoder.weight <= 0] = 0
            self.decoder.weight.data[self.decoder.weight >= 1] = 1

        Ahat = out_avg.cpu().T.numpy()
        Ehat = self.decoder.weight.detach().cpu().numpy()
        self.time = time.time() - tic
        logger.info(self.print_time())

        return Ehat, Ahat


def check_model():
    from src.data.base import HSI

    hsi = HSI("Sim1")
    print(hsi)

    model = MiSiCNet(hsi=hsi, niters=1000, lambd=0.1)
    model.compute_endmembers_and_abundances(hsi.Y, hsi.p, seed=42)


if __name__ == "__main__":
    check_model()
