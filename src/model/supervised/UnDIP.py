"""
UnDIP simple PyTorch implementation
"""

import logging
import time

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .base import SupervisedUnmixingModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class UnDIP(nn.Module, SupervisedUnmixingModel):
    def __init__(
        self,
        hsi,
        niters=3000,
        lr=0.001,
        exp_weight=0.99,
        noisy_input=True,
    ):
        super().__init__()

        # Hyperparameters
        self.L = hsi.L  # number of channels
        self.p = hsi.p  # number of endmembers
        self.H = hsi.H  # number of lines
        self.W = hsi.W  # number of samples per line

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.kernel_sizes = [3, 3, 3, 1, 1, 1]
        self.strides = [2, 1, 1, 1, 1, 1]
        self.padding = [(k - 1) // 2 for k in self.kernel_sizes]

        self.lrelu_params = {
            "negative_slope": 0.1,
            "inplace": True,
        }

        self.niters = niters
        self.lr = lr
        self.exp_weight = exp_weight
        self.noisy_input = noisy_input

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

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.layerskip = nn.Sequential(
            nn.ReflectionPad2d(self.padding[-1]),
            nn.Conv2d(self.L, 4, self.kernel_sizes[-1], stride=self.strides[-1]),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(260),
            nn.ReflectionPad2d(self.padding[2]),
            nn.Conv2d(260, 256, self.kernel_sizes[2], stride=self.strides[2]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer4 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[3]),
            nn.Conv2d(256, 256, self.kernel_sizes[3], stride=self.strides[3]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer5 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[4]),
            nn.Conv2d(256, self.p, self.kernel_sizes[4], stride=self.strides[4]),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.upsample(self.layer2(self.layer1(x)))
        xskip = self.layerskip(x)
        xcat = self.custom_cat(x1, xskip)
        out = self.softmax(self.layer5(self.layer4(self.layer3(xcat))))
        return out

    @staticmethod
    def custom_cat(x1, xskip):
        inputs = [x1, xskip]
        inputs_shape2 = [x.shape[2] for x in inputs]
        inputs_shape3 = [x.shape[3] for x in inputs]
        if np.all(np.array(inputs_shape2) == min(inputs_shape2)) and np.all(
            np.array(inputs_shape3) == min(inputs_shape3)
        ):
            inputs_ = inputs
        else:

            inputs_ = []

            target_shape2 = min(inputs_shape2)
            target_shape3 = min(inputs_shape3)

            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[
                        :,
                        :,
                        diff2 : diff2 + target_shape2,
                        diff3 : diff3 + target_shape3,
                    ]
                )

        return torch.cat(inputs_, dim=1)

    def compute_abundances(self, Y, E, seed=0, *args, **kwargs):
        tic = time.time()
        logger.debug("Solving started...")

        self.init_architecture(seed=seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        l, h, w = self.L, self.H, self.W

        Y = torch.Tensor(Y)
        Y = Y.view(1, l, h, w)

        self = self.to(self.device)
        Y = Y.to(self.device)
        E = torch.Tensor(E).to(self.device)

        noisy_input = torch.rand_like(Y) if self.noisy_input else Y

        progress = tqdm(range(self.niters))
        for ii in progress:
            optimizer.zero_grad()

            abund = self(noisy_input)

            if ii == 0:
                out_avg = abund.detach()
            else:
                out_avg = out_avg * self.exp_weight + abund.detach() * (
                    1 - self.exp_weight
                )

            # Reshape data
            loss = F.mse_loss(Y.view(-1, h * w), E @ abund.view(-1, h * w))

            progress.set_postfix_str(f"loss={loss.item():.3e}")

            loss.backward()
            optimizer.step()

        A = out_avg.cpu().numpy().reshape(-1, h * w)
        self.time = time.time() - tic
        logger.info(self.print_time())

        return A
