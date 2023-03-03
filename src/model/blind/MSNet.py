"""
Multi-stage convolutional autoencoder network for hyperspectral unmixing
simple PyTorch implementation

Source: https://github.com/yuyang95/JAG-MSNet
"""

import logging
import time
from math import ceil

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .base import BlindUnmixingModel
from src.model.extractors import VCA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MSNet(nn.Module, BlindUnmixingModel):
    def __init__(
        self,
        hsi,
        epochs=800,
        alpha=0.1,
        beta=0.03,
        drop_out=0.2,
        learning_rate=0.03,
        weight_decay=1e-4,
        step_size=30,
        gamma=0.6,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Hyperparameters
        self.L = hsi.L  # number of channels
        self.p = hsi.p  # number of endmembers
        self.H = hsi.H  # number of lines
        self.W = hsi.W  # number of samples per line
        self.N = hsi.N  # number of pixels

        self.epochs = epochs

        self.alpha = alpha
        self.beta = beta
        self.drop_out = drop_out
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        k = torch.Tensor([[0.05, 0.25, 0.40, 0.25, 0.05]])
        kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(self.p, 1, 1, 1)
        self.kernel = kernel.to(self.device)
        self.down22 = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.eps = 1e-3

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def edge_loss(self, x, y):
        x = self.down22(x)
        x_laplace = self.laplacian_kernel(x)
        y_laplace = self.laplacian_kernel(y)
        diff = x_laplace - y_laplace
        return (torch.sqrt(diff**2 + self.eps**2)).mean()

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.L + self.p, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(self.drop_out),
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(self.drop_out),
            nn.Conv2d(48, self.p, kernel_size=3, stride=1, padding=1),
        )

        self.downsampling22 = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.downsampling44 = nn.AvgPool2d(4, 4, ceil_mode=True)

        self.layer2 = nn.Sequential(
            nn.Conv2d(self.L + self.p, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(self.drop_out),
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(self.drop_out),
            nn.Conv2d(48, self.p, kernel_size=3, stride=1, padding=1),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(self.L, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(self.drop_out),
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(self.drop_out),
            nn.Conv2d(48, self.p, kernel_size=3, stride=1, padding=1),
        )

        self.softmax = nn.Softmax(dim=1)

        self.transconv = nn.Conv2d(self.p, self.p, kernel_size=1, stride=1)
        self.transconv2 = nn.Conv2d(self.L, self.L, kernel_size=1, stride=1)

        self.decoderlayer4 = nn.Conv2d(self.p, self.L, kernel_size=1, bias=False)
        self.decoderlayer5 = nn.Conv2d(self.p, self.L, kernel_size=1, bias=False)
        self.decoderlayer6 = nn.Conv2d(self.p, self.L, kernel_size=1, bias=False)

    def forward(self, x):
        # Layer 3
        down44 = self.downsampling44(x)
        layer3out = self.layer3(down44)

        en_result3 = self.softmax(layer3out)
        de_result3 = self.decoderlayer6(en_result3)

        translayer3 = F.interpolate(
            layer3out,
            (ceil(self.H / 2), ceil(self.W / 2)),
            mode="bilinear",
        )
        translayer3 = self.transconv(translayer3)

        # Layer 2
        down22 = self.downsampling22(x)
        convlayer2 = self.transconv2(down22)
        layer2in = torch.cat((convlayer2, translayer3), 1)
        layer2out = self.layer2(layer2in)

        en_result2 = self.softmax(layer2out)
        de_result2 = self.decoderlayer5(en_result2)

        translayer2 = F.interpolate(
            layer2out,
            (self.H, self.W),
            mode="bilinear",
        )
        translayer2 = self.transconv(translayer2)

        # Layer 1
        convlayer1 = self.transconv2(x)
        layer1in = torch.cat((convlayer1, translayer2), 1)
        layer1out = self.layer1(layer1in)

        # Layer out
        en_result1 = self.softmax(layer1out)
        de_result1 = self.decoderlayer4(en_result1)
        return (
            en_result1,
            de_result1,
            en_result2,
            de_result2,
            en_result3,
            de_result3,
            down22,
            down44,
        )

    @staticmethod
    def reconstruction_SAD_loss(output, target):
        assert output.shape == target.shape

        _, band, h, w = output.shape
        output = torch.reshape(output, (band, h * w))
        target = torch.reshape(target, (band, h * w))
        return torch.acos(torch.cosine_similarity(output, target, dim=0)).mean()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(img, self.kernel, groups=n_channels)

    def compute_endmembers_and_abundances(
        self,
        Y,
        p,
        seed=0,
        *args,
        **kwargs,
    ):
        tic = time.time()
        logger.debug("Solving started...")

        self.init_architecture(seed=seed)

        extractor = VCA()
        Ehat = extractor.extract_endmembers(
            Y,
            p,
            seed=seed,
        )
        Einit = torch.Tensor(Ehat).unsqueeze(2).unsqueeze(3)
        self.decoderlayer4.weight.data = Einit
        self.decoderlayer5.weight.data = Einit
        self.decoderlayer6.weight.data = Einit

        l, h, w = self.L, self.H, self.W

        Y = torch.Tensor(Y)
        Y = Y.view(1, l, h, w)

        self = self.to(self.device)
        Y = Y.to(self.device)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )

        progress = tqdm(range(self.epochs))

        self.train()
        for ii in progress:

            (
                en_abund,
                reconst_res,
                en_abund2,
                reconst_res2,
                en_abund3,
                reconst_res3,
                Y2,
                Y3,
            ) = self(Y)

            sad1 = self.reconstruction_SAD_loss(Y, reconst_res)
            sad2 = self.reconstruction_SAD_loss(Y2, reconst_res2)
            sad3 = self.reconstruction_SAD_loss(Y3, reconst_res3)
            A = sad1 + sad2 + sad3

            mse1 = F.mse_loss(Y, reconst_res)
            mse2 = F.mse_loss(Y2, reconst_res2)
            mse3 = F.mse_loss(Y3, reconst_res3)
            B = mse1 + mse2 + mse3

            edge1 = self.edge_loss(en_abund2, en_abund3)
            edge2 = self.edge_loss(en_abund, en_abund2)
            C = edge1 + edge2

            loss = A + self.alpha * B + self.beta * C

            progress.set_postfix_str(f"loss={loss.item():.3e}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        self.eval()
        (
            en_abund,
            reconst_res,
            en_abund2,
            reconst_res2,
            en_abund3,
            reconst_res3,
            Y2,
            Y3,
        ) = self(Y)

        self.time = time.time() - tic
        logger.info(self.print_time())

        Ahat = en_abund.squeeze(0).reshape(self.p, self.N).detach().cpu().numpy()
        Ehat = (
            self.decoderlayer4.weight.data.squeeze(-1)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        return Ehat, Ahat
