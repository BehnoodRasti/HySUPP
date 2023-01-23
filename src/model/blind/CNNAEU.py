"""
CNNAEU simple PyTorch implementation
"""

import logging
import time

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.image import extract_patches_2d

from .base import BlindUnmixingModel
from src.model.extractors import SiVM, VCA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CNNAEU(nn.Module, BlindUnmixingModel):
    def __init__(
        self,
        hsi,
        scale=3.0,
        epochs=320,
        lr=0.0003,
        batch_size=15,
        patch_size=40,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Hyperparameters
        self.L = hsi.L  # number of channels
        self.p = hsi.p  # number of dictionary atoms
        self.H = hsi.H  # number of lines
        self.W = hsi.W  # number of samples per line

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.lrelu_params = {
            "negative_slope": 0.02,
            "inplace": True,
        }

        self.hsi = hsi
        self.scale = scale
        self.num_patches = int(250 * self.H * self.W * self.L / (307 * 307 * 162))
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patch_size = patch_size

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.L,
                48,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(48, self.p, kernel_size=1, bias=False),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(self.p),
            nn.Dropout2d(p=0.2),
        )

        self.decoder = nn.Conv2d(
            self.p,
            self.L,
            kernel_size=11,
            padding=5,
            padding_mode="reflect",
            bias=False,
        )

    def forward(self, x):
        code = self.encoder(x)
        abund = F.softmax(code * self.scale, dim=1)
        x_hat = self.decoder(abund)
        return abund, x_hat

    @staticmethod
    def loss(target, output):
        assert target.shape == output.shape

        dot_product = (target * output).sum(dim=1)
        target_norm = target.norm(dim=1)
        output_norm = output.norm(dim=1)
        sad_score = torch.clamp(dot_product / (target_norm * output_norm), -1, 1).acos()
        return sad_score.mean()

    def compute_endmembers_and_abundances(self, Y, p, seed=0, *args, **kwargs):
        tic = time.time()
        logger.debug("Solving started...")

        self.init_architecture(seed=seed)

        l, h, w = self.L, self.H, self.W

        Y_numpy = Y.reshape((l, h, w)).transpose((1, 2, 0))

        logger.info(f"{self.num_patches} patches extracted...")
        input_patches = extract_patches_2d(
            Y_numpy,
            max_patches=self.num_patches,
            patch_size=(self.patch_size, self.patch_size),
        )
        input_patches = torch.Tensor(input_patches.transpose((0, 3, 1, 2)))

        # Send model to GPU
        self = self.to(self.device)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(
            input_patches,
            batch_size=self.batch_size,
            shuffle=True,
        )

        progress = tqdm(range(self.epochs))
        for ee in progress:

            running_loss = 0
            for ii, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                optimizer.zero_grad()

                _, outputs = self(batch)

                # Reshape data
                loss = self.loss(batch, outputs)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

            progress.set_postfix_str(f"loss={running_loss:.3e}")

        # Get final abundances and endmembers
        self.eval()

        Y_eval = torch.Tensor(Y.reshape((1, l, h, w))).to(self.device)

        abund, _ = self(Y_eval)

        Ahat = abund.detach().cpu().numpy().reshape(self.p, self.H * self.W)
        Ehat = self.decoder.weight.data.mean((2, 3)).detach().cpu().numpy()

        self.time = time.time() - tic
        logger.info(self.print_time())

        return Ehat, Ahat


def check_model():
    from src.data.base import HSI

    hsi = HSI("Sim1")
    print(hsi)

    model = CNNAEU(hsi=hsi, epochs=10)
    model.compute_endmembers_and_abundances(hsi.Y, hsi.p, seed=42)


if __name__ == "__main__":
    check_model()
