"""
EndNet simple PyTorch implementation
"""

import logging
import time
from math import pi

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .base import BlindUnmixingModel
from src.model.extractors import VCA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EndNet(nn.Module, BlindUnmixingModel):
    def __init__(
        self,
        hsi,
        num_spectra=4000,
        batch_size=64,
        lr=0.001,
        epochs=40,
        lambda0=0.01,
        lambda1=50.0,
        weight_decay=1e-5,
        noise_std=0.3,
        dropout_rate=0.1,
        lr_beta_1=0.7,
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

        self.hsi = hsi
        self.num_spectra = num_spectra
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.lr_beta_1 = lr_beta_1

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)

        # Extract encoder/decoder weight
        extractor = VCA()
        Ehat = torch.Tensor(extractor.extract_endmembers(hsi=self.hsi, seed=seed))

        self.hidden = nn.Linear(  # TODO Too simple for now
            self.L,
            self.p,
            bias=False,
        )
        self.masked_noise = GaussianNoise(self.noise_std)
        self.BN_scaleless = nn.BatchNorm1d(
            self.p,
            affine=False,  # TODO Remove learnable parameters altogether
        )
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.decoder = nn.Linear(
            self.p,
            self.L,
            bias=False,
        )
        self.decoder.weight.data = Ehat.clone()
        self.hidden.weight.data = Ehat.T.clone()

    @staticmethod
    def SAD(x, xhat):
        return torch.acos(-F.cosine_similarity(x, xhat, dim=1))

    def C(self, x, xhat):
        return 1.0 - self.SAD(x, xhat) / pi

    def forward(self, x):
        code = self.masked_noise(x)
        code = self.hidden(code)
        code = self.BN_scaleless(code)
        code = self.dropout(code)
        code = F.relu(code)
        abunds = F.softmax(code, dim=1)
        output = self.decoder(abunds)
        return abunds, output

    def loss(self, target, output):
        assert target.shape == output.shape
        MSE = F.mse_loss(target, output)
        KL = -torch.log(self.C(target, output).mean())
        return self.lambda0 / 2.0 * MSE + self.lambda1 * KL

    def freeze_hidden(self, freeze=True):
        self.hidden.weight.requires_grad = not freeze

    def freeze_decoder(self, freeze=True):
        self.decoder.weight.requires_grad = not freeze

    def compute_endmembers_and_abundances(self, Y, p, seed=0, *args, **kwargs):
        tic = time.time()
        logger.debug("Solving started...")

        self.init_architecture(seed=seed)

        data = Y.T
        training_data = torch.Tensor(
            data[np.random.randint(0, data.shape[0], self.num_spectra)]
        )

        # Send model to GPU
        self = self.to(self.device)
        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.lr_beta_1, 0.999),
            weight_decay=self.weight_decay,
        )

        # Dataloader
        dataloader = torch.utils.data.DataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=True,
        )

        breakpoint()
        progress = tqdm(range(self.epochs))
        for ee in progress:
            # Alternate training between encoder/decoder
            for ii in range(2):
                self.freeze_decoder(freeze=True)
                self.freeze_hidden(freeze=False)
                optimizer = self.fit(dataloader, optimizer, progress)
            for jj in range(1):
                self.freeze_decoder(freeze=False)
                self.freeze_hidden(freeze=True)
                optimizer = self.fit(dataloader, optimizer, progress)

        # # Get final abundances and endmembers
        self.eval()

        Y_eval = torch.Tensor(data).to(self.device)

        abunds, _ = self(Y_eval)

        Ahat = abunds.T.detach().cpu().numpy()
        Ehat = self.decoder.weight.data.detach().cpu().numpy()

        self.time = time.time() - tic
        logger.info(self.print_time())

        breakpoint()

        return Ehat, Ahat

    def fit(self, dataloader, optimizer, progress):
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
            # Enforce physical constraints on endmembers
            self.decoder.weight.data[self.decoder.weight <= 0] = 0
            self.decoder.weight.data[self.decoder.weight >= 1] = 1

        progress.set_postfix_str(f"loss={running_loss:.3e}")
        return optimizer


class GaussianNoise(nn.Module):
    """
    Gaussian noise regularizer
    """

    def __init__(self, noise_std=0.3):
        super().__init__()
        self.noise_std = noise_std
        self.register_buffer("noise", torch.tensor(0))

    def forward(self, x):
        if self.training and self.noise_std != 0:
            mask = F.dropout(x, 0.4)
            scale = self.noise_std * x.detach()
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + mask * sampled_noise
        return x


def check_model():
    from src.data.base import HSI

    hsi = HSI("Sim1")
    print(hsi)

    model = EndNet(hsi=hsi, epochs=10)
    model.compute_endmembers_and_abundances(hsi.Y, hsi.p, seed=42)


if __name__ == "__main__":
    check_model()
