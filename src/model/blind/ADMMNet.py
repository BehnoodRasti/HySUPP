"""
ADMMNet simple PyTorch implementation
"""

import logging
import time

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .base import BlindUnmixingModel

from src import EPS
from src.extract import VCA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class X_block(nn.Module):
    def __init__(self, L, p, A_init, mu, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.W = nn.Linear(L, p, bias=False)
        self.B = nn.Linear(p, p, bias=False)

        # init
        M = A_init.T @ A_init + mu * torch.eye(p)

        self.W.weight.data = torch.linalg.solve(M, A_init.T)
        self.B.weight.data = torch.linalg.solve(M, mu * torch.eye(p))

    def forward(self, y, z, d):
        return self.W(y) + self.B(z + d)


class D_block(nn.Module):
    def __init__(self, eta_init=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.eta = nn.Parameter(
            data=eta_init * torch.ones(1),
            requires_grad=True,
        )

    def forward(self, x, z, d):
        return d - self.eta * (x - z)


class Z_block(nn.Module):
    def __init__(self, p, theta_init=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # init
        self.theta = nn.Parameter(
            data=theta_init * torch.ones(p),
            requires_grad=True,
        )

    def forward(self, x, d):
        return F.relu(x - d - self.theta)


class ADMMNet(nn.Module, BlindUnmixingModel):
    def __init__(
        self,
        lr,
        epochs,
        batchsize,
        nblocks,
        lambd,
        mu,
        tied,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.nblocks = nblocks
        self.tied = tied

        # Hyperparameters
        self.lambd = lambd
        self.mu = mu

    def init_architecture(
        self,
        A_init,
        eta_init=1.0,
    ):

        self.x_blocks = nn.ModuleList()
        self.z_blocks = nn.ModuleList()
        self.d_blocks = nn.ModuleList()

        # NOTE this is for tied params

        if self.tied:
            x_block = X_block(self.L, self.p, A_init=A_init, mu=self.mu)
            z_block = Z_block(self.p, theta_init=self.lambd / self.mu)
            d_block = D_block(eta_init=eta_init)

            for _ in range(self.nblocks):
                self.x_blocks.append(x_block)
                self.z_blocks.append(z_block)
                self.d_blocks.append(d_block)

        else:
            for _ in range(self.nblocks):
                self.x_blocks.append(X_block(self.L, self.p, A_init=A_init, mu=self.mu))
                self.z_blocks.append(Z_block(self.p, theta_init=self.lambd / self.mu))
                self.d_blocks.append(D_block(eta_init=eta_init))

        self.decoder = nn.Linear(
            self.p,
            self.L,
            bias=False,
        )
        self.decoder.weight.data = A_init

    def forward(self, y):
        bs, l = y.shape
        z = torch.zeros((bs, self.p)).to(self.device)
        d = torch.zeros((bs, self.p)).to(self.device)
        for ii in range(self.nblocks):
            x = self.x_blocks[ii](y, z, d)
            z = self.z_blocks[ii](x, d)
            d = self.d_blocks[ii](x, z, d)

        abund = z
        abund = abund / (abund.sum(1, keepdims=True) + EPS)
        output = self.decoder(abund)
        return abund, output

    def compute_endmembers_and_abundances(self, Y, p, *args, **kwargs):

        tic = time.time()
        logger.debug("Solving started...")

        L, N = Y.shape

        # Hyperparameters
        self.L = L
        self.p = p

        # endmembers initialization
        extractor = VCA()
        Ehat = extractor.extract_endmembers(Y, p)
        A_init = torch.Tensor(Ehat)
        self.init_architecture(A_init=A_init)

        self = self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_db = torch.utils.data.TensorDataset(torch.Tensor(Y.T))
        dataloader = torch.utils.data.DataLoader(
            train_db,
            batch_size=self.batchsize,
            shuffle=True,
        )

        progress = tqdm(range(self.epochs))
        self.train()

        for ii in progress:
            for x, y in enumerate(dataloader):
                y = y[0].to(self.device)

                abund, output = self(y)

                loss = F.mse_loss(y, output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Enforce non-negativity on endmembers
                self.decoder.weight.data[self.decoder.weight <= 0] = 0
                self.decoder.weight.data[self.decoder.weight >= 1] = 1

        self.eval()
        with torch.no_grad():
            abund, _ = self(torch.Tensor(Y.T).to(self.device))
            Ahat = abund.cpu().numpy().T
            Ehat = self.decoder.weight.detach().cpu().numpy()

        self.time = time.time() - tic
        logger.info(self.print_time())

        return Ehat, Ahat
