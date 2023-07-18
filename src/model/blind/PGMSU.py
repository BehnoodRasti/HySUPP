"""
Probabilistic Generative Model for Hyperspectral Unmixing (PGMSU)
simple PyTorch implementation
based on https://github.com/shuaikaishi/PGMSU
"""

import logging
import time

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from .base import BlindUnmixingModel
from src.model.extractors import VCA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PGMSU(nn.Module, BlindUnmixingModel):
    def __init__(
        self,
        z_dim=4,
        lr=1e-3,
        epochs=200,
        lambda_kl=0.1,
        lambda_sad=0,
        lambda_vol=0.5,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.z_dim = z_dim  # VAE code size
        self.lr = lr
        self.epochs = epochs

        self.lambda_kl = lambda_kl
        self.lambda_sad = lambda_sad
        self.lambda_vol = lambda_vol

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available else "cpu",
        )

    def init_architecture(
        self,
        seed,
    ):

        # Set random seed
        torch.manual_seed(seed)
        # encoder z fc1 -> fc5
        self.fc1 = nn.Linear(self.L, 32 * self.p)
        self.bn1 = nn.BatchNorm1d(32 * self.p)

        self.fc2 = nn.Linear(32 * self.p, 16 * self.p)
        self.bn2 = nn.BatchNorm1d(16 * self.p)

        self.fc3 = nn.Linear(16 * self.p, 4 * self.p)
        self.bn3 = nn.BatchNorm1d(4 * self.p)

        self.fc4 = nn.Linear(4 * self.p, self.z_dim)
        self.fc5 = nn.Linear(4 * self.p, self.z_dim)

        # encoder a
        self.fc9 = nn.Linear(self.L, 32 * self.p)
        self.bn9 = nn.BatchNorm1d(32 * self.p)

        self.fc10 = nn.Linear(32 * self.p, 16 * self.p)
        self.bn10 = nn.BatchNorm1d(16 * self.p)

        self.fc11 = nn.Linear(16 * self.p, 4 * self.p)
        self.bn11 = nn.BatchNorm1d(4 * self.p)

        self.fc12 = nn.Linear(4 * self.p, 4 * self.p)
        self.bn12 = nn.BatchNorm1d(4 * self.p)

        self.fc13 = nn.Linear(4 * self.p, self.p)

        # decoder
        self.fc6 = nn.Linear(self.z_dim, 4 * self.p)
        self.bn6 = nn.BatchNorm1d(4 * self.p)

        self.fc7 = nn.Linear(4 * self.p, 64 * self.p)
        self.bn7 = nn.BatchNorm1d(64 * self.p)

        self.fc8 = nn.Linear(64 * self.p, self.L * self.p)

    def encoder_z(self, x):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)

        h1 = self.fc2(h1)
        h1 = self.bn2(h1)
        h1 = F.relu(h1)

        h1 = self.fc3(h1)
        h1 = self.bn3(h1)
        h1 = F.relu(h1)

        mu = self.fc4(h1)
        log_var = self.fc5(h1)
        return mu, log_var

    def encoder_a(self, x):
        h1 = self.fc9(x)
        h1 = self.bn9(h1)
        h1 = F.relu(h1)

        h1 = self.fc10(h1)
        h1 = self.bn10(h1)
        h1 = F.relu(h1)

        h1 = self.fc11(h1)
        h1 = self.bn11(h1)
        h1 = F.relu(h1)

        h1 = self.fc12(h1)
        h1 = self.bn12(h1)
        h1 = F.relu(h1)

        h1 = self.fc13(h1)

        a = F.softmax(h1, dim=1)
        return a

    def reparametrize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=self.device)
        return mu + eps * std

    def decoder(self, z):
        h1 = self.fc6(z)
        h1 = self.bn6(h1)
        h1 = F.relu(h1)

        h1 = self.fc7(h1)
        h1 = self.bn7(h1)
        h1 = F.relu(h1)

        h1 = self.fc8(h1)
        em = torch.sigmoid(h1)
        return em

    def forward(self, inputs):
        mu, log_var = self.encoder_z(inputs)
        a = self.encoder_a(inputs)

        # reparametrization trick
        z = self.reparametrize(mu, log_var)
        em = self.decoder(z)

        em_tensor = em.view(-1, self.p, self.L)
        a_tensor = a.view(-1, 1, self.p)
        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)

        return y_hat, mu, log_var, a, em_tensor

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find("Linear") != -1:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def compute_endmembers_and_abundances(self, Y, p, seed=0, *args, **kwargs):
        tic = time.time()
        logger.debug("Solving started...")

        L, N = Y.shape
        # Hyperparameters
        self.L = L  # number of channels
        self.p = p  # number of endmembers
        self.N = N  # number of pixels

        self.batchsz = self.N // 10

        self.init_architecture(seed=seed)

        # Process data
        train_db = torch.utils.data.TensorDataset(torch.Tensor(Y.T))
        dataloader = torch.utils.data.DataLoader(
            train_db,
            batch_size=self.batchsz,
            shuffle=True,
        )

        # Endmembers initialization
        extractor = VCA()
        EM = extractor.extract_endmembers(
            Y,
            p,
            seed=seed,
        )
        EM = EM.T
        EM = np.reshape(EM, [1, EM.shape[0], EM.shape[1]]).astype("float32")
        EM = torch.Tensor(EM).to(self.device)

        # Model initialization
        self = self.to(self.device)
        self.apply(self.weights_init)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        progress = tqdm(range(self.epochs))

        self.train()
        for ee in progress:

            for _, y in enumerate(dataloader):
                y = y[0].to(self.device)

                y_hat, mu, log_var, a, em_tensor = self(y)

                loss_rec = ((y_hat - y) ** 2).sum() / y.shape[0]

                kl_div = -0.5 * (log_var + 1 - mu**2 - log_var.exp())
                kl_div = kl_div.sum() / y.shape[0]
                # KL balance of VAE
                kl_div = torch.max(kl_div, torch.Tensor([0.2]).to(self.device))

                if ee < self.epochs // 2:
                    # pre-train process
                    loss_vca = (em_tensor - EM).square().sum() / y.shape[0]
                    loss = loss_rec + self.lambda_kl * kl_div + 0.1 * loss_vca

                else:
                    # training process
                    # constraint 1 min_vol of EMs
                    em_bar = em_tensor.mean(dim=1, keepdim=True)
                    loss_minvol = ((em_tensor - em_bar) ** 2).sum() / (
                        y.shape[0] * self.p * self.L
                    )

                    # constraint 2 SAD for same materials
                    em_bar = em_tensor.mean(dim=0, keepdim=True)
                    aa = (em_tensor * em_bar).sum(dim=2)
                    em_bar_norm = em_bar.square().sum(dim=2).sqrt()
                    em_tensor_norm = em_tensor.square().sum(dim=2).sqrt()

                    sad = torch.acos(
                        aa / ((em_bar_norm + 1e-6) * (em_tensor_norm + 1e-6))
                    )
                    loss_sad = sad.sum() / (y.shape[0] * self.p)
                    loss = (
                        loss_rec
                        + self.lambda_kl * kl_div
                        + self.lambda_vol * loss_minvol
                        + self.lambda_sad * loss_sad
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.eval()
        with torch.no_grad():
            y_hat, mu, log_var, A, E = self(torch.Tensor(Y.T).to(self.device))
            # breakpoint()
            Ahat = A.cpu().numpy().T
            # E shape => [N, p, L] ??
            Ehat = E.cpu().numpy().mean(0).T

        self.time = time.time() - tic
        logger.info(self.print_time())

        return Ehat, Ahat
