"""
Code to train a simple diffusion model, i.e. the epsilon_theta(x, t) model

TODO: substitute the model for an actual UNet
"""
from typing import Tuple
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import numpy.typing as npt

from src.model import Unet


T: int = 1000
beta_1: float = 1e-4
beta_T: float = 0.02
# following 3 for mnist
in_c: int = 1
in_h: int = 32
in_w: int = 32

lr: float = 6 * 1e-4
n_epoch: int = 200
batch_size: int = 64

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def get_beta_t(t: int) -> float:
    assert 1 <= t and t <= T, f"time index {t} must be in range [1, {T}]"
    return (beta_1 * (T - t) + beta_T * (t - 1)) / (T - 1)


betas: npt.NDArray[np.float_] = np.array([get_beta_t(t) for t in range(1, T + 1)])
alphas: torch.Tensor = torch.from_numpy(np.cumprod(1 - betas))
alphas = alphas.to(device=device)


class DiffusionTrainingDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        # assume that the input data has already been preprocessed into shape (N, in_c, in_h, in_w)
        assert data.ndim == 4 and data.shape[1:] == (in_c, in_h, in_w)
        # using mnist right now, can just load the whole dataset to cuda
        self.data = data.to(device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], int]:
        """
        return the i-th data, along with a random normal of same shape, and a random time step in {1, ..., T}
        """
        return (
            self.data[idx, ...],
            torch.randn((in_c, in_h, in_w), device=device, dtype=torch.float32),
            torch.randint(0, T, (1,), device=device, dtype=torch.int32),
        )


if __name__ == "__main__":
    # Train with mnist
    mnist_upsampled: torch.Tensor = torch.load("mnist_upsampled.pt")
    mnist_upsampled = mnist_upsampled.float()
    train_dataloader = DataLoader(
        DiffusionTrainingDataset(mnist_upsampled),
        batch_size=batch_size,
        shuffle=True,
    )

    model = Unet(c_start=in_c)
    model.to(device=device)

    loss_fn = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    model.train()
    for i in range(n_epoch):
        epoch_start = time()
        for j, batch in enumerate(train_dataloader):
            if (j + 1) % 100 == 0:
                print(f"epoch {i}, batch {j}, loss {np.mean(train_losses[-100:])}")
            data, noise, times = batch
            #data, noise, times = (
            #    torch.from_numpy(data),
            #    torch.from_numpy(noise),
            #    torch.from_numpy(times),
            #)
            #data, noise, times = data.to(device), noise.to(device), times.to(device)
            optimizer.zero_grad()
            alpha_ts = alphas[times]
            alpha_ts = alpha_ts.view(alpha_ts.size(0), alpha_ts.size(1), 1, 1)
            x_ts = (torch.sqrt(alpha_ts) * data + torch.sqrt(1 - alpha_ts) * noise).float()
            pred = model(x_ts, times)
            loss_vals = loss_fn(pred, noise)
            loss_vals.backward()
            optimizer.step()

            train_losses.append(loss_vals.item())
        print(f"finished epoch {i}, took {time() - epoch_start} seconds")
        np.save("losses/train_losses.npy", np.array(train_losses))
        if (i + 1) % 20 == 0:
            torch.save(model.state_dict(), f"models/model_epoch_{i + 1}.pth")

