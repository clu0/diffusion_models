"""
Code to train a simple diffusion model, i.e. the epsilon_theta(x, t) model

TODO: substitute the model for an actual UNet
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import numpy.typing as npt


T: int = 1000
beta_1: float = 1e-4
beta_T: float = 0.02
in_c: int = 3
in_h: int = 128
in_w: int = 128
lr: float = 6 * 1e-4
n_epoch: int = 200
batch_size: int = 32

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def get_beta_t(t: int) -> float:
    assert 1 <= t and t <= T, f"time index {t} must be in range [1, {T}]"
    return (beta_1 * (T - t) + beta_T * (t - 1)) / (T - 1)


betas: npt.NDArray[np.float_] = np.array([get_beta_t(t) for t in range(1, T + 1)])
alphas = np.cumprod(1 - betas)


class Unet(nn.Module):
    """
    TODO: swap with actual UNet
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        return x


class DiffusionTrainingDataset(Dataset):
    def __init__(self, data: npt.NDArray[np.float_]):
        # assume that the input data has already been preprocessed into shape (N, in_c, in_h, in_w)
        assert data.ndim == 4 and data.shape[1:] == (in_c, in_h, in_w)
        self.data = data

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
            np.random.normal(size=(1, in_c, in_h, in_w)),
            np.random.randint(1, T + 1),
        )


if __name__ == "__main__":
    # TODO substitute with some actual data
    train_dataloader = DataLoader(
        Dataset(np.random.rand(100, in_c, in_h, in_w)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = Unet()
    model.to(device=device)

    loss_fn = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for i in range(n_epoch):
        for batch in train_dataloader:
            data, noise, times = batch
            data, noise, times = (
                torch.from_numpy(data),
                torch.from_numpy(noise),
                torch.from_numpy(times),
            )
            data, noise, times = data.to(device), noise.to(device), times.to(device)
            optimizer.zero_grad()
            alpha_ts = alphas[times].reshape(-1, 1, 1, 1)
            x_ts = torch.sqrt(alpha_ts) * data + torch.sqrt(1 - alpha_ts) * noise
            pred = model(x_ts, times)
            loss_vals = loss_fn(pred, noise)
            optimizer.step()
        print(f"finished epoch {i}")

    # TODO save the model when we actually have a real model
    # torch.save(model.state_dict(), "model.pth")
