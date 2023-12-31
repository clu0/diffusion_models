"""
I keep saying that diffusion inference is trivial, but we might as well do it.
"""
import torch
import torch.nn as nn
import numpy as np

from train import (
    betas,
    alphas,
    T,
    in_c,
    in_h,
    in_w,
)
from src.model import Unet

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def sample_next_step(
    x_t: torch.Tensor,  # (B, in_c, in_h, in_w)
    t: torch.Tensor,  # (B, 1)
    model: nn.Module,
) -> torch.Tensor:
    """
    sample x_{t-1} given x_t and x_0
    """
    assert torch.all(1 <= t) and torch.all(t < T), f"time index {t} must be in range [2, {T}]"
    beta_t = betas[t - 1]
    alpha_t = alphas[t - 1]
    alpha_t_1 = alphas[t - 2]

    # conditional mean and variance
    mu_t = (x_t - (beta_t / torch.sqrt(1 - alpha_t)) * model(x_t, t)) / torch.sqrt(
        1 - beta_t
    )
    sigma_t = torch.sqrt((1 - alpha_t_1) / (1 - alpha_t) * beta_t)

    return mu_t + sigma_t * torch.randn_like(mu_t)


if __name__ == "__main__":
    model = Unet()
    model.load_state_dict(torch.load("model.pt"))
    model.to(device=device)
    model.eval()

    # generation for unconditional diffusion
    x_curr = torch.randn(1, in_c, in_h, in_w).to(device=device)

    for t in range(T, 1, -1):
        t_in = torch.tensor([[t]], dtype=torch.int32, device=device)
        x_curr = sample_next_step(x_curr, t, model)

    # TODO need to move x_curr to cpu
    # then do some postprocessing on pixel values
