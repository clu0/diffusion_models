"""
Code to train a simple diffusion model, i.e. the epsilon_theta(x, t) model

TODO: substitute the model for an actual UNet
"""
from typing import Tuple, Optional
from time import time
import argparse
import random
import sys
import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.model import Unet
from src.logger import Logger, HumanOutputFormat, CSVOutputFormat


T: int = 1000
beta_1: float = 1e-4
beta_T: float = 0.02
# following 3 for mnist
in_c: int = 1
in_h: int = 32
in_w: int = 32


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def get_beta_t(t: int) -> float:
    assert 1 <= t and t <= T, f"time index {t} must be in range [1, {T}]"
    return (beta_1 * (T - t) + beta_T * (t - 1)) / (T - 1)


betas: torch.Tensor = torch.Tensor([get_beta_t(t) for t in range(1, T + 1)])
alphas = torch.cumprod(1 - betas, dim=0)
alphas = alphas.to(device=device)


class DiffusionTrainingDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        classes: npt.NDArray[np.int_],
        n_classes: int,
        p_null: Optional[float] = None,
    ):
        # assume that the input data has already been preprocessed into shape (N, in_c, in_h, in_w)
        assert data.ndim == 4 and data.shape[1:] == (in_c, in_h, in_w)
        # using mnist right now, can just load the whole dataset to cuda
        self.data = data.to(device=device)
        self.classes = classes
        self.n_classes = n_classes
        self.p_null = p_null

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], int]:
        """
        return the i-th data, along with a random normal of same shape, and a random time step in {1, ..., T}
        """
        if self.p_null is not None and random.random() < self.p_null:
            # with some probability, train with null class token
            t = torch.tensor([self.n_classes], device=device, dtype=torch.long)
        else:
            t = torch.randint(0, T, (1,), device=device, dtype=torch.long)
        return (
            self.data[idx, ...],
            torch.randn((in_c, in_h, in_w), device=device, dtype=torch.float32),
            t,
            torch.tensor([self.classes[idx]], device=device, dtype=torch.long),
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=6*1e-4, help="learning rate")
    parser.add_argument("--p_null", type=float, default=0.1, help="probability of null class token")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="path to past model")
    parser.add_argument("--past_n_epoch", type=int, default=None, help="epoch of past model")
    parser.add_argument("--data_tensor_path", type=str, default="mnist_upsampled.pt", help="path to data tensor")
    parser.add_argument("--metadata_path", type=str, default="mnist/mnist.csv", help="path to metadata")
    parser.add_argument("--model_save_dir", type=str, default="models/", help="prefix for saving model")
    parser.add_argument("--model_save_prefix", type=str, default="")
    parser.add_argument("--log_save_suffix", type=str, default="")
    parser.add_argument("--log_save_dir", type=str, default="logs/", help="prefix for saving model")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--classifier_free", dest='classifier_free', action='store_true', help="use classifier free training")
    parser.set_defaults(classifier_free=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(f"training args: \n{args}")

    # Train with mnist
    mnist_upsampled: torch.Tensor = torch.load(args.data_tensor_path)
    mnist_upsampled = mnist_upsampled.float()
    mnist_metadata = pd.read_csv(args.metadata_path)
    mnist_classes = mnist_metadata["label"].to_numpy()
    p_null = None
    if args.classifier_free:
        print(f"training with classifier free")
        p_null = args.p_null
    train_dataloader = DataLoader(
        DiffusionTrainingDataset(mnist_upsampled, mnist_classes, n_classes=args.n_classes, p_null=p_null),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = Unet(c_start=in_c, classifier_free=args.classifier_free, n_classes=args.n_classes)
    if args.model_checkpoint is not None:
        model.load_state_dict(torch.load(args.model_checkpoint))
    model.to(device=device)
    print(f"loaded model {args.model_checkpoint}")

    loss_fn = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_losses = []
    epoch_losses = []

    model.train()
    print(f"starting training for {args.n_epoch} epochs")

    # initialize save dir and logger
    log_save_dir = os.path.join(args.log_save_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.mkdir(log_save_dir, exist_ok=True)
    model_save_dir = os.path.join(args.model_save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(model_save_dir, exist_ok=True)
    output_formats = [
        HumanOutputFormat(sys.stdout),
        HumanOutputFormat(os.path.join(log_save_dir, f"log{args.log_save_suffix}.txt")),
        CSVOutputFormat(os.path.join(log_save_dir, f"progress{args.log_save_suffix}.csv"))
    ]
    logger = Logger(output_formats)
    
    for i in range(args.n_epoch):
        epoch_start = time()
        if args.past_n_epoch is not None:
            i += args.past_n_epoch

        logger.logkv("epoch", i)
        for j, batch in enumerate(train_dataloader):
            if (j + 1) % 100 == 0:
                print(f"epoch {i}, batch {j}, loss {np.mean(train_losses[-100:])}")
            data, noise, times, classes = batch
            if not args.classifier_free:
                classes = None
            optimizer.zero_grad()
            alpha_ts = alphas[times]
            alpha_ts = alpha_ts.view(alpha_ts.size(0), alpha_ts.size(1), 1, 1)
            x_ts = (torch.sqrt(alpha_ts) * data + torch.sqrt(1 - alpha_ts) * noise).float()
            pred = model(x_ts, times, classes)
            loss_vals = loss_fn(pred, noise)
            loss_vals.backward()
            optimizer.step()

            train_losses.append(loss_vals.item())
        print(f"finished epoch {i}, took {time() - epoch_start} seconds")
        np.save(f"{args.loss_save_prefix}_{i+1}.npy", np.array(train_losses))
        if (i + 1) % 20 == 0:
            torch.save(model.state_dict(), f"{args.model_save_prefix}_epoch_{i + 1}.pth")

