"""
Unet mode for diffusion
Closely following the DDPM paper: https://github.com/hojonathanho/diffusion/tree/master

Except that we use more than one head for the attention block
"""
import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


EMBEDDING_DIM = 256
DROPOUT_RATE = 0.1


class Resnet(nn.Module):
    """Resnet block"""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        name: str,
        n_group_norm: int = 4,
        embd_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()
        self.name = name
        assert (
            c_out // n_group_norm > 1 and c_in // n_group_norm > 1
        ), f"c_out = {c_out}, c_in = {c_in}, n_group_norm = {n_group_norm},  need c // n_group_norm > 1"
        self.pre_timestep_block = nn.Sequential(
            nn.GroupNorm(n_group_norm, c_in),
            nn.SiLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(c_in, c_out, 3, padding=1),
        )
        self.timestep_block = nn.Sequential(
            nn.GroupNorm(n_group_norm, embd_dim),
            nn.Linear(embd_dim, c_out),
        )
        self.post_timestep_block = nn.Sequential(
            nn.GroupNorm(n_group_norm, c_out),
            nn.SiLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Conv2d(c_out, c_out, 3, padding=1),
        )
        if c_in != c_out:
            self.identity_conv = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        h: torch.Tensor = self.pre_timestep_block(x)
        t_emb = self.timestep_block(t_emb)
        h = h + t_emb[:, :, None, None]
        h = self.post_timestep_block(h)
        if hasattr(self, "identity_conv"):
            x = self.identity_conv(x)
        print(f"Resnet {self.name} h={h.size()}, x={x.size()}, t={t_emb.size()}")
        return x + h


class Downsample(nn.Module):
    """Unet downsampling block"""

    def __init__(self, c_in: int, c_out: int, name: str):
        super().__init__()
        self.name = name
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        assert out.size(2) == x.size(2) // 2
        assert out.size(3) == x.size(3) // 2
        print(f"Downsample {self.name} x={x.size()}, out={out.size()}")
        return out


class Upsample(nn.Module):
    """Unet upsampling block, use F.interpolate to upsample"""

    def __init__(self, c_in: int, c_out: int, name: str):
        super().__init__()
        self.name = name
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        assert out.size(2) == x.size(2) * 2
        assert out.size(3) == x.size(3) * 2
        print(f"Upsample {self.name} x={x.size()}, out={out.size()}")
        return out


def get_embedding(
    num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
):
    """Build sinusoidal embeddings.

    Copied from fairseq

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(
        0
    )
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


class Unet(nn.Module):
    """Unet
    Also keeps a trainable timestep embedding
    """

    def __init__(
        self,
        c_start: int,
        c_last: int = 64,
        c_between: List[int] = [16, 32, 64, 128],
        n_group_norm: int = 4,
        embd_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()
        self.name = "Unet"

        self.register_buffer(
            "timestep_embedding", get_embedding(1000, embd_dim), persistent=False
        )

        self.learnable_embedding_block = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.SiLU(),
            nn.Linear(embd_dim, embd_dim),
        )

        self.initial_conv = nn.Conv2d(c_start, c_between[0], 3, padding=1)

        self.downblocks = nn.Sequential()
        for i, c_in in enumerate(c_between[:-1]):
            c_out = c_between[i + 1]
            self.downblocks.add_module(
                f"res_{i}",
                Resnet(
                    c_in,
                    c_out,
                    name=f"res_{i}",
                    n_group_norm=n_group_norm,
                    embd_dim=embd_dim,
                ),
            )
            self.downblocks.add_module(
                f"down_{i}", Downsample(c_out, c_out, name=f"down_{i}")
            )
        self.downblocks.add_module(
            "res_middle_1",
            Resnet(
                c_between[-1],
                c_between[-1],
                name="res_middle_1",
                n_group_norm=n_group_norm,
                embd_dim=embd_dim,
            ),
        )

        self.attn = nn.MultiheadAttention(c_between[-1], 4, dropout=DROPOUT_RATE)

        self.upblocks = nn.Sequential()
        self.upblocks.add_module(
            f"res_middle_2",
            Resnet(
                c_between[-1],
                c_between[-1],
                name="res_middle_2",
                n_group_norm=n_group_norm,
                embd_dim=embd_dim,
            ),
        )
        reversed_c = list(reversed(c_between))
        for i, c_in in enumerate(reversed_c[:-1]):
            c_out = reversed_c[i + 1]
            self.upblocks.add_module(
                f"res_{i}",
                Resnet(
                    c_in,
                    c_out,
                    name=f"res_{i}",
                    n_group_norm=n_group_norm,
                    embd_dim=embd_dim,
                ),
            )
            self.upblocks.add_module(f"up_{i}", Upsample(c_in, c_out, name=f"up_{i}"))

        self.output_conv = nn.Conv2d(c_last, c_start, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # get timestep embeddings
        t_emb: torch.Tensor = self.timestep_embedding
        t_emb = self.learnable_embedding_block(t_emb)
        t_emb = t_emb.index_select(0, t.view(-1)).view(t.size(0), t.size(1), -1)

        x = self.downblocks(x, t_emb)
        h = self.attn(h, h, h)[0]
        x = x + h
        x = self.upblocks(x, t_emb)
        x = self.output_conv(x)
        return x
