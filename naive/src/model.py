"""
Unet mode for diffusion
Closely following the DDPM paper: https://github.com/hojonathanho/diffusion/tree/master

Except that we use more than one head for the attention block
"""
import logging
import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


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
        classifier_free: bool = False,
    ):
        super().__init__()
        self.name = name
        self.c_in = c_in
        self.c_out = c_out
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
        self.classifier_free = classifier_free
        if self.classifier_free:
            self.class_block = nn.Sequential(
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

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        logger.info(
            f"Resnet {self.name} input: x={x.size()}, t={t_emb.size()}, c_in={self.c_in}, c_out={self.c_out}"
        )
        h: torch.Tensor = self.pre_timestep_block(x)
        t_emb = self.timestep_block(t_emb)
        h = h + t_emb[:, :, None, None]
        logger.info(f"t_emb={t_emb.size()}")
        if c_emb is not None:
            c_emb = self.class_block(c_emb)
            h += c_emb[:, :, None, None]
            logger.info(f"Resnet {self.name} c_emb={c_emb.size()}")
        logger.info(f"Resnet {self.name} pre timestep: h={h.size()}, t={t_emb.size()}")
        h = self.post_timestep_block(h)
        if hasattr(self, "identity_conv"):
            x = self.identity_conv(x)
        logger.info(
            f"Resnet {self.name} output: h={h.size()}, x={x.size()}, t={t_emb.size()}"
        )
        return x + h


class Downsample(nn.Module):
    """Unet downsampling block"""

    def __init__(self, c_in: int, c_out: int, name: str):
        super().__init__()
        self.name = name
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        assert out.size(2) == x.size(2) // 2
        assert out.size(3) == x.size(3) // 2
        logger.info(f"Downsample {self.name} x={x.size()}, out={out.size()}")
        return out


class Upsample(nn.Module):
    """Unet upsampling block, use F.interpolate to upsample"""

    def __init__(self, c_in: int, c_out: int, name: str):
        super().__init__()
        self.name = name
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out: torch.Tensor = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        assert out.size(2) == x.size(2) * 2
        assert out.size(3) == x.size(3) * 2
        logger.info(f"Upsample {self.name} x={x.size()}, out={out.size()}")
        return out


def get_embedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: Optional[int] = None,
    null_token: bool = False,
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
    if null_token:
        # add another row of zeros
        emb = torch.cat([emb, torch.zeros(1, embedding_dim)], dim=0)
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
        classifier_free: bool = False,
        n_classes: int = 0,
    ):
        super().__init__()
        self.name = "Unet"
        self.classifier_free = classifier_free

        self.register_buffer(
            "timestep_embedding", get_embedding(1000, embd_dim), persistent=False
        )
        if classifier_free:
            self.register_buffer(
                "classifier_embedding",
                get_embedding(n_classes, embd_dim, null_token=True),
                persistent=False,
            )
            logger.info(f"Classifier embedding: {self.classifier_embedding.size()}")

        self.learnable_embedding_block = nn.Sequential(
            nn.Linear(embd_dim, embd_dim),
            nn.SiLU(),
            nn.Linear(embd_dim, embd_dim),
        )

        self.initial_conv = nn.Conv2d(c_start, c_between[0], 3, padding=1)

        self.downblocks = nn.ModuleList()
        for i, c_in in enumerate(c_between[:-1]):
            c_out = c_between[i + 1]
            self.downblocks.append(
                Resnet(
                    c_in,
                    c_out,
                    name=f"res_down_{i}",
                    n_group_norm=n_group_norm,
                    embd_dim=embd_dim,
                    classifier_free=self.classifier_free,
                )
            )
            self.downblocks.append(Downsample(c_out, c_out, name=f"down_{i}"))
        self.downblocks.append(
            Resnet(
                c_between[-1],
                c_between[-1],
                name="res_middle_1",
                n_group_norm=n_group_norm,
                embd_dim=embd_dim,
                classifier_free=self.classifier_free,
            )
        )

        self.attn = nn.MultiheadAttention(c_between[-1], 4, dropout=DROPOUT_RATE)

        self.upblocks = nn.ModuleList()
        logger.info(f"first upblock: {c_between[-1]} -> {c_between[-1]}")
        self.upblocks.append(
            Resnet(
                c_between[-1],
                c_between[-1],
                name="res_middle_2",
                n_group_norm=n_group_norm,
                embd_dim=embd_dim,
                classifier_free=self.classifier_free,
            )
        )
        reversed_c = list(reversed(c_between))
        for i, c_in in enumerate(reversed_c[:-1]):
            c_out = reversed_c[i + 1]
            logger.info(f"upblock {i}: {c_in} -> {c_out}")
            self.upblocks.append(
                Resnet(
                    c_in,
                    c_in,
                    name=f"res_up_{i}",
                    n_group_norm=n_group_norm,
                    embd_dim=embd_dim,
                    classifier_free=self.classifier_free,
                )
            )
            self.upblocks.append(Upsample(c_in, c_out, name=f"up_{i}"))

        self.output_res = Resnet(
            c_between[0],
            c_last,
            name="res_output",
            n_group_norm=n_group_norm,
            embd_dim=embd_dim,
            classifier_free=self.classifier_free,
        )
        self.output_conv = nn.Conv2d(c_last, c_start, 3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # get timestep embeddings
        t_emb: torch.Tensor = self.timestep_embedding
        c_emb: Optional[torch.Tensor] = None
        if self.classifier_free:
            assert c is not None
            c_emb: torch.Tensor = self.classifier_embedding
            c_emb = c_emb.index_select(0, c.view(-1)).view(c.size(0), -1)
            logger.info(f"c_emb={c_emb.size()}")

        # the next step is copying the DDPM implementation, which passes the timestep embeddings through some dense layers
        # But this is actually quite weird, because if we're passing time indices through a randomly initialized dense layer anyway,
        # then why not just use trainable embeddings, like nn.Embedding?
        t_emb = self.learnable_embedding_block(t_emb)
        t_emb = t_emb.index_select(0, t.view(-1)).view(t.size(0), -1)

        x = self.initial_conv(x)
        for module in self.downblocks:
            x = module(x, t_emb, c_emb)
        logger.info(f"x={x.size()}, t_emb={t_emb.size()}")
        x_embedded = x.view(x.size(0), x.size(1), -1)  # (B, C, H*W)
        x_embedded = x_embedded.transpose(1, 2)  # (B, H*W, C)
        logger.info(f"x_embedded={x_embedded.size()}")
        h = self.attn(x_embedded, x_embedded, x_embedded)[0]
        h = h.transpose(1, 2).contiguous().view(x.size())
        logger.info(f"h={h.size()}")
        x = x + h
        for module in self.upblocks:
            x = module(x, t_emb, c_emb)
        x = self.output_res(x, t_emb, c_emb)
        x = self.output_conv(x)
        return x
