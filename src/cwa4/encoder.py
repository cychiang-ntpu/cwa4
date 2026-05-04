"""Reusable building blocks for the Method 3/4 networks.

Only the symbols still referenced by `cwa4.models.*` are kept here:
  - Swish: SiLU with a learnable β scale
  - GRUSwishNorm: MinGRU → Swish → LayerNorm block (used by ClassifierM4)

Older helpers (Classifier1, Conv1dBTC, ConvSwishNorm, sequence-cut utilities)
were removed when the legacy training scripts were retired in favour of
`m3_train_all.py` / `m4_train.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mingru import MinGRU
from .layernorm import LayerNorm1d


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.beta * x)


class GRUSwishNorm(nn.Module):
    def __init__(self, x_dim: int, y_dim: int):
        super().__init__()
        self.gru_fwd = MinGRU(x_dim, y_dim)
        self.swish = Swish()
        self.norm = LayerNorm1d(y_dim, channels_last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru_fwd(x)
        y = self.swish(y)
        y = self.norm(y)
        return y
