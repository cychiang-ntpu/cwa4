"""Method 4 binary classifier (logits output).

Same architecture as the original `Classifier1` in `cwa4/encoder.py`
(GRUSwishNorm × 3 + Linear), but the final sigmoid is removed so the model
emits logits and pairs with `BCEWithLogitsLoss` / focal-loss (numerically
stable, avoids log(0) when paired with raw probabilities).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..encoder import GRUSwishNorm


class ClassifierM4(nn.Module):
    def __init__(self, x_dim: int, h_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            GRUSwishNorm(x_dim, h_dim),
            GRUSwishNorm(h_dim, h_dim),
            GRUSwishNorm(h_dim, h_dim),
        )
        self.output = nn.Linear(h_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, x_dim) → logits (B,) for the final timestep."""
        h = self.encoder(x)
        return self.output(h[:, -1, :]).squeeze(-1)
