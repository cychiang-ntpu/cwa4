"""Method 3 backbone — PDF Table 1.

Layer order (input shape (B, 365, x_dim)):
  Linear(x_dim, h)  → LeakyReLU → LayerNorm
  MinGRU(h, h)      → LayerNorm
  MinGRU(h, h)      → LayerNorm
  MinGRU(h, h)      → Linear(h, 50 or 1)
  Reshape           → SoftPlus  (counts head)
                    → identity / logits  (binary head; pair with BCEWithLogitsLoss)

The forward returns the entire 365-step sequence; callers take the final step
(`out[:, -1]`) for next-day prediction (matches `scripts/4_train_model.py`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mingru import MinGRU
from ..layernorm import LayerNorm1d


class _MinGRUBlock(nn.Module):
    def __init__(self, x_dim: int, y_dim: int):
        super().__init__()
        self.gru = MinGRU(x_dim, y_dim)
        self.norm = LayerNorm1d(y_dim, channels_last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.norm(y)


class ModelA(nn.Module):
    def __init__(
        self,
        x_dim: int,
        h_ch: int = 128,
        d_ch: int = 5,
        m_ch: int = 10,
        head: str = "counts",
    ):
        super().__init__()
        if head not in ("counts", "binary"):
            raise ValueError(f"head must be counts/binary, got {head}")
        self.head = head
        self.d_ch = d_ch
        self.m_ch = m_ch

        self.embed = nn.Sequential(
            nn.Linear(x_dim, h_ch),
            nn.LeakyReLU(),
            LayerNorm1d(h_ch, channels_last=True),
        )
        self.gru1 = _MinGRUBlock(h_ch, h_ch)
        self.gru2 = _MinGRUBlock(h_ch, h_ch)
        self.gru3 = _MinGRUBlock(h_ch, h_ch)
        out_dim = d_ch * m_ch if head == "counts" else 1
        self.proj = nn.Linear(h_ch, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, x_dim) → counts: (B, T, d_ch, m_ch); binary: (B, T) logits."""
        h = self.embed(x)
        h = self.gru1(h)
        h = self.gru2(h)
        h = self.gru3(h)
        out = self.proj(h)
        if self.head == "counts":
            B, T, _ = out.shape
            out = out.view(B, T, self.d_ch, self.m_ch)
            return F.softplus(out)
        # binary: return logits directly; pair with BCEWithLogitsLoss
        return out.squeeze(-1)
