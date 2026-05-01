"""Loss functions for highly imbalanced binary classification.

All losses operate on raw logits (not sigmoid-ed probabilities) for numerical
stability — they internally use `F.binary_cross_entropy_with_logits`.

References:
  - PDF 4.2 eq. (4.5): balanced BCE with α weighting
  - PDF 4.2 eq. (4.6): focal loss (Lin et al. 2017)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedBCEWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        # weight pos by alpha, neg by 1-alpha
        weight = torch.where(target > 0.5, self.alpha, 1.0 - self.alpha)
        return (bce * weight).mean()


class FocalLossWithLogits(nn.Module):
    """Focal loss in PDF eq. (4.6).

    γ=0 reduces to balanced BCE. The report uses γ=3 for Method 4.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be ≥0, got {gamma}")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # p_t = p if y=1 else 1-p
        prob = torch.sigmoid(logits)
        p_t = torch.where(target > 0.5, prob, 1.0 - prob)
        alpha_t = torch.where(target > 0.5, self.alpha, 1.0 - self.alpha)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()
