import torch

from cwa4.losses import BalancedBCEWithLogits, FocalLossWithLogits


def test_focal_reduces_to_balanced_bce_at_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(64)
    target = (torch.rand(64) > 0.5).float()
    alpha = 0.7
    a = BalancedBCEWithLogits(alpha=alpha)(logits, target)
    b = FocalLossWithLogits(alpha=alpha, gamma=0.0)(logits, target)
    torch.testing.assert_close(a, b)
