"""Train a single Method 4 binary classifier.

Defaults match PDF Method 4: Adam(1e-4), batch=128, 5 epochs, focal loss γ=3,
α auto-set to negative ratio in train set, dev = 1/10 of train (random),
keep checkpoint with lowest dev loss.

Usage:
    uv run scripts/m4_train.py --T 730 --tau 365 --input all
    uv run scripts/m4_train.py --T 730 --tau 1   --input gnss --loss focal --gamma 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from cwa4.data import Method4Dataset, load_method4_sources
from cwa4.data.method4 import make_train_dev
from cwa4.losses import BalancedBCEWithLogits, FocalLossWithLogits
from cwa4.models import ClassifierM4

BATCH_SIZE = 128
N_EPOCHS = 5
LR = 1e-4
DEV_SEED = 0


def make_loss(name: str, alpha: float, gamma: float) -> torch.nn.Module:
    if name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    if name == "balanced":
        return BalancedBCEWithLogits(alpha=alpha)
    if name == "focal":
        return FocalLossWithLogits(alpha=alpha, gamma=gamma)
    if name == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"unknown loss {name}")


def auto_alpha(dataset) -> float:
    """α = negative ratio (PDF 4.2 description)."""
    pos = dataset.positive_count() if hasattr(dataset, "positive_count") else int(
        sum(int(dataset[i][1].item() > 0.5) for i in range(len(dataset)))
    )
    n = len(dataset)
    neg = n - pos
    return float(neg / max(n, 1))


def train(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    sources = load_method4_sources()

    full_trn = Method4Dataset(sources, T=args.T, tau=args.tau, input_kind=args.input,
                              target_kind=args.target, split="trn")
    tst = Method4Dataset(sources, T=args.T, tau=args.tau, input_kind=args.input,
                         target_kind=args.target, split="tst")
    trn_subset, dev_subset = make_train_dev(full_trn, dev_seed=DEV_SEED)

    alpha = args.alpha if args.alpha is not None else auto_alpha(full_trn)
    loss_fn = make_loss(args.loss, alpha=alpha, gamma=args.gamma)
    print(f"[m4_train] T={args.T} tau={args.tau} input={args.input} target={args.target} "
          f"loss={args.loss} alpha={alpha:.4f} gamma={args.gamma}")
    print(f"[m4_train] |trn|={len(trn_subset)} |dev|={len(dev_subset)} |tst|={len(tst)} "
          f"input_dim={full_trn.input_dim}")

    torch.manual_seed(args.seed)
    trn_dl = DataLoader(trn_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_dl = DataLoader(dev_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ClassifierM4(full_trn.input_dim, h_dim=args.h_dim).to(device)
    optim = Adam(model.parameters(), lr=LR)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.input}_T{args.T}_tau{args.tau}_{args.target}_{args.loss}"
    ckpt_path = out_dir / f"{tag}.pt"
    log = []

    best_dev = float("inf")
    for epoch in range(N_EPOCHS):
        model.train()
        for x, y in trn_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y) if args.loss != "mse" else F.mse_loss(torch.sigmoid(logits), y)
            loss.backward()
            optim.step()
        # dev
        model.eval()
        dev_loss_sum, n = 0.0, 0
        with torch.inference_mode():
            for x, y in dev_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                l = loss_fn(logits, y) if args.loss != "mse" else F.mse_loss(torch.sigmoid(logits), y)
                dev_loss_sum += float(l.item()) * y.shape[0]
                n += y.shape[0]
        dev_loss = dev_loss_sum / max(n, 1)
        log.append({"epoch": epoch, "dev_loss": dev_loss})
        print(f"  epoch {epoch} dev_loss={dev_loss:.4f}")
        if dev_loss < best_dev:
            best_dev = dev_loss
            torch.save({
                "state_dict": model.state_dict(),
                "args": vars(args),
                "alpha": alpha,
                "input_dim": full_trn.input_dim,
                "best_dev_loss": best_dev,
                "epoch": epoch,
            }, ckpt_path)

    (out_dir / f"{tag}.log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[m4_train] best_dev={best_dev:.4f}, ckpt -> {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=730)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--input", choices=("gnss", "stats", "all"), default="all")
    parser.add_argument("--target", choices=("binary", "count", "logE"), default="binary")
    parser.add_argument("--loss", choices=("focal", "bce", "balanced", "mse"), default="focal")
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=None,
                        help="weight for positive class; default = negative ratio")
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="exp/m4")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
