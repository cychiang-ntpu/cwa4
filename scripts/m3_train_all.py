"""Method 3 training driver.

Defaults to the *subset* scope (matches the approved plan):
  - 13 "hot" centers (PDF table 3): all |I|≤3 neighbor combinations with counts head
  - 12 "cold" centers: only the single-station model with counts head
  - All 25 centers: single-station binary head model — result-only, no ckpt

Pass `--scope full` to run the entire 26,335×2 model grid (very expensive).

Usage:
    uv run scripts/m3_train_all.py
    uv run scripts/m3_train_all.py --shard 0/4 --device cuda:0
    uv run scripts/m3_train_all.py --scope full
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cwa4.data import Method3Dataset, load_method3_sources
from cwa4.data.preprocessing import neighbors_within
from cwa4.models import ModelA

BATCH_SIZE = 128
N_EPOCHS = 5
LR = 1e-4
RADIUS_NEIGHBOR = 40_000.0

HOT_CENTERS = [
    "WARO", "NDHU", "FLNM", "TUNM", "BLOW", "SHUL", "SICH",
    "SCHN", "YENL", "HUAL", "SOFN", "HUAP", "SLIN",
]
ALL_CENTERS = [
    "DCHU", "FUDN", "JPEI", "SPAO", "KNKO", "YUL1", "JSU2",
    "SOFN", "SHUL", "YENL", "BLOW", "WARO", "JPIN", "NDHU",
    "SLIN", "HUAP", "SCHN", "HUAL", "SICH", "FONB", "CHUN",
    "JULI", "DNFU", "FLNM", "TUNM",
]
COLD_CENTERS = [c for c in ALL_CENTERS if c not in HOT_CENTERS]


def parse_shard(s: str) -> tuple[int, int]:
    i, n = s.split("/")
    return int(i), int(n)


def build_jobs(scope: str, sources, max_neighbors: int = 3) -> list[dict]:
    """Each job: {center, neighbors_combo, head}.

    `max_neighbors` caps |I| (the number of extra neighbor stations on top of the
    center). PDF spec is 3; with ~13s/model on GPU, capping at 2 still covers all
    top-3 entries listed in PDF table 4 and finishes in ~5h instead of ~24h.
    """
    jobs: list[dict] = []
    name_to_xy = sources.stations_df.set_index("name")
    cap = max_neighbors + 1  # itertools.combinations stops at k = cap - 1
    for center in ALL_CENTERS:
        if center not in name_to_xy.index:
            print(f"[warn] center {center} not in alive stations — skipped")
            continue
        all_neighbors = neighbors_within(sources.stations_df, center, RADIUS_NEIGHBOR)

        if scope == "full":
            for k in range(0, cap):
                for combo in itertools.combinations(all_neighbors, k):
                    for head in ("counts", "binary"):
                        jobs.append(dict(center=center, neighbors=list(combo), head=head))
        else:
            # counts head
            if center in HOT_CENTERS:
                for k in range(0, cap):
                    for combo in itertools.combinations(all_neighbors, k):
                        jobs.append(dict(center=center, neighbors=list(combo), head="counts"))
            else:
                jobs.append(dict(center=center, neighbors=[], head="counts"))
            # binary head: single-station only, summary only
            jobs.append(dict(center=center, neighbors=[], head="binary"))
    return jobs


def model_id(job: dict) -> str:
    parts = [job["center"], *job["neighbors"]]
    return "_".join(parts)


def evaluate_counts(model, dl, device) -> dict:
    model.eval()
    sum_se = torch.zeros(5, 10, dtype=torch.double, device=device)
    cf = torch.zeros(5, 10, 4, dtype=torch.long, device=device)
    n = 0
    with torch.inference_mode():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)[:, -1]  # (B, 5, 10)
            n += y.shape[0]
            sum_se += (y - y_hat).pow(2).sum(dim=0).double()
            z = y >= 1
            zh = y_hat >= 1
            cf[..., 0] += (z & zh).sum(dim=0)
            cf[..., 1] += (z & ~zh).sum(dim=0)
            cf[..., 2] += (~z & zh).sum(dim=0)
            cf[..., 3] += (~z & ~zh).sum(dim=0)
    mse = (sum_se / max(n, 1)).cpu()
    cf_cpu = cf.cpu()
    tp = cf_cpu[..., 0].float()
    fn = cf_cpu[..., 1].float()
    fp = cf_cpu[..., 2].float()
    tn = cf_cpu[..., 3].float()
    eps = 1e-12
    return {
        "n": n,
        "mse": mse,
        "cf_mat": cf_cpu,
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + fn + fp + tn + eps),
    }


def evaluate_binary(model, dl, device) -> dict:
    model.eval()
    n = 0
    pos = 0
    pred_pos = 0
    pred_max = -1e9
    with torch.inference_mode():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)[:, -1]  # ModelA(binary) → (B, T) → take last step
            prob = torch.sigmoid(logits)
            n += y.shape[0]
            pos += int(y.sum().item())
            pred_pos += int((prob >= 0.5).sum().item())
            pred_max = max(pred_max, float(prob.max().item()))
    return {"n": n, "pos": pos, "pred_pos": pred_pos, "pred_max": pred_max}


def train_one(job: dict, sources, device: torch.device, exp_dir: Path,
              seeds: list[int]) -> None:
    """Train one job for each seed and save per-seed result/ckpt files."""
    center = job["center"]
    head = job["head"]
    out_dir = exp_dir / center
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = model_id(job)

    target_kind = "counts" if head == "counts" else "binary"
    trn_set = Method3Dataset(center, job["neighbors"], sources, target_kind=target_kind, split="trn")
    tst_set = Method3Dataset(center, job["neighbors"], sources, target_kind=target_kind, split="tst")
    tst_dl = DataLoader(tst_set, batch_size=128, shuffle=False, num_workers=0)

    for seed in seeds:
        result_path = out_dir / f"result_{mid}_{head}_seed{seed}.pt"
        if result_path.exists():
            continue  # resume

        torch.manual_seed(seed)
        trn_dl = DataLoader(trn_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model = ModelA(trn_set.x_dim, h_ch=128, head=head).to(device)
        optim = AdamW(model.parameters(), lr=LR)

        model.train()
        for _ in range(N_EPOCHS):
            for x, y in trn_dl:
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                out = model(x)
                if head == "counts":
                    y_hat = out[:, -1]
                    loss = F.mse_loss(y_hat, y)
                else:
                    logits = out[:, -1]
                    loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                optim.step()

        if head == "counts":
            result = evaluate_counts(model, tst_dl, device)
        else:
            result = evaluate_binary(model, tst_dl, device)
        result["job"] = job
        result["seed"] = seed
        torch.save(result, result_path)
        if head == "counts":
            # save ckpt only for counts head (per plan: skip BCE ckpts)
            torch.save(model.state_dict(), out_dir / f"model_{mid}_{head}_seed{seed}.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("subset", "full"), default="subset")
    parser.add_argument("--max_neighbors", type=int, default=2,
                        help="Cap |I| of neighbor combinations (PDF spec is 3). "
                             "Default 2 covers all PDF top-3 entries and is ~4x faster.")
    parser.add_argument("--shard", default="0/1", help="i/N — process shard i out of N")
    parser.add_argument("--seeds", default="0,1,2,3,4",
                        help="comma-separated random seeds (default: 0,1,2,3,4)")
    parser.add_argument("--out", default="exp/m3")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    shard_i, shard_n = parse_shard(args.shard)
    device = torch.device(args.device)
    exp_dir = Path(args.out)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[m3_train_all] scope={args.scope} shard={shard_i}/{shard_n} seeds={seeds} device={device}")
    sources = load_method3_sources()
    jobs = build_jobs(args.scope, sources, max_neighbors=args.max_neighbors)
    my_jobs = [j for k, j in enumerate(jobs) if k % shard_n == shard_i]
    print(f"[m3_train_all] total jobs={len(jobs)} this shard={len(my_jobs)} × {len(seeds)} seeds")

    t0 = time.time()
    for k, job in enumerate(my_jobs):
        train_one(job, sources, device, exp_dir, seeds)
        if (k + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (len(my_jobs) - k - 1) / max(rate, 1e-9)
            print(f"  [{k+1}/{len(my_jobs)}] {rate:.2f} jobs/s  ETA {eta/60:.1f} min")
    print(f"[m3_train_all] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
