"""Method 4 dataset: long-window forecasting for Hualien M ≥ 5.5, depth < 30km.

Inputs (selectable via `input_kind`):
  - "gnss":  daily dX/dY/dU for the 25 alive Hualien stations (75 features)
  - "stats": 6 daily statistics (per-depth count + accumulated log-energy)
  - "all":   gnss + stats concatenated

Targets (selectable via `target_kind`):
  - "binary": {0,1} — does at least one M≥5.5, depth<30 event happen in the next τ days
  - "count":  number of such events in the next τ days (regression)
  - "logE":   log10(total released energy) of such events in the next τ days

Splits:
  - "trn": 民國 89-108 (2000-01-01 .. 2019-12-31), 7305 days
  - "tst": 民國 109-112 (2020-01-01 .. 2023-12-31), 1461 days
  - "dev": 1/10 of trn, randomly sampled (use `make_train_dev`)

The future-window summation pads zeros past the canonical end date, matching
the report's table 6 sample counts (e.g. T=730 → trn 6576, tst 732).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset, Subset

from . import preprocessing as pp

TRAIN_END = pd.Timestamp("2019-12-31")  # 民國 108
TEST_END = pd.Timestamp("2023-12-31")  # 民國 112


@dataclass
class Method4Sources:
    gnss: np.ndarray  # (T, n_features_gnss) float32
    stats: np.ndarray  # (T, 6) float32
    target_cnt: np.ndarray  # (T,) int — count of M≥5.5 depth<30 events that day
    target_energy: np.ndarray  # (T,) float — released energy that day (J)
    dates: pd.DatetimeIndex


def load_method4_sources(data_dir: Path | str = pp.DATA_DIR) -> Method4Sources:
    data_dir = Path(data_dir)
    gnss = pd.read_pickle(data_dir / "hualian_daily_gnss_dXdYdU.pkl")
    stats = pd.read_pickle(data_dir / "hualian_daily_statistics.pkl")
    cnt = pd.read_pickle(data_dir / "hualian_target_cnt.pkl")

    # Align everything to the canonical FULL_DATES
    gnss = gnss.reindex(pp.FULL_DATES).fillna(0.0)
    stats = stats.reindex(pp.FULL_DATES).fillna(0.0)
    cnt = cnt.reindex(pp.FULL_DATES).fillna(0).astype(np.int32)

    # Released energy per day: derived from the statistics' "極淺層-能量" column
    # (which is log10 of summed energy clipped at 1e-2). For the logE target we
    # reconstruct by summing energy across depth bins.
    if {"極淺層-能量", "淺層-能量", "中層-能量"}.issubset(stats.columns):
        e = (
            np.power(10.0, stats["極淺層-能量"].to_numpy() * 1.5 + 11.8)
            + np.power(10.0, stats["淺層-能量"].to_numpy() * 1.5 + 11.8)
            + np.power(10.0, stats["中層-能量"].to_numpy() * 1.5 + 11.8)
        )
    else:
        e = np.zeros(len(pp.FULL_DATES), dtype=np.float64)

    return Method4Sources(
        gnss=gnss.to_numpy(dtype=np.float32),
        stats=stats.to_numpy(dtype=np.float32),
        target_cnt=cnt.to_numpy(),
        target_energy=e.astype(np.float64),
        dates=pp.FULL_DATES,
    )


def _future_window_sum(values: np.ndarray, tau: int) -> np.ndarray:
    """Match the report's tail-padded sliding window: out[i] = sum(values[i+1 : i+1+tau])."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    padded = np.pad(values, (0, tau))
    windows = sliding_window_view(padded, tau).sum(axis=-1)  # length len(values)+1
    return windows[1:]  # drop the [0:tau] window so we look strictly into the future


def _select_features(sources: Method4Sources, sl: slice, input_kind: str) -> np.ndarray:
    if input_kind == "gnss":
        return sources.gnss[sl]
    if input_kind == "stats":
        return sources.stats[sl]
    return np.concatenate([sources.gnss[sl], sources.stats[sl]], axis=1)


def compute_feature_stats(
    sources: Method4Sources,
    input_kind: str,
    train_end: pd.Timestamp = TRAIN_END,
    test_end: pd.Timestamp = TEST_END,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std computed on the train split — share with tst split."""
    n_days = len(sources.dates)
    train_slice, _ = pp.split_train_test_indices(n_days, train_end, test_end)
    feats = _select_features(sources, train_slice, input_kind).astype(np.float64)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class Method4Dataset(Dataset):
    def __init__(
        self,
        sources: Method4Sources,
        T: int = 730,
        tau: int = 1,
        input_kind: str = "all",
        target_kind: str = "binary",
        split: str = "trn",
        train_end: pd.Timestamp = TRAIN_END,
        test_end: pd.Timestamp = TEST_END,
        normalize: bool = False,
        feature_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        super().__init__()
        if input_kind not in ("gnss", "stats", "all"):
            raise ValueError(f"input_kind must be gnss/stats/all, got {input_kind}")
        if target_kind not in ("binary", "count", "logE"):
            raise ValueError(f"target_kind must be binary/count/logE, got {target_kind}")
        if split not in ("trn", "tst"):
            raise ValueError(f"split must be trn/tst, got {split}")

        self.T = int(T)
        self.tau = int(tau)
        self.input_kind = input_kind
        self.target_kind = target_kind
        self.split = split

        n_days = len(sources.dates)
        train_slice, test_slice = pp.split_train_test_indices(n_days, train_end, test_end)
        sl = train_slice if split == "trn" else test_slice

        feats = _select_features(sources, sl, input_kind)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if normalize:
            if feature_stats is None:
                feature_stats = compute_feature_stats(sources, input_kind, train_end, test_end)
            mean, std = feature_stats
            feats = (feats - mean) / std
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.feature_stats = feature_stats
        self.features = feats  # (n_split_days, F)

        # Build label vector aligned with the GLOBAL day index, then slice.
        if target_kind == "count":
            y_global = _future_window_sum(sources.target_cnt.astype(np.float32), self.tau)
        elif target_kind == "binary":
            cnt_window = _future_window_sum(sources.target_cnt.astype(np.float32), self.tau)
            y_global = (cnt_window > 0).astype(np.float32)
        else:  # logE
            e_window = _future_window_sum(sources.target_energy, self.tau)
            y_global = np.log10(np.clip(e_window, 1e-2, None)).astype(np.float32)
        y_split = y_global[sl]

        # After unfolding with window T, sample i (0..len-T) ends at split day i+T-1
        n = len(self.features) - self.T + 1
        if n <= 0:
            raise ValueError(f"split too short for T={T}: only {len(self.features)} days available")
        self.n = int(n)
        self.labels = y_split[self.T - 1 :].astype(np.float32)  # length n
        # Sanity: labels should align with samples
        assert len(self.labels) == self.n, (len(self.labels), self.n)

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # input window: features[index : index+T] shape (T, F)
        x = torch.from_numpy(self.features[index : index + self.T])
        y = torch.tensor(float(self.labels[index]))
        return x, y

    def positive_count(self) -> int:
        """Number of positive labels (only meaningful for binary)."""
        return int((self.labels > 0).sum())


def make_train_dev(dataset: Method4Dataset, dev_seed: int = 0, dev_frac: float = 0.1) -> tuple[Subset, Subset]:
    """Random 1/10 dev split from the training dataset."""
    rng = np.random.default_rng(dev_seed)
    perm = rng.permutation(len(dataset))
    n_dev = int(round(len(dataset) * dev_frac))
    dev_idx = sorted(perm[:n_dev].tolist())
    trn_idx = sorted(perm[n_dev:].tolist())
    return Subset(dataset, trn_idx), Subset(dataset, dev_idx)
