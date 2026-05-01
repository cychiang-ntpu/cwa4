"""Method 3 dataset: per-center sliding 365-day windows.

Input: 365 days of (|J| stations × 3 GNSS axes + 5×10 daily event hist)
Output:
  - target_kind="counts": next-day 5×10 integer count tensor (PDF 3.1 def 1)
  - target_kind="binary": next-day {0,1} for "M ≥ 4 and depth < 30 km" (def 2)

The split follows PDF Method 3: 民國 89-110 train / 111-112 test, i.e.
2000-01-01..2021-12-31 train, 2022-01-01..2023-12-31 test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import preprocessing as pp

INPUT_WIDTH = 365
RADIUS_TARGET = 20_000.0  # m
RADIUS_NEIGHBOR = 40_000.0  # m
TRAIN_END = pd.Timestamp("2021-12-31")  # 民國 110
TEST_END = pd.Timestamp("2023-12-31")  # 民國 112


@dataclass
class Method3Sources:
    stations_df: pd.DataFrame  # alive stations
    gnss_df: pd.DataFrame  # canonical daily index, multi-col (station, dX/dY/dU)
    pfile_df: pd.DataFrame  # all events (tz-aware, Asia/Taipei)


def load_method3_sources(data_dir: Path | str = pp.DATA_DIR) -> Method3Sources:
    data_dir = Path(data_dir)
    stations = pp.alive_stations(pp.load_station_locations(data_dir / "station_locations.pkl"))
    gnss = pp.load_gnss_xyu(data_dir / "GNSS_XYU.pkl")
    pfile = pp.load_pfile(data_dir / "Pfile.pkl")
    return Method3Sources(stations_df=stations, gnss_df=gnss, pfile_df=pfile)


class Method3Dataset(Dataset):
    def __init__(
        self,
        center: str,
        neighbors: list[str],
        sources: Method3Sources,
        target_kind: str = "counts",
        split: str = "trn",
        train_end: pd.Timestamp = TRAIN_END,
        test_end: pd.Timestamp = TEST_END,
    ):
        super().__init__()
        if target_kind not in ("counts", "binary"):
            raise ValueError(f"target_kind must be counts/binary, got {target_kind}")
        if split not in ("trn", "tst"):
            raise ValueError(f"split must be trn/tst, got {split}")

        self.center = center
        self.neighbors = list(neighbors)
        self.target_kind = target_kind
        self.split = split

        stations = sources.stations_df
        row = stations.loc[stations["name"] == center].iloc[0]
        cx, cy = float(row["X"]), float(row["Y"])

        # 5×10 daily hist for events within radius_target of the center
        target_events = pp.events_within(sources.pfile_df, cx, cy, RADIUS_TARGET)
        self.hist = pp.daily_depth_magnitude_hist(target_events)  # (T, 5, 10)

        # GNSS: stack center + neighbors columns and fill gaps with 0
        cols: list[tuple[str, str]] = []
        for st in [center, *self.neighbors]:
            for axis in ("dX", "dY", "dU"):
                cols.append((st, axis))
        gnss = sources.gnss_df.reindex(columns=pd.MultiIndex.from_tuples(cols))
        gnss = gnss.fillna(0.0)
        self.gnss = gnss.to_numpy(dtype=np.float32)  # (T, |J|*3)

        # Sliding window indices: each sample i has input days [i:i+INPUT_WIDTH], label day i+INPUT_WIDTH
        n_days = self.hist.shape[0]
        last_label_day = n_days - 1
        all_label_days = np.arange(INPUT_WIDTH, last_label_day + 1)

        full_start = pp.FULL_START
        if train_end.tzinfo is None:
            train_end = train_end.tz_localize(pp.TZ)
        if test_end.tzinfo is None:
            test_end = test_end.tz_localize(pp.TZ)
        train_last = (train_end - full_start).days
        test_last = (test_end - full_start).days

        if split == "trn":
            label_days = all_label_days[all_label_days <= train_last]
        else:
            label_days = all_label_days[(all_label_days > train_last) & (all_label_days <= test_last)]
        self.label_days = label_days

        # Pre-build x dim
        self.x_dim = self.gnss.shape[1] + 50  # |J|*3 + 5*10

    def __len__(self) -> int:
        return int(len(self.label_days))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = int(self.label_days[index])
        gnss = self.gnss[d - INPUT_WIDTH : d]  # (365, |J|*3)
        hist = self.hist[d - INPUT_WIDTH : d]  # (365, 5, 10)
        x = np.concatenate([gnss, hist.reshape(INPUT_WIDTH, -1)], axis=1)
        x = torch.from_numpy(x.astype(np.float32))

        target = self.hist[d]  # (5, 10)
        if self.target_kind == "counts":
            y = torch.from_numpy(target.astype(np.float32))
        else:  # binary: M >= 4 (i.e. m_bin in {4..9}) and depth < 30 (d_bin in {0,1,2})
            y = torch.tensor(float(target[:3, 4:].sum() >= 1))
        return x, y
