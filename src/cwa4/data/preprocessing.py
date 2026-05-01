"""Shared helpers for Method 3 / Method 4 dataset construction.

Coordinates are in EPSG:3826 (TWD97 / TM2 zone 121). All time-of-day decisions
are aligned to Asia/Taipei dates. The canonical daily index spans
2000-01-01 .. 2023-12-31 (8766 days), which matches the report's 民國 89-112
window for Method 4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
TZ = "Asia/Taipei"
FULL_START = pd.Timestamp("2000-01-01", tz=TZ)
FULL_END = pd.Timestamp("2023-12-31", tz=TZ)
FULL_DATES = pd.date_range(FULL_START, FULL_END, freq="D")  # 8766 days


def load_station_locations(path: Path | str = DATA_DIR / "station_locations.pkl") -> pd.DataFrame:
    return pd.read_pickle(path)


def load_pfile(path: Path | str = DATA_DIR / "Pfile.pkl") -> pd.DataFrame:
    df = pd.read_pickle(path)
    df.index = df.index.tz_convert(TZ)
    return df


def load_gnss_xyu(path: Path | str = DATA_DIR / "GNSS_XYU.pkl") -> pd.DataFrame:
    df = pd.read_pickle(path)
    df.index = df.index.tz_convert(TZ).normalize()
    df = df.reindex(FULL_DATES, fill_value=np.nan)
    return df


def alive_stations(stations_df: pd.DataFrame, cutoff: pd.Timestamp = pd.Timestamp("2023-12-31")) -> pd.DataFrame:
    """Return stations whose `last_epoch` is at or after `cutoff`.

    `last_epoch` is stored as `datetime` (see scripts/1_convert_gnss_format.py),
    so we compare with a tz-naive Timestamp.
    """
    last = pd.to_datetime(stations_df["last_epoch"])
    return stations_df[last >= cutoff].reset_index(drop=True)


def neighbors_within(stations_df: pd.DataFrame, center_name: str, radius: float) -> list[str]:
    row = stations_df.loc[stations_df["name"] == center_name].iloc[0]
    dx = stations_df["X"] - row["X"]
    dy = stations_df["Y"] - row["Y"]
    distance = np.hypot(dx, dy)
    mask = (stations_df["name"] != center_name) & (distance <= radius)
    return stations_df.loc[mask, "name"].tolist()


def events_within(pfile_df: pd.DataFrame, x: float, y: float, radius: float) -> pd.DataFrame:
    distance = np.hypot(pfile_df["X"] - x, pfile_df["Y"] - y)
    return pfile_df[distance <= radius]


def daily_depth_magnitude_hist(
    pfile_df: pd.DataFrame,
    dates: pd.DatetimeIndex = FULL_DATES,
) -> np.ndarray:
    """Bin events into a (T, 5 depths, 10 magnitudes) integer count array.

    Depth bins: [<5, <10, <30, <70, <300] (last is open-ended for the report).
    Magnitude bins: int(magnitude) clipped into [0, 9].
    """
    hist = np.zeros((len(dates), 5, 10), dtype=np.int32)
    if len(pfile_df) == 0:
        return hist
    df = pfile_df.copy()
    df.index = df.index.tz_convert(TZ).normalize()
    df = df[(df.index >= dates[0]) & (df.index <= dates[-1])]

    day_idx = (df.index - dates[0]).days.to_numpy()
    depth = df["深度"].to_numpy()
    mag = df["規模"].to_numpy()

    d_bin = np.full(len(df), 4, dtype=np.int32)
    d_bin[depth < 70] = 3
    d_bin[depth < 30] = 2
    d_bin[depth < 10] = 1
    d_bin[depth < 5] = 0
    m_bin = np.clip(mag.astype(np.int32), 0, 9)

    np.add.at(hist, (day_idx, d_bin, m_bin), 1)
    return hist


def split_train_test_indices(
    n_days: int,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    full_start: pd.Timestamp = FULL_START,
) -> tuple[slice, slice]:
    """Return slices into the canonical daily index for trn / tst splits.

    `train_end` and `test_end` are inclusive last days for each split.
    """
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize(TZ)
    if test_end.tzinfo is None:
        test_end = test_end.tz_localize(TZ)
    train_last = (train_end - full_start).days  # inclusive
    test_last = (test_end - full_start).days
    if train_last >= n_days or test_last >= n_days:
        raise ValueError("split end exceeds available days")
    return slice(0, train_last + 1), slice(train_last + 1, test_last + 1)
