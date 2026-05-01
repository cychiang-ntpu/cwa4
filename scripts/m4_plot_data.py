"""Reproduce PDF figures 3-5 (Method 4 data overview).

  fig3: 25 alive Hualien stations + neighborhood map (≈ PDF 圖二)
  fig4: target events spatial density + station/major-event overlay (≈ PDF 圖三)
  fig5: target events time series with train/test bands (≈ PDF 圖五)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from cwa4.data import load_method3_sources
from cwa4.data.method4 import TRAIN_END, TEST_END
from cwa4.data.preprocessing import alive_stations, events_within

REPORTS = Path("reports")


def to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return lon, lat


def hualien_alive(stations_df: pd.DataFrame, alive_json: Path) -> pd.DataFrame:
    """Intersection of alive stations and Hualien-county station list."""
    names = json.loads(alive_json.read_text(encoding="utf-8"))
    hualien = set(names.get("花蓮縣", []))
    return stations_df[stations_df["name"].isin(hualien)].reset_index(drop=True)


def fig3(stations_hualien: pd.DataFrame, fig_dir: Path) -> None:
    lon, lat = to_lonlat(stations_hualien["X"].to_numpy(), stations_hualien["Y"].to_numpy())
    plt.figure(figsize=(7, 8))
    plt.scatter(lon, lat, marker="*", color="gold", edgecolor="black",
                s=70, zorder=3, label="alive station")
    for n, lo, la in zip(stations_hualien["name"], lon, lat):
        plt.text(lo + 0.005, la + 0.005, n, fontsize=6)
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Alive Hualien GNSS stations (Method 4)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "m4_fig3_stations.png", dpi=150)
    plt.close()


def fig4(sources, stations_hualien: pd.DataFrame, fig_dir: Path) -> None:
    # All events within 20 km of any Hualien station
    pfile = sources.pfile_df
    mask = np.zeros(len(pfile), dtype=bool)
    for _, row in stations_hualien.iterrows():
        d = np.hypot(pfile["X"].to_numpy() - row["X"], pfile["Y"].to_numpy() - row["Y"])
        mask |= d <= 20_000.0
    events = pfile[mask]
    big = events[(events["規模"] >= 5.5) & (events["深度"] < 30.0)]

    e_lon, e_lat = to_lonlat(events["X"].to_numpy(), events["Y"].to_numpy())
    b_lon, b_lat = to_lonlat(big["X"].to_numpy(), big["Y"].to_numpy())
    s_lon, s_lat = to_lonlat(stations_hualien["X"].to_numpy(), stations_hualien["Y"].to_numpy())

    plt.figure(figsize=(7, 9))
    plt.scatter(e_lon, e_lat, color="crimson", s=2, alpha=0.05, label="all events")
    plt.scatter(b_lon, b_lat, color="magenta", marker="x", s=30,
                label="M≥5.5, depth<30 km")
    plt.scatter(s_lon, s_lat, marker="*", color="gold", edgecolor="black",
                s=70, zorder=3, label="station")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Hualien target-event distribution (within 20 km of any alive station)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "m4_fig4_target_map.png", dpi=150)
    plt.close()


def fig5(target_pickle: Path, fig_dir: Path) -> None:
    target = pd.read_pickle(target_pickle)
    idx = pd.to_datetime(target.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    plt.figure(figsize=(10, 4))
    plt.bar(idx, target.values, width=2.0, color="steelblue", label="next 1 day")
    train_end = pd.Timestamp(TRAIN_END).tz_localize(None) if pd.Timestamp(TRAIN_END).tzinfo else pd.Timestamp(TRAIN_END)
    test_end = pd.Timestamp(TEST_END).tz_localize(None) if pd.Timestamp(TEST_END).tzinfo else pd.Timestamp(TEST_END)
    plt.axvspan(idx.min(), train_end, color="lightblue", alpha=0.3, label="train")
    plt.axvspan(train_end, test_end, color="lightgreen", alpha=0.3, label="test")
    plt.ylabel("M≥5.5 depth<30 events / day")
    plt.title("Method 4 target time series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "m4_fig5_target_time.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--alive_json", default="data/各縣市存活測站.json")
    parser.add_argument("--target", default="data/hualian_target_cnt.pkl")
    args = parser.parse_args()
    reports_dir = Path(args.reports)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sources = load_method3_sources()
    stations_hualien = hualien_alive(sources.stations_df, Path(args.alive_json))
    fig3(stations_hualien, fig_dir)
    fig4(sources, stations_hualien, fig_dir)
    fig5(Path(args.target), fig_dir)
    print(f"[m4_plot_data] figures -> {fig_dir}/")


if __name__ == "__main__":
    main()
