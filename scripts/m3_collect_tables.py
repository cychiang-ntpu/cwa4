"""Aggregate exp/m3/**/result_*.pt into PDF tables 2-5 + summary.md.

Run after `scripts/m3_train_all.py` finishes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cwa4.data import load_method3_sources
from cwa4.data.method3 import RADIUS_TARGET, RADIUS_NEIGHBOR
from cwa4.data.preprocessing import events_within, neighbors_within

# Same 25-station Hualien-area list used by m3_train_all.py
ALL_CENTERS = [
    "DCHU", "FUDN", "JPEI", "SPAO", "KNKO", "YUL1", "JSU2",
    "SOFN", "SHUL", "YENL", "BLOW", "WARO", "JPIN", "NDHU",
    "SLIN", "HUAP", "SCHN", "HUAL", "SICH", "FONB", "CHUN",
    "JULI", "DNFU", "FLNM", "TUNM",
]

# PDF table 3 only ever flags up to these (depth, magnitude) bins; we report
# them prominently but the raw CSVs include all 50.
HEADLINE_CELLS = [(1, 1), (2, 1), (2, 2)]


def parse_result_filename(p: Path) -> tuple[str, list[str], str]:
    # result_{center}[_{neighbor}...]_{head}.pt
    stem = p.stem  # result_FLNM_FONB_KNKO_counts
    assert stem.startswith("result_")
    parts = stem[len("result_") :].split("_")
    head = parts[-1]
    center = parts[0]
    neighbors = parts[1:-1]
    return center, neighbors, head


def build_table2(sources, out_csv: Path) -> pd.DataFrame:
    rows = []
    centers_df = sources.stations_df[sources.stations_df["name"].isin(ALL_CENTERS)]
    for _, row in centers_df.iterrows():
        name = row["name"]
        x, y = float(row["X"]), float(row["Y"])
        events = events_within(sources.pfile_df, x, y, RADIUS_TARGET)
        m4_shallow = events[(events["規模"] >= 4.0) & (events["深度"] < 30.0)]
        n_neighbors = len(neighbors_within(sources.stations_df, name, RADIUS_NEIGHBOR))
        # number of |I|≤3 combinations (model count for one head)
        from math import comb
        n_models = sum(comb(n_neighbors, k) for k in range(0, 4))
        rows.append(
            dict(center=name, n_neighbors=n_neighbors, n_events=len(events),
                 n_m4_shallow=len(m4_shallow), n_models=n_models)
        )
    df = pd.DataFrame(rows).sort_values("center").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def collect_counts_results(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for p in exp_dir.rglob("result_*.pt"):
        center, neighbors, head = parse_result_filename(p)
        if head != "counts":
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        prec = r["precision"]  # (5, 10) float
        n = r["n"]
        rows.append(dict(
            center=center,
            neighbors=tuple(neighbors),
            n_neighbors=len(neighbors),
            n_test=n,
            precision=prec.numpy(),
            cf_mat=r["cf_mat"].numpy(),
            file=str(p),
        ))
    return pd.DataFrame(rows)


def build_table3(counts_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Single-station (no neighbors) precision in headline cells."""
    single = counts_df[counts_df["n_neighbors"] == 0].copy()
    rows = []
    for _, r in single.iterrows():
        prec = r["precision"]
        record = {"center": r["center"]}
        any_pred = False
        for d, m in HEADLINE_CELLS:
            p = float(prec[d, m])
            record[f"P[{d},{m}]"] = p
            if p > 0:
                any_pred = True
        record["has_prediction"] = any_pred
        rows.append(record)
    df = pd.DataFrame(rows).sort_values("center").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def build_table4(counts_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """For each center, given baseline single-station P at cell [2,1], how many
    combinations raise / lower P, and what are the top-3 combinations?"""
    rows = []
    for center, sub in counts_df.groupby("center"):
        single = sub[sub["n_neighbors"] == 0]
        if len(single) == 0:
            continue
        baseline = float(single.iloc[0]["precision"][2, 1])
        # accuracy at [2,1]
        cf = single.iloc[0]["cf_mat"]
        denom = cf[2, 1].sum()
        baseline_a = float((cf[2, 1, 0] + cf[2, 1, 3]) / max(denom, 1))

        with_neighbors = sub[sub["n_neighbors"] > 0].copy()
        if len(with_neighbors) == 0:
            rows.append(dict(center=center, baseline_p=baseline, baseline_a=baseline_a,
                             improved=0, decreased=0, top1="", top2="", top3=""))
            continue
        with_neighbors["P_2_1"] = with_neighbors["precision"].apply(lambda x: float(x[2, 1]))
        with_neighbors["A_2_1"] = with_neighbors["cf_mat"].apply(
            lambda c: float((c[2, 1, 0] + c[2, 1, 3]) / max(c[2, 1].sum(), 1))
        )
        improved = int((with_neighbors["P_2_1"] > baseline).sum())
        decreased = int((with_neighbors["P_2_1"] < baseline).sum())
        top = with_neighbors.sort_values("P_2_1", ascending=False).head(3)
        names = []
        for _, row in top.iterrows():
            tag = "_".join([center, *row["neighbors"]])
            names.append(f"({row['P_2_1']:.2f}, {row['A_2_1']:.2f}, {tag})")
        while len(names) < 3:
            names.append("")
        rows.append(dict(center=center, baseline_p=baseline, baseline_a=baseline_a,
                         improved=improved, decreased=decreased,
                         top1=names[0], top2=names[1], top3=names[2]))
    df = pd.DataFrame(rows).sort_values("center").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def build_table5(counts_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Models that ever predict M≥4 (m_bin∈{4..9}, depth<30 i.e. d_bin∈{0,1,2})."""
    rows = []
    for _, r in counts_df.iterrows():
        cf = r["cf_mat"]  # (5,10,4)
        # any cell with predicted positives in M≥4 (ignoring depth filter for now,
        # but report the (d,m) where prediction ever fires for M≥4)
        for d in range(5):
            for m in range(4, 10):
                # if there were any predicted positives (TP+FP > 0)
                if int(cf[d, m, 0] + cf[d, m, 2]) > 0:
                    p = float(cf[d, m, 0]) / max(int(cf[d, m, 0] + cf[d, m, 2]), 1)
                    name = "_".join([r["center"], *r["neighbors"]])
                    rows.append(dict(model=name, d=d, m=m, precision=p,
                                     tp=int(cf[d, m, 0]), fp=int(cf[d, m, 2])))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def build_binary_summary(exp_dir: Path) -> dict:
    pred_pos = 0
    n_models = 0
    pred_max_overall = -1.0
    for p in exp_dir.rglob("result_*_binary.pt"):
        r = torch.load(p, map_location="cpu", weights_only=False)
        n_models += 1
        pred_pos += r.get("pred_pos", 0)
        pred_max_overall = max(pred_max_overall, r.get("pred_max", -1.0))
    return dict(n_models=n_models, total_pred_pos=pred_pos, max_prob_observed=pred_max_overall)


def write_summary(reports_dir: Path, table2: pd.DataFrame, table3: pd.DataFrame,
                  table4: pd.DataFrame, table5: pd.DataFrame, bin_summary: dict) -> None:
    has_pred = table3[table3["has_prediction"]]
    md = []
    md.append("# Method 3 — aggregated results\n")
    md.append(f"- Total alive centers: **{len(table2)}**")
    md.append(f"- Centers with at least one non-zero precision in headline cells: **{len(has_pred)}**\n")
    md.append("## Headline finding (matches PDF 4.1.4 / 4.1.5)")
    md.append(f"- Binary (BCE, M≥4 depth<30) models trained: {bin_summary['n_models']}")
    md.append(f"- Sum of test-set positive predictions across all binary models: "
              f"**{bin_summary['total_pred_pos']}**")
    md.append(f"- Max sigmoid output observed in any binary model on the test set: "
              f"{bin_summary['max_prob_observed']:.4f}")
    md.append("\nThe report concludes that no binary model ever predicts a positive on the "
              "test set; the figure above lets you confirm that on this run.\n")
    (reports_dir / "m3_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp/m3")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    exp_dir = Path(args.exp)
    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sources = load_method3_sources()
    table2 = build_table2(sources, reports_dir / "m3_table2_centers.csv")
    counts_df = collect_counts_results(exp_dir)
    if len(counts_df) == 0:
        print(f"[warn] no counts results found in {exp_dir}")
    table3 = build_table3(counts_df, reports_dir / "m3_table3_single_station.csv")
    table4 = build_table4(counts_df, reports_dir / "m3_table4_neighbor_effect.csv")
    table5 = build_table5(counts_df, reports_dir / "m3_table5_m4_models.csv")
    bin_summary = build_binary_summary(exp_dir)
    write_summary(reports_dir, table2, table3, table4, table5, bin_summary)
    print(f"[m3_collect_tables] wrote tables and summary to {reports_dir}/")


if __name__ == "__main__":
    main()
