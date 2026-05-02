"""Aggregate exp/m3/**/result_*_seed*.pt into PDF tables 2-5 + summary.md.

Multi-seed friendly: per (center, neighbors, head) combo, the script collects
all seed-level results, computes mean ± std for precision/accuracy at the
relevant cells, and reports per-combo aggregate values in tables 3-5.

Run after `scripts/m3_train_all.py` finishes.
"""
from __future__ import annotations

import argparse
import re
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
SEED_RE = re.compile(r"_seed(\d+)$")
HEADS = ("counts", "binary")


def parse_result_filename(p: Path) -> tuple[str, list[str], str, int]:
    """result_{center}[_{neighbor}...]_{head}_seed{N}.pt → (center, [neighbors], head, seed)."""
    stem = p.stem
    assert stem.startswith("result_")
    body = stem[len("result_"):]
    m = SEED_RE.search(body)
    if m is None:
        raise ValueError(f"missing _seed in {p}")
    seed = int(m.group(1))
    body = body[: m.start()]
    parts = body.split("_")
    head = parts[-1]
    if head not in HEADS:
        raise ValueError(f"unknown head {head} in {p}")
    center = parts[0]
    neighbors = parts[1:-1]
    return center, neighbors, head, seed


def fmt_mean_std(xs: list[float]) -> str:
    if not xs:
        return "nan"
    return f"{np.mean(xs):.3f} ± {np.std(xs):.3f}"


def build_table2(sources, out_csv: Path) -> pd.DataFrame:
    rows = []
    centers_df = sources.stations_df[sources.stations_df["name"].isin(ALL_CENTERS)]
    for _, row in centers_df.iterrows():
        name = row["name"]
        x, y = float(row["X"]), float(row["Y"])
        events = events_within(sources.pfile_df, x, y, RADIUS_TARGET)
        m4_shallow = events[(events["規模"] >= 4.0) & (events["深度"] < 30.0)]
        n_neighbors = len(neighbors_within(sources.stations_df, name, RADIUS_NEIGHBOR))
        from math import comb
        n_models = sum(comb(n_neighbors, k) for k in range(0, 4))
        rows.append(dict(center=name, n_neighbors=n_neighbors, n_events=len(events),
                         n_m4_shallow=len(m4_shallow), n_models=n_models))
    df = pd.DataFrame(rows).sort_values("center").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def collect_counts_results(exp_dir: Path) -> pd.DataFrame:
    """Returns one row per (center, neighbors) combo with stacked per-seed
    precision (n_seeds, 5, 10) and cf_mat (n_seeds, 5, 10, 4) arrays."""
    bucket: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for p in exp_dir.rglob("result_*_seed*.pt"):
        center, neighbors, head, seed = parse_result_filename(p)
        if head != "counts":
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        bucket.setdefault((center, tuple(neighbors)), []).append({
            "seed": seed,
            "precision": r["precision"].numpy(),
            "cf_mat": r["cf_mat"].numpy(),
            "n": r["n"],
        })
    rows = []
    for (center, neighbors), seeds in bucket.items():
        prec_stack = np.stack([s["precision"] for s in seeds])  # (S, 5, 10)
        cf_stack = np.stack([s["cf_mat"] for s in seeds])       # (S, 5, 10, 4)
        rows.append(dict(
            center=center,
            neighbors=neighbors,
            n_neighbors=len(neighbors),
            n_seeds=len(seeds),
            n_test=seeds[0]["n"],
            precision_stack=prec_stack,
            cf_stack=cf_stack,
        ))
    return pd.DataFrame(rows)


def build_table3(counts_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    single = counts_df[counts_df["n_neighbors"] == 0].copy()
    rows = []
    for _, r in single.iterrows():
        prec = r["precision_stack"]  # (S, 5, 10)
        record = {"center": r["center"], "n_seeds": int(r["n_seeds"])}
        any_pred = False
        for d, m in HEADLINE_CELLS:
            seed_p = prec[:, d, m]
            record[f"P[{d},{m}]_mean"] = float(np.mean(seed_p))
            record[f"P[{d},{m}]_std"] = float(np.std(seed_p))
            record[f"P[{d},{m}]"] = fmt_mean_std(seed_p.tolist())
            if np.mean(seed_p) > 0:
                any_pred = True
        record["has_prediction"] = any_pred
        rows.append(record)
    df = pd.DataFrame(rows).sort_values("center").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def _accuracy_at_cell(cf_stack: np.ndarray, d: int, m: int) -> np.ndarray:
    """(S, 4) cf cells at (d, m) → S accuracy values."""
    cf = cf_stack[:, d, m]  # (S, 4): TP, FN, FP, TN
    denom = cf.sum(axis=-1)
    return (cf[:, 0] + cf[:, 3]) / np.maximum(denom, 1)


def build_table4(counts_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """For each center: baseline single-station P[2,1] (mean ± std), and
    over all combos with at least 1 neighbor, count how many have mean_P > /
    < baseline_mean, plus top-3 combos by mean_P."""
    rows = []
    for center, sub in counts_df.groupby("center"):
        single = sub[sub["n_neighbors"] == 0]
        if len(single) == 0:
            continue
        s_prec = single.iloc[0]["precision_stack"][:, 2, 1]
        s_acc = _accuracy_at_cell(single.iloc[0]["cf_stack"], 2, 1)
        baseline_mean = float(np.mean(s_prec))
        baseline_std = float(np.std(s_prec))
        baseline_a = float(np.mean(s_acc))

        with_n = sub[sub["n_neighbors"] > 0].copy()
        if len(with_n) == 0:
            rows.append(dict(center=center, n_seeds=int(single.iloc[0]["n_seeds"]),
                             baseline_p=fmt_mean_std(s_prec.tolist()),
                             baseline_p_mean=baseline_mean,
                             baseline_a_mean=baseline_a,
                             n_combos=0, improved=0, decreased=0,
                             top1="", top2="", top3=""))
            continue

        with_n["P_mean"] = with_n["precision_stack"].apply(lambda ps: float(np.mean(ps[:, 2, 1])))
        with_n["P_std"] = with_n["precision_stack"].apply(lambda ps: float(np.std(ps[:, 2, 1])))
        with_n["A_mean"] = with_n["cf_stack"].apply(lambda cs: float(np.mean(_accuracy_at_cell(cs, 2, 1))))
        improved = int((with_n["P_mean"] > baseline_mean).sum())
        decreased = int((with_n["P_mean"] < baseline_mean).sum())
        top = with_n.sort_values("P_mean", ascending=False).head(3)
        names = []
        for _, row in top.iterrows():
            tag = "_".join([center, *row["neighbors"]])
            names.append(f"({row['P_mean']:.2f}±{row['P_std']:.2f}, {row['A_mean']:.2f}, {tag})")
        while len(names) < 3:
            names.append("")
        rows.append(dict(center=center, n_seeds=int(single.iloc[0]["n_seeds"]),
                         baseline_p=fmt_mean_std(s_prec.tolist()),
                         baseline_p_mean=baseline_mean,
                         baseline_a_mean=baseline_a,
                         n_combos=int(len(with_n)),
                         improved=improved, decreased=decreased,
                         top1=names[0], top2=names[1], top3=names[2]))
    df = pd.DataFrame(rows).sort_values("center").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def build_table5(counts_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Models that ever predict M≥4 in any seed (m_bin∈{4..9}). Output mean P
    over seeds (with seeds where TP+FP=0 contributing 0)."""
    rows = []
    for _, r in counts_df.iterrows():
        cf = r["cf_stack"]  # (S, 5, 10, 4)
        prec = r["precision_stack"]  # (S, 5, 10)
        for d in range(5):
            for m in range(4, 10):
                # any seed with TP+FP>0?
                any_fired = int((cf[:, d, m, 0] + cf[:, d, m, 2]).max()) > 0
                if not any_fired:
                    continue
                seed_p = prec[:, d, m]
                seed_tp = cf[:, d, m, 0]
                seed_fp = cf[:, d, m, 2]
                name = "_".join([r["center"], *r["neighbors"]])
                rows.append(dict(model=name, d=d, m=m, n_seeds=int(r["n_seeds"]),
                                 precision_mean=float(np.mean(seed_p)),
                                 precision_std=float(np.std(seed_p)),
                                 tp_sum=int(seed_tp.sum()), fp_sum=int(seed_fp.sum())))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def build_binary_summary(exp_dir: Path) -> dict:
    """Aggregate over BCE binary results (per-seed stored)."""
    by_combo: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for p in exp_dir.rglob("result_*_binary_seed*.pt"):
        center, neighbors, head, seed = parse_result_filename(p)
        if head != "binary":
            continue
        r = torch.load(p, map_location="cpu", weights_only=False)
        by_combo.setdefault((center, tuple(neighbors)), []).append(r)
    n_models = sum(len(v) for v in by_combo.values())
    n_combos = len(by_combo)
    pred_pos = sum(int(s.get("pred_pos", 0)) for v in by_combo.values() for s in v)
    pred_max_overall = max(
        (float(s.get("pred_max", -1.0)) for v in by_combo.values() for s in v),
        default=-1.0,
    )
    # combos where ALL seeds have pred_pos == 0
    combos_all_zero = sum(
        1 for v in by_combo.values() if all(int(s.get("pred_pos", 0)) == 0 for s in v)
    )
    return dict(n_models=n_models, n_combos=n_combos, total_pred_pos=pred_pos,
                max_prob_observed=pred_max_overall, combos_all_zero=combos_all_zero)


def write_summary(reports_dir: Path, table2: pd.DataFrame, table3: pd.DataFrame,
                  table4: pd.DataFrame, table5: pd.DataFrame, bin_summary: dict) -> None:
    has_pred = table3[table3["has_prediction"]]
    md = [
        "# Method 3 — aggregated results (multi-seed)\n",
        f"- Total alive centers in scope: **{len(table2)}**",
        f"- Centers with mean(P) > 0 in any headline cell: **{len(has_pred)}**",
        f"- Counts-head combos with seeds: **{len(table4)}**",
        "\n## Headline finding (matches PDF 4.1.4 / 4.1.5)",
        f"- Binary (BCE, M≥4 depth<30) seed-level runs: {bin_summary['n_models']}"
        f" ({bin_summary['n_combos']} unique combos × seeds)",
        f"- Combos where ALL seeds output 0 positives on test: **{bin_summary['combos_all_zero']}** "
        f"/ {bin_summary['n_combos']}",
        f"- Sum of test-set positive predictions across all seed runs: "
        f"**{bin_summary['total_pred_pos']}**",
        f"- Max sigmoid output observed across all seed runs: "
        f"{bin_summary['max_prob_observed']:.4f}",
        "\nPDF 4.1.5 reports that all 26,335 binary models output 0 on the test set; the "
        "`combos_all_zero` count is the multi-seed analogue (i.e. how many combos consistently "
        "collapse to 0 across all seeds).\n",
    ]
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
