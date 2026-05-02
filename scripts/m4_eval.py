"""Evaluate Method 4 multi-seed checkpoints — produce PDF tables 6-10 and figures 6-17.

Walks `exp/m4/*_seed{N}.pt`, groups by (input, T, tau), computes per-seed
ROC/AUC on full train and test splits, and writes:
  reports/m4_table6_pos_ratio.csv
  reports/m4_table{7,8}_*_auc.csv          — exp1 train/test AUC, "mean ± std" cells
  reports/m4_table{7,8}_*_auc_per_seed.csv — exp1 per-seed raw values
  reports/m4_table{9,10}_*_auc_grid.csv         — exp2 grid, "mean ± std"
  reports/m4_table{9,10}_*_auc_grid_per_seed.csv — exp2 per-seed
  reports/figures/m4_fig{6,7}_roc_{tr,te}.png   — exp1 ROC: thin per-seed + thick mean
  reports/figures/m4_fig{8..17}_*.png            — exp2 sweeps, same style
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from cwa4.data import Method4Dataset, load_method4_sources
from cwa4.models import ClassifierM4

EXP1_INPUTS = ("gnss", "stats", "all")
EXP1_TAUS = (1, 90, 180, 365, 730)
EXP2_TS = (30, 90, 180, 365, 730)
EXP2_TAUS = (1, 90, 180, 365, 730)
COMMON_FPR = np.linspace(0.0, 1.0, 201)
SEED_RE = re.compile(r"_seed(\d+)\.pt$")


def predict(model: torch.nn.Module, ds, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    logits, ys = [], []
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(device)
            logits.append(model(x).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def roc_auc(model, ds, device) -> tuple[np.ndarray, np.ndarray, float]:
    z, y = predict(model, ds, device)
    prob = 1.0 / (1.0 + np.exp(-z))
    fpr, tpr, _ = roc_curve(y, prob)
    return fpr, tpr, float(auc(fpr, tpr))


def fmt_mean_std(xs: list[float]) -> str:
    if not xs:
        return "nan"
    return f"{np.mean(xs):.3f} ± {np.std(xs):.3f}"


def interp_to_common(fpr: np.ndarray, tpr: np.ndarray) -> np.ndarray:
    """Interpolate TPR to COMMON_FPR grid (preserves area)."""
    return np.interp(COMMON_FPR, fpr, tpr)


def build_table6(sources, out_csv: Path) -> pd.DataFrame:
    rows = []
    for tau in EXP1_TAUS:
        trn = Method4Dataset(sources, T=730, tau=tau, input_kind="all", target_kind="binary", split="trn")
        tst = Method4Dataset(sources, T=730, tau=tau, input_kind="all", target_kind="binary", split="tst")
        rows.append(dict(
            tau=tau,
            train_pos=trn.positive_count(), train_n=len(trn),
            train_pct=trn.positive_count() / len(trn) * 100,
            test_pos=tst.positive_count(), test_n=len(tst),
            test_pct=tst.positive_count() / len(tst) * 100,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def collect_runs(exp_dir: Path, sources, device) -> dict:
    """Returns runs[(input, T, tau)] = list of dicts (one per seed) with
    fpr_tr, tpr_tr, auc_tr, fpr_te, tpr_te, auc_te, seed."""
    runs: dict[tuple[str, int, int], list[dict]] = {}
    for ckpt in sorted(exp_dir.glob("*_seed*.pt")):
        m = SEED_RE.search(ckpt.name)
        if m is None:
            continue
        seed = int(m.group(1))
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = meta["args"]
        if a.get("target") != "binary":
            continue
        model = ClassifierM4(meta["input_dim"], h_dim=a["h_dim"]).to(device)
        model.load_state_dict(meta["state_dict"])
        trn = Method4Dataset(sources, T=a["T"], tau=a["tau"], input_kind=a["input"],
                              target_kind="binary", split="trn")
        tst = Method4Dataset(sources, T=a["T"], tau=a["tau"], input_kind=a["input"],
                              target_kind="binary", split="tst")
        fpr_tr, tpr_tr, auc_tr = roc_auc(model, trn, device)
        fpr_te, tpr_te, auc_te = roc_auc(model, tst, device)
        key = (a["input"], a["T"], a["tau"])
        runs.setdefault(key, []).append(dict(
            seed=seed, fpr_tr=fpr_tr, tpr_tr=tpr_tr, auc_tr=auc_tr,
            fpr_te=fpr_te, tpr_te=tpr_te, auc_te=auc_te,
        ))
        print(f"  {ckpt.name}: AUC train={auc_tr:.4f} test={auc_te:.4f}")
    return runs


def write_exp1_tables(runs: dict, reports_dir: Path) -> None:
    rows_tr_str, rows_te_str = [], []
    rows_tr_raw, rows_te_raw = [], []
    for inp in EXP1_INPUTS:
        rec_tr_s = {"input": inp}; rec_te_s = {"input": inp}
        for tau in EXP1_TAUS:
            seeds = runs.get((inp, 730, tau), [])
            tr_aucs = [r["auc_tr"] for r in seeds]
            te_aucs = [r["auc_te"] for r in seeds]
            rec_tr_s[f"tau{tau}"] = fmt_mean_std(tr_aucs)
            rec_te_s[f"tau{tau}"] = fmt_mean_std(te_aucs)
            for r in seeds:
                rows_tr_raw.append({"input": inp, "tau": tau, "seed": r["seed"], "auc": r["auc_tr"]})
                rows_te_raw.append({"input": inp, "tau": tau, "seed": r["seed"], "auc": r["auc_te"]})
        rows_tr_str.append(rec_tr_s); rows_te_str.append(rec_te_s)
    pd.DataFrame(rows_tr_str).to_csv(reports_dir / "m4_table7_train_auc.csv", index=False)
    pd.DataFrame(rows_te_str).to_csv(reports_dir / "m4_table8_test_auc.csv", index=False)
    pd.DataFrame(rows_tr_raw).to_csv(reports_dir / "m4_table7_train_auc_per_seed.csv", index=False)
    pd.DataFrame(rows_te_raw).to_csv(reports_dir / "m4_table8_test_auc_per_seed.csv", index=False)


def write_exp2_grids(runs: dict, reports_dir: Path) -> None:
    grid_tr = pd.DataFrame(index=EXP2_TS, columns=EXP2_TAUS, dtype=object)
    grid_te = pd.DataFrame(index=EXP2_TS, columns=EXP2_TAUS, dtype=object)
    raw_rows = []
    for T in EXP2_TS:
        for tau in EXP2_TAUS:
            seeds = runs.get(("all", T, tau), [])
            tr_aucs = [r["auc_tr"] for r in seeds]
            te_aucs = [r["auc_te"] for r in seeds]
            grid_tr.at[T, tau] = fmt_mean_std(tr_aucs)
            grid_te.at[T, tau] = fmt_mean_std(te_aucs)
            for r in seeds:
                raw_rows.append({"T": T, "tau": tau, "seed": r["seed"],
                                  "auc_tr": r["auc_tr"], "auc_te": r["auc_te"]})
    grid_tr.index.name = "T"; grid_te.index.name = "T"
    grid_tr.to_csv(reports_dir / "m4_table9_train_auc_grid.csv")
    grid_te.to_csv(reports_dir / "m4_table10_test_auc_grid.csv")
    pd.DataFrame(raw_rows).to_csv(reports_dir / "m4_table9_10_per_seed.csv", index=False)


def plot_roc_with_mean(ax, seeds_list: list[dict], key: str, label_prefix: str, color):
    """Plot per-seed thin lines + thick mean line over COMMON_FPR."""
    if not seeds_list:
        return
    interps = []
    for r in seeds_list:
        tpr_i = interp_to_common(r[f"fpr_{key}"], r[f"tpr_{key}"])
        interps.append(tpr_i)
        ax.plot(COMMON_FPR, tpr_i, color=color, alpha=0.25, linewidth=0.8)
    mean_tpr = np.mean(interps, axis=0)
    mean_auc = float(auc(COMMON_FPR, mean_tpr))
    ax.plot(COMMON_FPR, mean_tpr, color=color, linewidth=2,
            label=f"{label_prefix} (AUC={mean_auc:.3f}, n={len(seeds_list)})")


def write_exp1_figures(runs: dict, fig_dir: Path) -> None:
    cmap = plt.get_cmap("tab20")
    for fig_id, key, title in [("fig6", "tr", "Training set"), ("fig7", "te", "Test set")]:
        plt.figure(figsize=(7, 6))
        ax = plt.gca()
        for i, (inp, tau) in enumerate([(inp, tau) for inp in EXP1_INPUTS for tau in EXP1_TAUS]):
            seeds = runs.get((inp, 730, tau), [])
            plot_roc_with_mean(ax, seeds, key, f"{inp}(τ={tau})", cmap(i % 20))
        ax.plot([0, 1], [0, 1], "k--", label="random")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"M4 exp1 ROC — {title}")
        ax.legend(fontsize=6, loc="lower right", ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"m4_{fig_id}_roc_{key}.png", dpi=150)
        plt.close()


def write_exp2_figures(runs: dict, fig_dir: Path) -> None:
    cmap = plt.get_cmap("viridis")
    fig_id_T = {30: "fig8", 90: "fig9", 180: "fig10", 365: "fig11", 730: "fig12"}
    for T, fig_id in fig_id_T.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, key, sub in zip(axes, ("tr", "te"), ("Train", "Test")):
            for i, tau in enumerate(EXP2_TAUS):
                seeds = runs.get(("all", T, tau), [])
                plot_roc_with_mean(ax, seeds, key, f"τ={tau}", cmap(i / max(1, len(EXP2_TAUS) - 1)))
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"T={T} {sub}")
            ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_dir / f"m4_{fig_id}_T{T}.png", dpi=150)
        plt.close()

    fig_id_tau = {1: "fig13", 90: "fig14", 180: "fig15", 365: "fig16", 730: "fig17"}
    for tau, fig_id in fig_id_tau.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, key, sub in zip(axes, ("tr", "te"), ("Train", "Test")):
            for i, T in enumerate(EXP2_TS):
                seeds = runs.get(("all", T, tau), [])
                plot_roc_with_mean(ax, seeds, key, f"T={T}", cmap(i / max(1, len(EXP2_TS) - 1)))
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"τ={tau} {sub}")
            ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_dir / f"m4_{fig_id}_tau{tau}.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp/m4")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    exp_dir = Path(args.exp); reports_dir = Path(args.reports)
    fig_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sources = load_method4_sources()
    build_table6(sources, reports_dir / "m4_table6_pos_ratio.csv")
    runs = collect_runs(exp_dir, sources, device)
    write_exp1_tables(runs, reports_dir)
    write_exp2_grids(runs, reports_dir)
    write_exp1_figures(runs, fig_dir)
    write_exp2_figures(runs, fig_dir)
    print(f"[m4_eval] tables -> {reports_dir}/, figures -> {fig_dir}/")


if __name__ == "__main__":
    main()
