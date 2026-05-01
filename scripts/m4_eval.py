"""Evaluate Method 4 checkpoints — produce PDF tables 6-10 and figures 6-17.

Walks `exp/m4/*.pt`, reconstructs each model from its saved args, computes
ROC/AUC on full train and test splits, and writes:
  reports/m4_table6_pos_ratio.csv          — positive-sample ratio per τ
  reports/m4_table7_train_auc.csv          — exp1 training AUC (3 inputs × 5 τ)
  reports/m4_table8_test_auc.csv           — exp1 test AUC
  reports/m4_table9_train_auc_grid.csv     — exp2 training AUC (5 T × 5 τ)
  reports/m4_table10_test_auc_grid.csv     — exp2 test AUC
  reports/figures/m4_fig6_train_roc.png    — all 15 exp1 ROC curves on train
  reports/figures/m4_fig7_test_roc.png     — same on test
  reports/figures/m4_fig{8..12}_T{30..730}.png — fixed-T sweeps (train+test)
  reports/figures/m4_fig{13..17}_tau{1..730}.png — fixed-τ sweeps (train+test)
"""
from __future__ import annotations

import argparse
import json
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


def load_ckpt_meta(p: Path) -> dict:
    blob = torch.load(p, map_location="cpu", weights_only=False)
    return blob


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp/m4")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    exp_dir = Path(args.exp)
    reports_dir = Path(args.reports)
    fig_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    sources = load_method4_sources()

    # Table 6 — directly from datasets, no checkpoints needed
    build_table6(sources, reports_dir / "m4_table6_pos_ratio.csv")

    # Index every checkpoint by (input, T, tau)
    rocs: dict[tuple[str, int, int], dict] = {}
    for ckpt in exp_dir.glob("*.pt"):
        meta = load_ckpt_meta(ckpt)
        a = meta["args"]
        if a["target"] != "binary":
            continue
        model = ClassifierM4(meta["input_dim"], h_dim=a["h_dim"]).to(device)
        model.load_state_dict(meta["state_dict"])
        trn = Method4Dataset(sources, T=a["T"], tau=a["tau"], input_kind=a["input"],
                              target_kind="binary", split="trn")
        tst = Method4Dataset(sources, T=a["T"], tau=a["tau"], input_kind=a["input"],
                              target_kind="binary", split="tst")
        fpr_tr, tpr_tr, auc_tr = roc_auc(model, trn, device)
        fpr_te, tpr_te, auc_te = roc_auc(model, tst, device)
        rocs[(a["input"], a["T"], a["tau"])] = dict(
            fpr_tr=fpr_tr, tpr_tr=tpr_tr, auc_tr=auc_tr,
            fpr_te=fpr_te, tpr_te=tpr_te, auc_te=auc_te,
        )
        print(f"  {ckpt.name}: AUC train={auc_tr:.4f} test={auc_te:.4f}")

    # Tables 7, 8 — exp1 (T=730, vary input × τ)
    rows_tr, rows_te = [], []
    for inp in EXP1_INPUTS:
        rec_tr = {"input": inp}
        rec_te = {"input": inp}
        for tau in EXP1_TAUS:
            r = rocs.get((inp, 730, tau))
            rec_tr[f"tau{tau}"] = r["auc_tr"] if r else float("nan")
            rec_te[f"tau{tau}"] = r["auc_te"] if r else float("nan")
        rows_tr.append(rec_tr)
        rows_te.append(rec_te)
    pd.DataFrame(rows_tr).to_csv(reports_dir / "m4_table7_train_auc.csv", index=False)
    pd.DataFrame(rows_te).to_csv(reports_dir / "m4_table8_test_auc.csv", index=False)

    # Tables 9, 10 — exp2 (input=all, T × τ grid)
    grid_tr = pd.DataFrame(index=EXP2_TS, columns=EXP2_TAUS, dtype=float)
    grid_te = pd.DataFrame(index=EXP2_TS, columns=EXP2_TAUS, dtype=float)
    for T in EXP2_TS:
        for tau in EXP2_TAUS:
            r = rocs.get(("all", T, tau))
            grid_tr.at[T, tau] = r["auc_tr"] if r else float("nan")
            grid_te.at[T, tau] = r["auc_te"] if r else float("nan")
    grid_tr.index.name = "T"
    grid_te.index.name = "T"
    grid_tr.to_csv(reports_dir / "m4_table9_train_auc_grid.csv")
    grid_te.to_csv(reports_dir / "m4_table10_test_auc_grid.csv")

    # Figures 6 & 7 — exp1 ROC overlays
    for fig_id, key, title in [("fig6", "tr", "Training set"), ("fig7", "te", "Test set")]:
        plt.figure(figsize=(7, 6))
        for inp in EXP1_INPUTS:
            for tau in EXP1_TAUS:
                r = rocs.get((inp, 730, tau))
                if r is None:
                    continue
                plt.plot(r[f"fpr_{key}"], r[f"tpr_{key}"], label=f"{inp}({tau})", linewidth=1)
        plt.plot([0, 1], [0, 1], "k--", label="random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"M4 exp1 ROC — {title}")
        plt.legend(fontsize=7, loc="best", ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"m4_{fig_id}_roc_{key}.png", dpi=150)
        plt.close()

    # Figures 8-12 — fixed T sweeps
    fig_id_T = {30: "fig8", 90: "fig9", 180: "fig10", 365: "fig11", 730: "fig12"}
    for T, fig_id in fig_id_T.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, key, sub in zip(axes, ("tr", "te"), ("Train", "Test")):
            for tau in EXP2_TAUS:
                r = rocs.get(("all", T, tau))
                if r is None:
                    continue
                ax.plot(r[f"fpr_{key}"], r[f"tpr_{key}"], label=f"τ={tau}", linewidth=1)
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"T={T} {sub}")
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / f"m4_{fig_id}_T{T}.png", dpi=150)
        plt.close()

    # Figures 13-17 — fixed τ sweeps
    fig_id_tau = {1: "fig13", 90: "fig14", 180: "fig15", 365: "fig16", 730: "fig17"}
    for tau, fig_id in fig_id_tau.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, key, sub in zip(axes, ("tr", "te"), ("Train", "Test")):
            for T in EXP2_TS:
                r = rocs.get(("all", T, tau))
                if r is None:
                    continue
                ax.plot(r[f"fpr_{key}"], r[f"tpr_{key}"], label=f"T={T}", linewidth=1)
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"τ={tau} {sub}")
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / f"m4_{fig_id}_tau{tau}.png", dpi=150)
        plt.close()

    print(f"[m4_eval] tables -> {reports_dir}/, figures -> {fig_dir}/")


if __name__ == "__main__":
    main()
