"""Method 4 experiment 1: input ablation × τ.

Trains 15 models: input ∈ {gnss, stats, all} × τ ∈ {1, 90, 180, 365, 730}
with T=730, target=binary, loss=focal(γ=3). Reproduces PDF figures 6-7
and tables 7-8.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="exp/m4")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    inputs = ("gnss", "stats", "all")
    taus = (1, 90, 180, 365, 730)
    for inp, tau in product(inputs, taus):
        cmd = [
            sys.executable, "scripts/m4_train.py",
            "--T", "730", "--tau", str(tau),
            "--input", inp, "--target", "binary",
            "--loss", "focal", "--gamma", "3",
            "--out", args.out,
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
