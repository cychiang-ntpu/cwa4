"""Method 4 experiment 2: T × τ grid (PDF figures 8-17, tables 9-10).

Trains 25 models: T ∈ {30, 90, 180, 365, 730} × τ ∈ {1, 90, 180, 365, 730}
with input=all, target=binary, loss=focal(γ=3).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="exp/m4")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    Ts = (30, 90, 180, 365, 730)
    taus = (1, 90, 180, 365, 730)
    for T, tau in product(Ts, taus):
        cmd = [
            sys.executable, "scripts/m4_train.py",
            "--T", str(T), "--tau", str(tau),
            "--input", "all", "--target", "binary",
            "--loss", "focal", "--gamma", "3",
            "--out", args.out,
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
