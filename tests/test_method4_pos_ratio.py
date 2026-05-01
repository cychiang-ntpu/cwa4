"""Verify that Method4Dataset reproduces PDF Table 6's sample / positive counts.

Reference (PDF table 6, T=730, target=binary, M ≥ 5.5, depth < 30):

    τ (天)    train H/N            test H/N
    1         15/6576              8/732
    90        1083/6576            433/732
    180       2163/6576            523/732
    365       3678/6576            650/732
    730       4979/6576            650/732

Run requires the preprocessed pickles produced by `scripts/3_create_datasets.py`.
The test skips (xfail-like) if those files are missing so the suite can still
run on a fresh checkout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cwa4.data.method4 import Method4Dataset, load_method4_sources, _future_window_sum

DATA_DIR = Path("data")
REQUIRED = [
    DATA_DIR / "hualian_daily_gnss_dXdYdU.pkl",
    DATA_DIR / "hualian_daily_statistics.pkl",
    DATA_DIR / "hualian_target_cnt.pkl",
]

EXPECTED = {
    1: (15, 6576, 8, 732),
    90: (1083, 6576, 433, 732),
    180: (2163, 6576, 523, 732),
    365: (3678, 6576, 650, 732),
    730: (4979, 6576, 650, 732),
}


def _pickles_exist() -> bool:
    return all(p.exists() for p in REQUIRED)


def test_future_window_sum_basic():
    # values:    [0, 1, 0, 0, 2, 0]
    # τ=2 → out[i] = sum(values[i+1 : i+3]); padded with zeros at end
    values = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])
    out = _future_window_sum(values, tau=2)
    np.testing.assert_array_equal(out, np.array([1.0, 0.0, 2.0, 2.0, 0.0, 0.0]))


@pytest.mark.skipif(not _pickles_exist(), reason="run scripts 1-3 to create the pickles first")
@pytest.mark.parametrize("tau", list(EXPECTED.keys()))
def test_table6_counts(tau: int):
    sources = load_method4_sources(DATA_DIR)
    trn = Method4Dataset(sources, T=730, tau=tau, input_kind="all", target_kind="binary", split="trn")
    tst = Method4Dataset(sources, T=730, tau=tau, input_kind="all", target_kind="binary", split="tst")
    h_trn_exp, n_trn_exp, h_tst_exp, n_tst_exp = EXPECTED[tau]
    assert len(trn) == n_trn_exp, f"τ={tau} trn N expected {n_trn_exp}, got {len(trn)}"
    assert len(tst) == n_tst_exp, f"τ={tau} tst N expected {n_tst_exp}, got {len(tst)}"
    assert trn.positive_count() == h_trn_exp, (
        f"τ={tau} trn H expected {h_trn_exp}, got {trn.positive_count()}"
    )
    assert tst.positive_count() == h_tst_exp, (
        f"τ={tau} tst H expected {h_tst_exp}, got {tst.positive_count()}"
    )
