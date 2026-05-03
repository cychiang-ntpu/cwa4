# Method 3 — aggregated results (multi-seed)

- Total alive centers in scope: **25**
- Centers with mean(P) > 0 in any headline cell: **25**
- Counts-head combos with seeds: **25**

## Headline finding (matches PDF 4.1.4 / 4.1.5)
- Binary (BCE, M≥4 depth<30) seed-level runs: 125 (25 unique combos × seeds)
- Combos where ALL seeds output 0 positives on test: **0** / 25
- Sum of test-set positive predictions across all seed runs: **22977**
- Max sigmoid output observed across all seed runs: 0.9995

PDF 4.1.5 reports that all 26,335 binary models output 0 on the test set; the `combos_all_zero` count is the multi-seed analogue (i.e. how many combos consistently collapse to 0 across all seeds).
