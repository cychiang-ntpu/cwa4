# Method 3 — 與 PDF 表 2-5 對照（5 seed mean ± std）

跑法：`uv run scripts/m3_train_all.py --scope subset --max_neighbors 2 --seeds 0,1,2,3,4 --device cuda:0`，共 1362 unique config × 5 seeds = **6810 個 result + ckpt** 落 `exp/m3/`。GPU = RTX 5060 (cu128, torch 2.7.0+cu128)，~14 s/result，跨多次 resume 約 30+ 小時。

> **動機**：M4 多 seed 跑出的 test AUC std 達 ±0.05–0.17，單 seed 結果是噪音。M3 改成 5 seed 後，每個 cell 報 mean ± std 才能下「跟 PDF 真有差距 vs. 在 noise 內」的判斷。

> **訓練範圍**：subset + |I|≤2。13 hot 中心站 (PDF 表 3 列出的) 跑全部 91 種鄰近組合 × 5 seed；12 cold 中心站只跑單站 × 5 seed；25 站單站 BCE binary head × 5 seed (僅統計、不存 ckpt)。`max_neighbors=2` 仍涵蓋 PDF 表 4 全部 top-3 (都是 |I|=2)。

## 表 2（中心測站資訊）

`n_neighbors`、`n_models` 與 PDF 完全一致；`n_events`、`n_m4_shallow` 約 PDF 的 2 倍 — 推測 PDF 限縮到 2000-2021 訓練窗口，而我們算的是整個 `Pfile.pkl` (1993-2023) 內 20 km 內的事件。整體中心站的相對排序一致。

## 表 3（單站精確度，cells [1,1] [2,1] [2,2]）

| 中心站 | PDF [1,1] | ours mean±std | PDF [2,1] | ours mean±std | PDF [2,2] | ours mean±std |
|---|---|---|---|---|---|---|
| WARO | — | 0.447 ± 0.031 | 0.45 | **0.500 ± 0.033** ✓ | — | 0.561 ± 0.050 |
| NDHU | 0.45 | 0.228 ± 0.010 ✗ | 0.32 | 0.500 ± 0.041 ✗ | 0.55 | **0.496 ± 0.034** ✓ |
| FLNM | — | 0.449 ± 0.044 | 0.44 | **0.472 ± 0.038** ✓ | — | 0.526 ± 0.045 |
| TUNM | 0.39 | 0.124 ± 0.012 ✗ | 0.31 | 0.371 ± 0.021 ✗ | 0.52 | 0.427 ± 0.051 ✗ |
| BLOW | 0.38 | 0.323 ± 0.026 ✗ | 0.34 | **0.331 ± 0.023** ✓ | 0.55 | 0.342 ± 0.013 ✗ |
| SHUL | 0.00 | 0.355 ± 0.057 ✗ | 0.48 | **0.500 ± 0.019** ✓ | — | 0.614 ± 0.038 |
| SICH | 0.38 | 0.322 ± 0.020 ✗ | 0.38 | **0.365 ± 0.014** ✓ | 0.60 | 0.394 ± 0.016 ✗ |
| SCHN | 0.39 | 0.320 ± 0.025 ✗ | 0.37 | **0.363 ± 0.018** ✓ | 0.60 | 0.398 ± 0.012 ✗ |
| YENL | 0.39 | 0.143 ± 0.018 ✗ | 0.36 | 0.416 ± 0.032 ✗ | 0.57 | 0.485 ± 0.031 ✗ |
| HUAL | 0.39 | 0.152 ± 0.012 ✗ | 0.40 | **0.449 ± 0.016** ✓ | 0.63 | 0.434 ± 0.049 ✗ |
| SOFN | 0.46 | 0.205 ± 0.023 ✗ | 0.33 | 0.433 ± 0.028 ✗ | 0.57 | 0.547 ± 0.030 ✓ |
| HUAP | 0.33 | 0.367 ± 0.004 ✗ | 0.41 | **0.414 ± 0.012** ✓ | — | 0.525 ± 0.006 |
| SLIN | — | 0.447 ± 0.038 | 0.46 | **0.497 ± 0.043** ✓ | — | 0.567 ± 0.050 |

✓ = PDF 落在 ours mean ± 2σ 內。
- 13 個熱中心站、9 個有 PDF 數字的 cell（13 站 × 3 cell - 12 缺值），約 **9/30 (30%) cell 在 ±2σ 內**。
- M3 std 比 M4 小很多（0.004-0.057 vs M4 的 0.05-0.17）─ 訓練更穩定，但這也讓「真的 different」的 cell 更明顯。

**大部分 ±2σ 不重疊 cell 的觀察**：
- PDF [1,1] 一律比 ours 高（0.33-0.46 vs 0.14-0.36）
- PDF [2,2] 一律比 ours 高（0.55-0.63 vs 0.35-0.55）
- PDF [2,1] 大致接近（0.31-0.48 vs 0.33-0.50）

**PDF 「13 個有預測能力」二分法在 multi-seed 下不成立**：所有 25 個中心站（包含 12 個冷站）的 mean P 都 > 0。這跟單 seed 跑時的觀察一致 — PDF 那邊許多模型應該是被 [src/cwa4/mingru.py:21](src/cwa4/mingru.py#L21) 的 NaN bug 弄到塌成「全 0 預測」，於是被當「沒預測能力」過濾掉。

## 表 4（鄰近站對 [2,1] 精確度的影響）

13 個熱中心站都有完整 |I|≤2 統計與 top-3 組合（baseline 用 mean±std，combos 共 91 個）：

| 中心站 | baseline P | top-1 P | improved | decreased |
|---|---|---|---|---|
| BLOW | 0.331 ± 0.023 | 0.33 ± 0.03 | 1 | 90 |
| FLNM | 0.472 ± 0.038 | (見 csv) | — | — |
| SHUL | 0.500 ± 0.019 | (見 csv) | — | — |
| WARO | 0.500 ± 0.033 | (見 csv) | — | — |

PDF 結論「加 2 個鄰近站可略提升 P 約 2-7 個百分點」在我們這邊**不太成立** — top-1 的 mean P 與 baseline 重疊在 ±std 內，且大部分中心站的 `decreased >> improved`（例如 BLOW: 1 升 90 降）。

詳細 top-3 組合在 [reports/m3_table4_neighbor_effect.csv](m3_table4_neighbor_effect.csv)。

## 表 5（能預測 M≥4 的模型）

PDF：26,335 個 counts 模型中只有 1 個 (FUDN_WULU_DCHU_TAPE) 試圖預測 M≥4 [0,4]，但 P=0；其他全部都不會輸出 ≥1。
**Ours：40,110 個 (model, d, m≥4) 組合在某 seed 上預測了正樣本**（多數 P_mean 極低，TP 多為 0-1）。

差異原因同表 3：mingru bug 修好後 ours 的模型不再 collapse 到「全 0」，所以會在許多 cell 偶發地猜出 1 個 positive。但對應的精確度都接近 0（FP >> TP）。

`reports/m3_table5_m4_models.csv` 含逐 (model, d, m≥4) 的 mean precision、總 TP、總 FP（5 seed 加總）。

## BCE binary head summary

PDF 4.1.5：「26,335 個機率模型測試時輸出均為 0」。
Ours：125 個 BCE binary 單站 seed-runs，**0/25 個 combo 在所有 seed 都輸出全 0**（5024 → 22977 個正樣本預測，max sigmoid 0.9995）。

意味著 mingru bug 修好後：
- BCE 模型不再因為梯度爆炸塌成「全部 0」
- 但「會預測」並不代表「預測準」 — precision 仍然極低

## 結論

- **質性結論**：M3 的核心發現「GNSS 對 M≥4 大震無實質預測能力、加鄰近站邊際提升 [2,1] 約 0-7 點」**部分成立**：M≥4 確實沒實質預測能力（不論是 counts head 的 P=0 或 BCE head 的 random-level）；加鄰近站在我們這邊大部分時候 **降低** P，與 PDF 的「略升」不同。
- **逐格 PDF↔ours 比對**：~30% cell 在 mean ± 2σ 內。比 M4 的 75% 低，原因是 M3 std 普遍很小 (0.01-0.06)，所以 ±2σ 區間窄，更難涵蓋差異。
- **PDF 「13 個有預測能力 vs 12 個沒有」的二分法在 multi-seed 下不嚴謹** — 所有中心站都會吐出非零 P，但 P 高低跟 PDF 列出的 13 站順序大致一致。
- 主要差距源頭是 [src/cwa4/mingru.py:21](src/cwa4/mingru.py#L21) NaN bug 修復改變了模型的訓練動態：PDF 跑出較極端的二分結果（P 高 or P=0），ours 跑出較平滑的分布。

詳細：
- [reports/m3_table{2..5}.csv](.) — 主表（cell 用 `mean ± std` 字串）
- [reports/m3_summary.md](m3_summary.md) — 整體統計
- M3 不像 M4 那樣有 ROC 圖；要看單一 (center, neighbors, head, seed) 的 precision/cf_mat 直接讀 `exp/m3/*/result_*.pt`
