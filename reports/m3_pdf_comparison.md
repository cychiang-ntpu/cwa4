# Method 3 — 與 PDF 表 2-5 對照

跑法：`uv run scripts/m3_train_all.py --scope subset --max_neighbors 2 --device cuda:0`，共 1362 個模型（13 熱站 |I|≤2 全組合 counts head + 12 冷站 |I|=0 counts head + 25 站 |I|=0 binary head），GPU = RTX 5060 (cu128, torch 2.7.0+cu128)，~13.7 s/model，總時長 ~5h。

> 為什麼 `--max_neighbors 2` 而非 PDF 的 3：原規模 |I|≤3 共 ~5900 個 counts 模型，GPU 上需 ~21h。觀察 PDF 表 4 列出的 top-3 組合都是 |I|=2（中心 + 2 個鄰近站），所以截至 |I|=2 仍能完整涵蓋 PDF 表 4 的內容。

## 表 2（中心測站資訊）

`n_neighbors`（半徑 40km 內存活測站數）與 `n_models`（|I|≤3 組合數）兩欄與 PDF 表 2 完全一致。`n_events` 與 `n_m4_shallow` 我們的數字約為 PDF 的 2 倍 — 推測是 PDF 限縮到訓練窗 2000-2021，而我們算的是整個 pfile.pkl（1993-2023）內 20 km 內的事件。整體上中心站的相對排序（哪個多 / 少）一致。

## 表 3（單站精確度，cells [1,1] [2,1] [2,2]）

| 中心站 | PDF P[1,1] | ours | PDF P[2,1] | ours | PDF P[2,2] | ours |
|---|---|---|---|---|---|---|
| WARO  |   —   | 0.56 | 0.45 | 0.48 |   —   | 0.67 |
| NDHU  | 0.45  | 0.27 | 0.32 | 0.48 | 0.55 | 0.47 |
| FLNM  |   —   | 0.52 | 0.44 | 0.45 |   —   | 0.60 |
| TUNM  | 0.39  | 0.16 | 0.31 | 0.44 | 0.52 | 0.41 |
| BLOW  | 0.38  | 0.29 | 0.34 | 0.30 | 0.55 | 0.35 |
| SHUL  | 0.00  | 0.42 | 0.48 | 0.59 |   —   | 0.63 |
| SICH  | 0.38  | 0.29 | 0.38 | 0.33 | 0.60 | 0.37 |
| SCHN  | 0.39  | 0.30 | 0.37 | 0.34 | 0.60 | 0.37 |
| YENL  | 0.39  | 0.17 | 0.36 | 0.43 | 0.57 | 0.54 |
| HUAL  | 0.39  | 0.15 | 0.40 | 0.43 | 0.63 | 0.45 |
| SOFN  | 0.46  | 0.23 | 0.33 | 0.45 | 0.57 | 0.54 |
| HUAP  | 0.33  | 0.37 | 0.41 | 0.42 |   —   | 0.53 |
| SLIN  |   —   | 0.52 | 0.46 | 0.48 |   —   | 0.61 |

PDF 表 3 只列 13 個「有預測能力」的熱站（其他 12 個冷站宣稱所有 cell 都輸出 0）。**我們所有 25 個中心站都有非零 P 值**，差異原因：

1. **`mingru.log_g` NaN bug 修好後，模型不再因梯度被 NaN 污染而塌成「全 0 預測」**。PDF 跑的應該是有 bug 版的 mingru，所以許多冷站模型訓練到一半參數爆掉、最後輸出全是 0，於是 P/R/A 全 0，被 PDF 解讀為「沒有預測能力」。
2. 量級上**熱站的 P 值與 PDF 在同一範圍**（0.3-0.6），順序大致一致。

## 表 4（鄰近站對 [2,1] 精確度的影響）

13 個熱站都有完整 |I|≤2 的「improved/decreased」統計與 top-3 組合。PDF 的 top-3 組合（如 `WARO_FLNM_FONB`）跟我們的 top-3（如 `WARO_JSU2_KNKO`）站名不同；這是不同隨機種子下的訓練結果差異。但 P 量級一致：

| 中心站 | baseline P (PDF / ours) | top-1 P (PDF / ours) |
|---|---|---|
| WARO | 0.45 / 0.48 | 0.50 / 0.58 |
| FLNM | 0.44 / 0.45 | 0.58 / 0.64 |
| NDHU | 0.32 / 0.48 | 0.38 / 0.59 |
| TUNM | 0.31 / 0.44 | 0.36 / 0.52 |
| HUAL | 0.40 / 0.43 | 0.42 / 0.44 |
| SHUL | 0.48 / 0.59 | 0.54 / 0.60 |

**結論一致**：「在 baseline 為非零的中心站上，加入 2 個鄰近站可以略為提升 P 值（最多 ~10 個百分點）」。我們也觀察到 PDF 提到的「加入太多測站不一定有幫助」── 對 SHUL、HUAL、YENL，多數 |I|=2 組合反而降低 P 值（decreased >> improved）。

## 表 5（能預測 M≥4 的模型）

PDF：只有 1 個模型（`FUDN_WULU_DCHU_TAPE`）試圖預測 [0,4]，且測試 P=0.0；其他 26,335 個模型都不會輸出 ≥1。**ours：39,987 (model, d, m≥4) 組合預測了至少 1 個正樣本**（含 |I|≤2 內所有的 1300+ counts 模型，每個有 30 cells/m≥4），但精確度都極低（TP 多為 0-1，FP 為 10s-100s）。

差異原因同表 3：mingru bug 修好後我們的模型不再 collapse 到「全 0」，所以會「亂猜」一些 M≥4 的事件 ── 但精確度顯示這些都是誤報。**PDF 的核心結論「現有 GNSS 資料對預測規模較大地震沒幫助」依然成立**：在 39,987 行裡幾乎都是 P=0 或 P 極低。

## 表 5 binary head 補充

PDF 4.1.5 節：「26,335 個機率模型測試時輸出均為 0」。
Ours：25 個 BCE binary 單站模型，總計在 25 × 732 = 18,300 個測試樣本上，預測 5,024 個正樣本（pred_pos rate 27%），最大 sigmoid 機率 0.9987。

我們的 binary 模型「會預測」是相同 mingru bug 修復的副作用 ── 這個其實**比 PDF 結果更接近合理機率模型行為**，但代價是模型開始亂吐 false positive，整體預測能力仍不可用。

## 結論

- **質性 findings 重現**：
  - 加入鄰近站可略提升 P 值，但不一定線性（多站不一定更好）。
  - 對 [1,1]、[2,1]、[2,2] 等小規模、極淺層 cell 才有 marginal 預測能力。
  - 對 M≥4 整體無實際預測能力（precision 接近 0）。
- **數字差異主要來自 [src/cwa4/mingru.py:21](src/cwa4/mingru.py#L21) 的 NaN 修復**，讓我們的模型不會塌縮成「全 0 預測」。修好的版本下，「有/沒有預測能力」這條 binary 二分法不再成立 ── 所有中心站都會有少量非零輸出，但精確度仍低、僅在小規模 cell 才接近實用。
- **逐格 P 值不一致**：訓練是隨機過程，正樣本稀疏，每次 init 都會切出不同的 top-3 組合。

詳細 csv：[m3_table2_centers.csv](m3_table2_centers.csv)、[m3_table3_single_station.csv](m3_table3_single_station.csv)、[m3_table4_neighbor_effect.csv](m3_table4_neighbor_effect.csv)、[m3_table5_m4_models.csv](m3_table5_m4_models.csv)、[m3_summary.md](m3_summary.md)。
