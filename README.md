# cwa4 — 114 年地震前兆觀測 (Method 3 & Method 4) 重現流水線

本 repo 對應江振宇老師「114 年地震前兆觀測作業與分析技術相關研究 - 以深度學習影像處理為基礎之大地變形地震前兆研究」(2025-11-17 期末報告)。目標是用花蓮地區 GNSS 測站資料 + 過去地震統計，重現報告 Method 3 與 Method 4 的全部表格與圖示。

## 計畫需求摘要
- 規模 ≥ 5.5、深度 < 30 km（具傷害性的淺層地震）
- 能量公式 log E = 11.8 + 1.5 M
- 預估時間區間：未來 1 / 90 / 180 / 365 / 730 天
- 觀測時間長度：30 / 90 / 180 / 365 / 730 天

## 安裝

需要 [uv](https://docs.astral.sh/uv/)。Windows 上若沒裝可執行：

```pwsh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```pwsh
uv python install 3.13     # torch 2.7 沒 py3.14 wheel；3.13 可
uv sync --python 3.13      # 自動裝 torch 2.7.0+cu128 (RTX 50-series 需要)
```

沒 GPU 的話以下所有 `--device cuda:0` 都改 `--device cpu` 即可（但會慢非常多）。

## 端到端流程

```pwsh
# 1. 原始資料 → pickle
uv run scripts/1_convert_gnss_format.py
uv run scripts/2_convert_pfile_format.py
uv run scripts/3_create_datasets.py

# 2. 驗 Method 4 資料切割對得上 PDF 表 6 的 (H/N) 數字 (7 個 test 全過)
uv run pytest tests/

# 3. Method 4 — 訓練 + 表 6-10、圖 6-17
uv run scripts/m4_run_exp1.py --device cuda:0       # 15 unique config × 5 seed = 75 ckpt
uv run scripts/m4_run_exp2.py --device cuda:0       # 25 unique config × 5 seed = 125 ckpt
uv run scripts/m4_eval.py --device cuda:0
uv run scripts/m4_plot_data.py                      # PDF 圖 3-5

# 4. Method 3 — 訓練 + 表 2-5（GPU 上 ~30h，會 resume）
uv run scripts/m3_train_all.py --device cuda:0      # 1362 unique config × 5 seed = 6810 result
uv run scripts/m3_collect_tables.py
```

中斷 (Ctrl+C / 重開機 / segfault) 後再跑同一條指令即可 resume — `m{3,4}_train.py` 內含 `if ckpt.exists(): continue` 跳過已完成的 (config, seed)。

## 對應表

| PDF 內容 | 產出檔案 |
| --- | --- |
| 表 2（中心測站資訊） | `reports/m3_table2_centers.csv` |
| 表 3（單站精確度，5 seed） | `reports/m3_table3_single_station.csv` |
| 表 4（鄰近站影響，5 seed） | `reports/m3_table4_neighbor_effect.csv` |
| 表 5（能預測 M≥4 的模型） | `reports/m3_table5_m4_models.csv` |
| 表 6（正樣本比例） | `reports/m4_table6_pos_ratio.csv` |
| 表 7-8（exp1 train/test AUC，5 seed mean±std） | `reports/m4_table{7,8}_*_auc.csv` (+ `*_per_seed.csv`) |
| 表 9-10（exp2 train/test AUC grid，5 seed） | `reports/m4_table{9,10}_*_auc_grid.csv` (+ `m4_table9_10_per_seed.csv`) |
| 圖 3-5（資料分布） | `reports/figures/m4_fig{3,4,5}_*.png` |
| 圖 6-7（exp1 ROC：5 條 per-seed + 1 條 mean） | `reports/figures/m4_fig{6,7}_roc_*.png` |
| 圖 8-12（固定 T 掃 τ） | `reports/figures/m4_fig{8..12}_T*.png` |
| 圖 13-17（固定 τ 掃 T） | `reports/figures/m4_fig{13..17}_tau*.png` |
| 與 PDF 逐格對照分析 | `reports/m{3,4}_pdf_comparison.md` |

## 多 seed 設定

所有訓練腳本預設跑 **5 seeds (`0,1,2,3,4`)**，可用 `--seeds` 旗標覆蓋（例：`--seeds 0` 只跑 1 個）。

> **動機**：M4 test AUC 跨 5 個 seed 標準差就有 ±0.05–0.17，單 seed 結果跟 PDF 比是雜訊。改成 mean ± std 才能下「真的有差距 vs. 在 noise 內」的結論。詳見 `reports/m4_pdf_comparison.md`。

聚合表格 (`m{3,4}_table*_*.csv`) 每格用 `mean ± std` 字串；逐 seed 原始值另存 `*_per_seed.csv`。

## 訓練範圍備註

- **Method 3 預設 `--scope subset --max_neighbors 2`**：
  - 13 個 PDF 表 3 列出有預測能力的中心站跑全部 |I|≤2 鄰近組合（counts head）
  - 12 個冷門中心站只跑單站
  - 25 個中心站單站 BCE/binary head 統計（不存 ckpt）
  - 共 1362 unique config × 5 seed = 6810 個 result/ckpt
  - PDF 規格 |I|≤3 約 5900 unique × 5 seed = 30k 訓練，需 100+ GPU 小時。`max_neighbors=2` 仍涵蓋 PDF 表 4 全部 top-3 (都是 |I|=2)。
  - 需要完整 |I|≤3 規模時，加 `--scope full` 與 `--shard i/N` 分片。
- **Method 4 預設 binary target + focal loss (γ=3)**。`m4_train.py` 仍接受 `--target {binary,count,logE}` 與 `--loss {focal,bce,balanced,mse}` 旗標方便擴充。
- **切分日期**：Method 3 訓 89-110 / 測 111-112；Method 4 訓 89-108 / 測 109-112。

## 程式碼結構

```
src/cwa4/
  data/preprocessing.py     # 共用資料 helper (alive stations, hist binning, date split)
  data/method3.py           # Method3Dataset (365-day window, 5×10 hist)
  data/method4.py           # Method4Dataset, make_train_dev, compute_feature_stats
  models/model_a.py         # PDF Table 1 架構（counts/binary 雙 head）
  models/classifier_m4.py   # Method 4 binary classifier (logits)
  losses.py                 # BalancedBCEWithLogits, FocalLossWithLogits
  mingru.py                 # MinGRU + parallel scan (含 log_g NaN-bug 的修復)
  encoder.py                # Swish, GRUSwishNorm (classifier_m4 共用)
  layernorm.py              # LayerNorm1d
scripts/
  1_convert_gnss_format.py / 2_convert_pfile_format.py / 3_create_datasets.py
  m3_train_all.py / m3_collect_tables.py
  m4_train.py / m4_run_exp1.py / m4_run_exp2.py / m4_eval.py / m4_plot_data.py
tests/
  test_method4_pos_ratio.py / test_losses.py
notebooks/                  # 探索性的 jupyter notebook (與主流水線無關，未必能跑)
```

## 與 PDF 的差異備註

- 我們在 `src/cwa4/mingru.py` 修了一個 `log_g(x)` 對 x<0 走入 `log(負數)` 的 NaN bug。PDF 跑的是 buggy 版本，許多模型訓練到一半梯度被 NaN 污染，輸出塌陷成「全 0」，因此 PDF 的訓練 AUC 看似較低 (M4) 或許多中心站被當「沒預測能力」(M3)。修好後 ours 的 AUC/精確度普遍較高，質性結論仍與 PDF 一致：**GNSS 資料對 M≥4 / M≥5.5 的強震無實質預測能力**。
- 完整逐格 (PDF 1 seed) vs (ours mean ± std) 對照與 ±2σ 是否重疊的判讀，看 `reports/m{3,4}_pdf_comparison.md`。
