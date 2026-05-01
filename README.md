# cwa4 — 114 年地震前兆觀測 (Method 3 & Method 4) 重現流水線

本 repo 對應江振宇老師「114 年地震前兆觀測作業與分析技術相關研究 - 以深度學習影像處理為基礎之大地變形地震前兆研究」(2025-11-17 期末報告)。目標是用花蓮地區 GNSS 測站資料 + 過去地震統計，重現報告 Method 3 與 Method 4 的全部表格與圖示。

## 計畫需求摘要
- 規模 ≥ 5.5、深度 < 30 km（具傷害性的淺層地震）
- 能量公式 log E = 11.8 + 1.5 M
- 預估時間區間：未來 1 / 90 / 180 / 365 / 730 天
- 觀測時間長度：30 / 90 / 180 / 365 / 730 天

## 安裝

需要 [uv](https://docs.astral.sh/uv/)。Windows 上若沒裝可執行 `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`。

```pwsh
uv python install 3.13     # torch 2.7 沒 py3.14 wheel；3.13 可
uv sync --python 3.13
```

## 端到端流程

```pwsh
# 1. 原始資料 → pickle
uv run scripts/1_convert_gnss_format.py
uv run scripts/2_convert_pfile_format.py
uv run scripts/3_create_datasets.py

# 2. 驗 Method 4 資料切割對得上 PDF 表 6 的 (H/N) 數字
uv run pytest tests/test_method4_pos_ratio.py

# 3. Method 3 — 訓練 + 表 2-5
uv run scripts/m3_train_all.py                     # 預設 subset 範圍
uv run scripts/m3_collect_tables.py

# 4. Method 4 — 訓練 + 表 6-10、圖 6-17（GPU 大幅快於 CPU）
uv run scripts/m4_run_exp1.py --device cuda:0       # PDF 圖 6-7、表 7-8 (15 個模型)
uv run scripts/m4_run_exp2.py --device cuda:0       # PDF 圖 8-17、表 9-10 (25 個模型)
uv run scripts/m4_eval.py --device cuda:0
uv run scripts/m4_plot_data.py                      # PDF 圖 3-5

# CUDA torch (RTX 50-series 需要 cu128 wheels；pyproject 已指向)：
#   uv sync 會自動裝 torch 2.7.0+cu128。沒 GPU 則用 --device cpu。
# 若要與 PDF 表 6-10 對照，看 reports/m4_pdf_comparison.md。
```

## 對應表

| PDF 內容 | 產出檔案 |
| --- | --- |
| 表 2（中心測站資訊） | `reports/m3_table2_centers.csv` |
| 表 3（單站精確度） | `reports/m3_table3_single_station.csv` |
| 表 4（鄰近站影響） | `reports/m3_table4_neighbor_effect.csv` |
| 表 5（能預測 M≥4 的模型） | `reports/m3_table5_m4_models.csv` |
| 表 6（正樣本比例） | `reports/m4_table6_pos_ratio.csv` |
| 表 7-8（exp1 train/test AUC） | `reports/m4_table{7,8}_*_auc.csv` |
| 表 9-10（exp2 train/test AUC grid） | `reports/m4_table{9,10}_*_auc_grid.csv` |
| 圖 3-5（資料分布） | `reports/figures/m4_fig{3,4,5}_*.png` |
| 圖 6-7（exp1 ROC） | `reports/figures/m4_fig{6,7}_roc_*.png` |
| 圖 8-12（固定 T 掃 τ） | `reports/figures/m4_fig{8..12}_T*.png` |
| 圖 13-17（固定 τ 掃 T） | `reports/figures/m4_fig{13..17}_tau*.png` |

## 訓練範圍備註

- **Method 3 預設 `--scope subset`**：13 個 PDF 表 3 列出有預測能力的中心站跑完整 `|I|≤3` 鄰近組合（counts head）+ 12 個冷門中心站只跑 `|I|=0`；所有 25 個中心站只跑單站 BCE/binary head 的彙整統計（不存 ckpt）。完整 26,335×2 個模型可用 `--scope full`，並建議搭配 `--shard i/N` 分片。
- **Method 4 只做 binary target + focal loss (γ=3)**。`m4_train.py` 仍接受 `--target {binary,count,logE}` 與 `--loss {focal,bce,balanced,mse}` 旗標方便擴充。
- 切分日期：Method 3 訓 89-110 / 測 111-112；Method 4 訓 89-108 / 測 109-112。

## 程式碼結構

```
src/cwa4/
  data/preprocessing.py     # 共用資料 helper
  data/method3.py           # Method3Dataset
  data/method4.py           # Method4Dataset + make_train_dev
  models/model_a.py         # PDF Table 1 架構（counts/binary 雙 head）
  models/classifier_m4.py   # Method 4 binary classifier (logits)
  losses.py                 # BalancedBCE、FocalLoss
  mingru.py / encoder.py / layernorm.py / conv_btc.py  # 既有元件
scripts/
  1_convert_gnss_format.py / 2_convert_pfile_format.py / 3_create_datasets.py
  m3_train_all.py / m3_collect_tables.py
  m4_train.py / m4_run_exp1.py / m4_run_exp2.py / m4_eval.py / m4_plot_data.py
tests/
  test_method4_pos_ratio.py / test_losses.py
```
