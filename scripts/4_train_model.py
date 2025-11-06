# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///
import argparse
import itertools

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from dataset import read_data, Dataset3
from model_a import ModelA
from torch.optim import AdamW

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--center", type=str, help="中心測站的名稱")
    parser.add_argument("-a", "--radius_a", type=float, default=20000.0, help="設定地震的範圍，半徑為 a")
    parser.add_argument("-b", "--radius_b", type=float, default=40000.0, help="設定測站的範圍, 半徑為 b")
    parser.add_argument("-n", "--n_comb", type=int, default=3, help="設定組合的上限")

    args = parser.parse_args()
    print(args)

    center = args.center
    batch_size = 128
    neighbors, gnss_df, pfile_df, hist = read_data(args.center, radius_a=args.radius_a, radius_b=args.radius_b)

    exp_dir: Path = Path("exp") / center
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    
    # 每個模型跑完整 5 個 epoch 再算測試集的結果
    for m in range(0, args.n_comb): # m 是組合的數量
        for neighbors_m in itertools.combinations(neighbors, m):
            # id of the model
            model_id = "_".join([center, *neighbors_m])
            step = 0
            # create dataset
            dataset = Dataset3(center, neighbors_m, gnss_df, hist)
            trn_set = Subset(dataset, list(range(0, len(dataset)-365*2)))
            tst_set = Subset(dataset, list(range(len(dataset)-365*2, len(dataset))))
            # reset seed
            torch.manual_seed(1713302033171)
            # create training data loader
            trn_dl = DataLoader(trn_set, batch_size, shuffle=True, num_workers=5)
            # create model
            model = ModelA(dataset.x_dim, d_ch=5, m_ch=10, h_ch=128).to(device)
            optimizer = AdamW(model.parameters(), lr=0.0001)
            for e in range(5): # 5 個 epoch
                for (x, y) in trn_dl:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    y_hat = model(x)[:, -1, :, :] # (B, T=-1, D, M)
                    loss = F.mse_loss(y_hat, y)
                    loss.backward()
                    optimizer.step()
                    print(f"{model_id}:{step} {loss.item()}")
            
            torch.save(model, exp_dir/f"model_{model_id}.pt")
            # evaluation
            with torch.inference_mode():
                sum_se = torch.zeros(5, 10, dtype=torch.double, device=device)
                cf_mat = torch.zeros(5, 10, 4, dtype=torch.long, device=device) # confusion
                tst_dl = DataLoader(tst_set, 10, shuffle=False, num_workers=5, drop_last=False)
                for (x, y) in tst_dl:
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)[:, -1, :, :] # (B, T=-1, D, M)
                    # sum square error
                    loss = (y - y_hat).pow(2).sum(dim=0) # (B, D, M) -> (D, M)
                    sum_se += loss.double()
                    # confusion matrix
                    z = (y >= 1) # (B, D, M)
                    z_hat = (y_hat >= 1) # (B, D, M)
                    cf_mat[:, :, 0] += (z & z_hat).sum(dim=0)    # TP
                    cf_mat[:, :, 1] += (z & ~z_hat).sum(dim=0)   # FN  
                    cf_mat[:, :, 2] += (~z & z_hat).sum(dim=0)   # FP
                    cf_mat[:, :, 3] += (~z & ~z_hat).sum(dim=0)  # TF
                # mse
                mean_se = sum_se / len(tst_set)
                print(f"MSELoss:\n{mean_se}")

                print(f"Confusion Matrix:\n{cf_mat}")
                # precision = TP/(TP+FP)
                precision = cf_mat[:, :, 0] / (cf_mat[:, :, 0] + cf_mat[:, :, 2])
                print(f"Precision:\n{precision}")
                # recall = TP/(TP+FN)
                recall = cf_mat[:, :, 0] / (cf_mat[:, :, 0] + cf_mat[:, :, 1])
                print(f"Recall:\n{recall}")
                # accuracy = (TP+TN) / (TP+TN+FP+FN)
                accuracy = (cf_mat[:, :, 0] + cf_mat[:, :, 3]) / cf_mat.sum(dim=-1)
                print(f"Accuracy:\n{accuracy}")

                # save the result
                result = {
                    "mse": mean_se,
                    "cf_mat": cf_mat,
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                }
                torch.save(result, exp_dir/f"result_{model_id}.pt")

                







    print(neighbors)


if __name__ == "__main__":
    main()