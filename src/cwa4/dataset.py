import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path


def read_data(
        center,
        gnss_path="data/GNSS_XYU.pkl", 
        pfile_path="data/Pfile.pkl", 
        stations_path="data/station_locations.pkl", 
        radius_a=20000.0, radius_b=40000.0,
        t0 = "2000-01-01", t1="2024-01-01"
):  
    # 讀取處理好的資料
    gnss_df = pd.read_pickle(gnss_path)
    pfile_df = pd.read_pickle(pfile_path)
    stations_df = pd.read_pickle(stations_path)
    # 取得中心位置
    _name, x, y, _u, _start_time, _end_time = stations_df[ stations_df['name'] == center].iloc[0]
    # 過濾已停用的測站
    stations_df = stations_df[ stations_df['last_epoch'] == '2023-12-31 11:59:00']
    # 取得鄰近測站
    distance = np.hypot(stations_df['X'] - x, stations_df['Y'] - y)
    neighbors = stations_df[ (stations_df['name'] != center) &  (distance <= radius_b )]
    neighbors = neighbors['name'].to_list()
    
    # 時間處理
    t0 = pd.to_datetime(t0).tz_localize("Asia/Taipei")
    t1 = pd.to_datetime(t1).tz_localize("Asia/Taipei")

    # 過濾 gnss 資料，轉換時區
    full_dates = pd.date_range("1994-01-01 11:59:00+00:00", "2023-12-31 11:59:00+00:00")
    gnss_df = gnss_df.reindex(full_dates, fill_value=np.NaN)
    gnss_df.index = gnss_df.index.tz_convert("Asia/Taipei")
    gnss_df = gnss_df[t0: t1]
    gnss_df = gnss_df.fillna(0.0)
    # 過濾地震資料，轉換時區
    distance = np.hypot(pfile_df['X'] - x, pfile_df['Y'] - y)
    pfile_df = pfile_df[distance <= radius_a]
    pfile_df.index = pfile_df.index.tz_convert("Asia/Taipei")
    pfile_df = pfile_df[t0: t1]
    # 按照深度、規模統計每一天的地震次數
    group = pfile_df[['深度', '規模']].groupby(pd.Grouper(freq='1d', origin=t0))
    hist = np.zeros((gnss_df.shape[0] , 5, 10))
    for i, (_date, data) in enumerate(group):
        for _, row in data.iterrows():
            depth, magnitude = row.iloc[0], row.iloc[1]
            if depth < 5:
                d = 0
            elif depth < 10:
                d = 1
            elif depth < 30:
                d = 2  # 極淺層
            elif depth < 70:
                d = 3  # 淺層
            else: #elif depth < 300:
                d = 4  # 中層
            m = int(magnitude)
            hist[i, d, m] += 1
    return neighbors, gnss_df, pfile_df, hist


class Dataset3(torch.utils.data.Dataset):
    def __init__(
            self, 
            center,
            neighbors,
            gnss_df,
            hist
        ):
        super().__init__()
        self.center = center,
        self.neighbors = neighbors
        self.gnss_df = gnss_df
        self.hist = hist
        # 拿前一整年長度的資料 365 天 來預估下一天有沒有 4 級以上的淺層地震
        assert(gnss_df.shape[0] == hist.shape[0])
        T = hist.shape[0]
        # 過濾測站
        gnss_data = torch.from_numpy(gnss_df[neighbors].values) # (T, K, 3)
        hist_data = torch.from_numpy(hist)                      # (T, 5, 10)

        # 組合成輸入
        self.x = torch.cat((gnss_data.view(T, -1), hist_data.view(T, -1)), dim=1)
        self.y = hist_data
        self.length = T - 1 - 365
        self.x_dim = self.x.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0:
            index = self.length + index
        return self.x[index: index+365], self.y[index+365]
    
class Dataset4(torch.utils.data.Dataset):
    def __init__(
            self, 
            gnss_path="data/hualian_daily_gnss_dXdYdU.pkl", 
            statistics_path="data/hulian_daily_stataistics.pkl", 
            target_path="hualian_target_cnt.pkl", 
            input_width=730, 
            target_width=1,
            subset="trn"):
        super().__init__()
        df0 = pd.read_pickle(gnss_path)
        df1 = pd.read_pickle(statistics_path)
        df2 = pd.read_pickle(target_path)

        # input/output 的處理
        input = pd.concat((df0, df1), axis=1)
        target = pd.DataFrame(sliding_window_view(np.pad(df2, (0,target_width)), target_width).sum(axis=-1)[1:], index=df2.index)


        input = torch.from_numpy(input.to_numpy()) # (T, C)
        target = torch.from_numpy(target.to_numpy())

        # 先切開
        if subset == "trn":
            input = input[:7305]
            target = target[:7305]
        else:
            input = input[7305:]
            target = target[7305:]

        input = input.unfold(dimension=0, size=input_width, step=1).transpose(1, 2) # (T-L+1, C, L)
        self.input = input.float()

        # target 的處理
        target = (target[input_width-1:] > 0).float()
        self.target = target
    
    @property
    def input_dim(self):
        return self.input.shape[-1]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        return self.input[index], self.target[index]












    
