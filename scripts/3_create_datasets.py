import pandas as pd
import geopandas as gpd
import numpy as np
import json
from pathlib import Path
#from pyproj import Transformer

# 設定
area_file = Path("data/main_island/area.shp")
pfile_file = Path("data/Pfile.pkl")
gnss_file = Path("data/GNSS_XYU.pkl")
station_loc_file = Path("data/station_locations.pkl")
alive_stations_file = Path("data/各縣市存活測站.json")

time_start = pd.Timestamp('2000-01-01', tz="Asia/Taipei") # 台灣時間
time_end = pd.Timestamp('2024-01-01', tz="Asia/Taipei")
#transformer = Transformer.from_crs("EPSG:3826", "EPSG:3824", always_xy=True)
full_dates = pd.date_range(time_start, time_end)

# 讀取 datasets
pfile = pd.read_pickle(pfile_file)
gnss = pd.read_pickle(gnss_file)
alive_stations = json.load(alive_stations_file.open("r"))
station_loc = pd.read_pickle(station_loc_file)


# 過濾出花蓮的測站
in_hualian = station_loc['name'].isin(alive_stations['花蓮縣'])
station_loc_hualian = station_loc[in_hualian]

# 過濾花蓮的地震事件
in_hualian = False
for i, row in station_loc_hualian.iterrows():
    inside_i = np.hypot(pfile['X'] - row['X'], pfile['Y'] - row['Y']) # 20km
    in_hualian = in_hualian | (inside_i < 20000.0)

pfile_hualian = pfile[in_hualian]
pfile_hualian.loc[:, '能量'] = np.power(10.0, pfile_hualian['規模'] * 1.5 + 11.8)

pfile_hualian.index = pfile_hualian.index.tz_convert("Asia/Taipei")

grouper = pd.Grouper(freq='1d', origin=time_start)

target_cnt = pfile_hualian[(pfile_hualian['規模'] > 5.5) & (pfile_hualian['深度'] < 30.0)].groupby(grouper).size()

target_cnt = target_cnt.reindex(full_dates, fill_value=0)
target_cnt.to_pickle("hualian_target_cnt.pkl")

# 分成 極淺、淺、中層 能量釋放 + 次數 
# 花蓮沒有深層的地震，所以不計
scales = []
counts = []
grouper = pd.Grouper(freq='1d', origin="2000-01-01 00:00:00.000000+08:00")
for d in [ (0, 30), (30, 70), (70, 300)]:
    target_d = pfile_hualian[pfile_hualian['深度'].between(d[0], d[1], inclusive='left')]
    group = target_d.groupby(by=grouper)
    count_d = group.size().reindex(full_dates).fillna(0)
    scale_d = (( np.log10( group['能量'].sum().clip(lower=1e-2))  - 11.8 ) / 1.5).reindex(full_dates).fillna(-9.2)
    counts.append(count_d)
    scales.append(scale_d)

statistics = pd.DataFrame({
    "極淺層-能量": scales[0],
    "淺層-能量": scales[1],
    "中層-能量": scales[2],
    "極淺層-次數": counts[0],
    "淺層-次數": counts[1],
    "中層-次數": counts[2],
})

statistics = statistics.reindex(full_dates)
statistics.to_pickle("hualian_daily_statistics.pkl")

# GNSS 資料
gnss.index = gnss.index.tz_convert("Asia/Taipei").normalize()
gnss = gnss[alive_stations['花蓮縣']].reindex(full_dates)
gnss = gnss.fillna(0.0)
gnss.to_pickle("hualian_daily_gnss_dXdYdU.pkl")