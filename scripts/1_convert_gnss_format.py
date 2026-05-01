# 讀取給的資料，存成比較好讀取的格式
# Gamit_Globk -- GNSS 測站資料
#    - 只取 N, E, U 三個欄位。
# Pfile
#    - 時間改成台灣時間
#    - 取時間、座標、規模


import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from pyproj import Transformer

# 座標轉換工具，從經緯度轉成 TWD97 橫麥卡托。
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3826") 

data_dir = Path("data/Gamit_Globk/")

all_stations = []
all_df = []

for pos_file in data_dir.iterdir():
    if pos_file.suffix != '.pos':
        continue
    print(f"處理 {pos_file}")
    # 取得測站名稱
    with pos_file.open('r') as f:
        lines = []
        for i in range(33):
            lines.append(f.readline().rstrip())
        station_id = lines[2].split()[-1]
        first_epoch = datetime.strptime(lines[4].split(":")[1].strip(), '%Y%m%d %H%M%S')
        last_epoch = datetime.strptime(lines[5].split(":")[1].strip(), '%Y%m%d %H%M%S')
        xyz = list(map(float, lines[7].split()[4:7]))
        neu = list(map(float, lines[8].split()[4:7]))
        xy = transformer.transform(neu[0], neu[1])
        xyu = (xy[0], xy[1], neu[2])



    df = pd.read_fwf(pos_file, skiprows=36) # Read a table of fixed-width formatted lines
    date = pd.to_datetime(df["*YYYYMMDD"].astype(str)+"T115900", utc=True)
    df['dateime'] = date
    df = df.set_index(date)
    df.index.name='datetime'
    df = df.drop(["*YYYYMMDD", "HHMMSS", "JJJJJ.JJJJ"], axis=1)
    #col_index = pd.MultiIndex.from_tuples([(str(station_id),'N'), (str(station_id),'E'), (str(station_id),'U')], 
    #                                        names=('station', 'coordinate'))
    col_index = pd.MultiIndex.from_tuples([(str(station_id),'dX'), (str(station_id),'dY'), (str(station_id),'dU')], 
                                            names=('station', 'coordinate'))
    #df = df[['NLat', 'Elong', 'Height']].set_axis(col_index, axis=1)
    df = df[['dE', 'dN', 'dU']].set_axis(col_index, axis=1)
    
    all_stations.append((station_id, xyu, first_epoch, last_epoch))    
    all_df.append(df)

all_station_info = {
    "name": [],
    "X": [],
    "Y": [],
    "U": [],
    "first_epoch" : [],
    "last_epoch" : []
}

for name, xyu, first_epoch, last_epoch in all_stations:
    all_station_info['name'].append(name)
    x, y, u = xyu
    all_station_info['X'].append(x)
    all_station_info['Y'].append(y)
    all_station_info['U'].append(u)
    all_station_info['first_epoch'].append(first_epoch)
    all_station_info['last_epoch'].append(last_epoch)

pd.DataFrame(all_station_info).to_pickle("data/station_locations.pkl")

df = pd.concat(all_df, axis=1)

full_dates = pd.date_range("1994-01-01 11:59:00+00:00", "2023-12-31 11:59:00+00:00")

df = df.reindex(full_dates, fill_value=np.nan)
df.to_pickle("data/GNSS_XYU.pkl")





