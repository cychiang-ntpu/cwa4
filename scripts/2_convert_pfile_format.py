import pandas as pd
from pathlib import Path
from pyproj import Transformer

# 要確定這邊經緯度是用
# - WGS84經緯度（全球性資料，如：GPS） ＝> EPSG:4326
# - TWD97經緯度（國土測繪中心發佈全國性資料）＝> EPSG:3824
transformer = Transformer.from_crs("EPSG:3824", "EPSG:3826", always_xy=True) 
# 設定
pfile_dir = Path("data/Pfile")
dataframe_pkl_file = Path("data/Pfile.pkl")


# feature name
column_names = [
    "年","月","日","時","分","秒","緯度","緯分","經度","經分",
    "深度","規模","測站數","最小震央距離","最大方位角間隙","時間均方差","水平誤差", "垂直誤差", "深度收斂方式",
    "相位數","品質", "關聯檔案名稱"
]

colspecs = [
    (0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 18), (18, 20), (20, 25), (25, 28), (28, 33),
    (33, 39), (39, 43), (43, 45), (45, 50), (50, 53), (53, 57), (57, 61), (61, 65), (66, 67),
    (67, 70), (70, 71), (72,84)
]

df_eqs = []
for dat_file in sorted(pfile_dir.glob("*")):
    if not dat_file.suffix.endswith(("dat", "DAT")):
        continue
    print(f"處理 {dat_file}")
    df = pd.read_fwf(dat_file, colspecs=colspecs, names=column_names) 
    df_eqs.append(df)
    

df_eqs = pd.concat(df_eqs)

# 去掉不要的 column
df_eqs = df_eqs.drop(["測站數","最小震央距離","最大方位角間隙","時間均方差","水平誤差", "垂直誤差", "深度收斂方式",
    "相位數","品質", "關聯檔案名稱"], axis=1)

print("調整經緯度格式")

# 調整經緯度
df_eqs['緯度'] = df_eqs['緯度'] + df_eqs['緯分'] / 60.0
df_eqs['經度'] = df_eqs['經度'] + df_eqs['經分'] / 60.0
df_eqs = df_eqs.drop(['緯分', '經分'], axis=1)

# 經緯度轉成 
x, y = transformer.transform(df_eqs['經度'], df_eqs['緯度'])

df_eqs['X'] = x
df_eqs['Y'] = y

print("加入 datetime")
df_eqs['datetime'] = pd.to_datetime(
    df_eqs[['年', '月', '日', '時', '分', '秒']]
        .set_axis(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1),
    utc=True
)

df_eqs = df_eqs.drop(['年', '月', '日', '時', '分', '秒'], axis=1)
df_eqs = df_eqs.set_index("datetime").sort_index()

print("儲存 dataframe")
df_eqs.to_pickle(dataframe_pkl_file)