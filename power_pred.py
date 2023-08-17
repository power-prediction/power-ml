import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from pytorch_tabnet.tab_model import TabNetRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

power = pd.read_csv("C:/Users/eagls/bootcamp/dacon_electric/train.csv")
building = pd.read_csv("C:/Users/eagls/bootcamp/dacon_electric/building_info.csv")

power_df = pd.DataFrame(power)
building_df = pd.DataFrame(building)

power_df["연도"] = 2022

def month(a):
    t = a[4:6]
    return int(t)

power_df["월"]=power_df["일시"].apply(month)

def day(a):
    t = a[6:8]
    return int(t)
power_df["일"]=power_df["일시"].apply(day)

def hour(a):
    t = a[-2:]
    return int(t)
power_df["시"]=power_df["일시"].apply(hour)
power_df['강수량(mm)'].fillna(0,inplace=True)

merged_df = power_df.merge(building_df, on='건물번호')

merged_df.dropna(subset=['풍속(m/s)'], inplace=True)
merged_df.dropna(subset=['습도(%)'], inplace=True)

merged_df=pd.get_dummies(merged_df, columns=['건물유형'])

x = merged_df[['기온(C)','강수량(mm)','습도(%)','연도','월','일','시','연면적(m2)','건물유형_건물기타',
       '건물유형_공공', '건물유형_대학교', '건물유형_데이터센터', '건물유형_백화점및아울렛', '건물유형_병원',
       '건물유형_상용', '건물유형_아파트', '건물유형_연구소', '건물유형_지식산업센터', '건물유형_할인마트',
       '건물유형_호텔및리조트']]
y = merged_df[['전력소비량(kWh)']]


scaler = StandardScaler()

ranfor = RandomForestRegressor()
X = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=8)

y_train_flat = y_train.values.reshape(-1)

ranfor.fit(X_train,y_train_flat)

ran_y_pred = ranfor.predict(X_test)

ran_mape =mean_absolute_percentage_error(y_test,ran_y_pred)

print(ran_mape)

