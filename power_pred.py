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

features =['기온(C)','강수량(mm)','풍속(m/s)','습도(%)','연도','월','일','시']
target = "일조(hr)"

labeled_data = merged_df[merged_df[target].notnull()] 
unlabeled_data = merged_df[merged_df[target].isnull()]

X_labeled = labeled_data[features]
y_labeled = labeled_data[target]
X_unlabeled = unlabeled_data[features]


clf = TabNetRegressor()
clf.fit(
    X_labeled.values, y_labeled.values.reshape(-1,1),
    eval_set=[(X_labeled.values, y_labeled.values.reshape(-1,1))]
)

unlabeled_predictions = clf.predict(X_unlabeled.values)
merged_df.loc[merged_df[target].isnull(), target] = unlabeled_predictions

cleaned_merged_df1 = merged_df.copy()

features1 =['기온(C)','풍속(m/s)','강수량(mm)','습도(%)','연도','월','일','시', "일조(hr)"]
target1 = "일사(MJ/m2)"

labeled_data1 = cleaned_merged_df1[cleaned_merged_df1[target1].notnull()] 
unlabeled_data1 = cleaned_merged_df1[cleaned_merged_df1[target1].isnull()]

X_labeled1 = labeled_data1[features1]
y_labeled1 = labeled_data1[target1]
X_unlabeled1 = unlabeled_data1[features1]

clf1 = TabNetRegressor()
clf1.fit(
    X_labeled1.values, y_labeled1.values.reshape(-1,1),
    eval_set=[(X_labeled1.values, y_labeled1.values.reshape(-1,1))]
)

unlabeled_predictions1 = clf1.predict(X_unlabeled1.values)

cleaned_merged_df1.loc[cleaned_merged_df1[target1].isnull(), target1] = unlabeled_predictions1

cleaned_merged_df1['일사(MJ/m2)'] = cleaned_merged_df1['일사(MJ/m2)'].apply(lambda x: max(x, 0))

X = cleaned_merged_df1[['습도(%)','연도','월','일','시','연면적(m2)','건물유형_건물기타', '건물유형_공공', '건물유형_대학교', '건물유형_데이터센터', '건물유형_백화점및아울렛',
       '건물유형_병원', '건물유형_상용', '건물유형_아파트', '건물유형_연구소', '건물유형_지식산업센터',
       '건물유형_할인마트', '건물유형_호텔및리조트']]
y = cleaned_merged_df1[['전력소비량(kWh)']]

scaler1 = StandardScaler()

ranfor_add = RandomForestRegressor()
X = scaler1.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=5)

y_train_flat = y_train.values.reshape(-1)

ranfor_add.fit(X_train,y_train_flat)

ran_y_pred1 = ranfor_add.predict(X_test)

ran_mape1 =mean_absolute_percentage_error(y_test,ran_y_pred1)

print(ran_mape1)

with open('rf_model.pkl', 'wb') as file:
    pickle.dump(ranfor_add, file)
