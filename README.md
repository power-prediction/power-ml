# ml-power-prediction
![image](https://github.com/power-prediction/ml-power-prediction/assets/57518426/3f005844-12c1-4c34-9203-6d63ed3b64bf)

## plan
- 데이터 형식
    - train.csv [204000 rows x 10 columns]
        - num_date_time
        - 건물번호 
        - 일시
        - 기온(C)
        - 강수량(mm)
        - 풍속(m/s)
        - 습도(%)
        - 일조(hr)
        - 일사(MJ/m2)
        - 전력소비량(kWh)     
    - building_info.csv [100 rows x 7 columns]
        - 건물번호 
        - 건물유형 
        - 연면적(m2)
        - 냉방면적(m2)
        - 태양광용량(kW)
        - ESS저장용량(kWh)
        - PCS용량(kW)

- Machine Learning
    - 건물 정보(building_info.csv)를 train.csv 데이터의 건물 번호 컬럼에 맞춰 merge
    - 일시 정보를 월/일/시 컬럼으로 분리
    - 1차
        - 결측치가 많은 컬럼 드롭 후 모델 학습
    - 2차
        - Self-training 학습 방식을 통해 결측치 처리 후 모델 학습
    - 평가지표 : Mean Absolute Percentage Error
        - MAPE가 가장 낮은 모델을 피클로 저장


 



