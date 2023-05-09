# -*- coding: utf-8 -*-
"""
문3) california 주택가격을 대상으로 다음과 같은 단계별로 선형회귀분석을 수행하시오.
"""

# california 주택가격 데이터셋 
'''
캘리포니아 주택 가격 데이터(회귀 분석용 예제 데이터)

•타겟 변수
1990년 캘리포니아의 각 행정 구역 내 주택 가격의 중앙값

•특징 변수(8) 
MedInc : 행정 구역 내 소득의 중앙값
HouseAge : 행정 구역 내 주택 연식의 중앙값
AveRooms : 평균 방 갯수
AveBedrms : 평균 침실 갯수
Population : 행정 구역 내 인구 수
AveOccup : 평균 자가 비율
Latitude : 해당 행정 구역의 위도
Longitude : 해당 행정 구역의 경도 : y변수 
'''

from sklearn.datasets import fetch_california_housing # dataset load
import pandas as pd # DataFrame 생성 
from sklearn.linear_model import LinearRegression  # model
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import mean_squared_error, r2_score # model 평가 
import matplotlib.pyplot as plt 

# 캘리포니아 주택 가격 dataset load 
california = fetch_california_housing()
print(california.DESCR)

X = california.data
type(X) # numpy.ndarray
X.shape # (20640, 8)


# 단계1 : 특징변수(8개)와 타켓변수(MEDV)를 이용하여 DataFrame 생성하기  
#  numpy -> DataFrame 
cal_df = pd.DataFrame(california.data, 
                      columns=california.feature_names)
cal_df["MEDV"] = california.target # y변수 추가 
print(cal_df.tail())
print(cal_df.info()) 
type(cal_df) # pandas.core.frame.DataFrame

cols = list(cal_df.columns)
cols

X = cal_df.iloc[:,:8]
y = cal_df.iloc[:,-1]

X.shape # (20640, 8)
y.shape # (20640,)


# 단계2 : 75%(train) vs 25%(val) 비율 데이터셋 split 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=123)


# 단계3 : 회귀모델 생성
lm = LinearRegression()
model = lm.fit(X = X_train, y = y_train)




# 단계4 : 모델 검정(evaluation)  : 과적합(overfitting) 확인  
model.score(X=X_train, y=y_train)
# 0.6055435032462828

model.score(X=X_val, y=y_val)
# 0.6078944086580228



# 단계5 : 모델 평가(test) : 평가방법 : MSE, r2_score - 50% 샘플링 자료 이용
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=123)


y_pred = model.predict(X = X_test)


mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
# 0.5233205828505497

score = r2_score(y_true = y_test, y_pred = y_pred)
# 0.6108012238625931



# 단계6 : 예측치 100개 vs 정답 100개 비교 시각화 
type(y_pred) # numpy.ndarray
type(y_test) # pandas.core.series.Series

y_pred.shape
y_test.shape


# Series 객체 -> list 객체 변환
y_true = y_test[:100].to_list()


plt.plot(y_pred[:100], 'r', label='y_pred')
plt.plot(y_true[:100], 'b', label='y_true')
plt.legend(loc='best')
plt.show()



































