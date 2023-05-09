# -*- coding: utf-8 -*-
"""
문4) california 주택가격을 대상으로 다음과 같은 단계별로 선형회귀분석을 수행하시오.
"""


from sklearn.datasets import fetch_california_housing # dataset load
from sklearn.linear_model import LinearRegression  # model
from sklearn.model_selection import train_test_split # dataset split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import zscore # 표준화(mu=0, st=1) 
import numpy as np # 로그변환 + 난수 

# 캘리포니아 주택 가격 dataset load 
california = fetch_california_housing()
print(california.DESCR)


# 단계1 : 특징변수(8개)와 타켓변수(MEDV) 선택  
X = california.data
y = california.target

X.shape # (20640, 8)
y.shape # (20640, )


# 단계2 : 데이터 스케일링 : X변수(표준화), y변수(로그변화)   
X_scale = zscore(X)  # standard scaler ??
y_log = np.log1p(y)


# 단계3 : 75%(train) vs 25(test) 비율 데이터셋 split : seed값 적용 
# 스케일링 X
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123)


# 스케일링 O
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_scale, y_log, test_size=0.25, random_state=123)




# 단계4 : 회귀모델 생성
model = LinearRegression().fit(X=X_train, y=y_train)
model1 = LinearRegression().fit(X=X_train1, y=y_train1)


# 단계5 : train과 test score 확인 : 스케일링 전과 후 score 확인  
# 스케일링 전
model_train_score = model.score(X_train, y_train) 
model_test_score = model.score(X_test, y_test) 
print('model train score =', model_train_score)
print('model test score =', model_test_score)



# 스케일링 후
model_train_score1 = model.score(X_train1, y_train1) 
model_test_score1 = model.score(X_test1, y_test1) 
print('model train score =', model_train_score1)
print('model test score =', model_test_score1)

  
