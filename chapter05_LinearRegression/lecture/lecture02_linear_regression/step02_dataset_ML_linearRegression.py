<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
step01_datasets_linearRegression.py

sklearn 패키지 
 - python 기계학습 관련 도구 제공 
"""

import sklearn
print(sklearn.__all__) # 패키지의 하위 모듈 확인 

from sklearn.datasets import load_diabetes # dataset 
from sklearn.linear_model import LinearRegression # model
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score  
import numpy as np

# 1. dataset load 
diabetes = load_diabetes() # 객체 반환 : X변수, y변수, 관련자료 
X = diabetes.data 
y = diabetes.target
X, y = load_diabetes(return_X_y = True) # X변수, y변수 반환 


# 2. X, y변수 특징 
X.mean() # -1.6638274468590581e-16
X.min() # -0.137767225690012
X.max() # 0.198787989657293


# y변수 
y.mean() # 152.13348416289594
y.min() # 25.0
y.max() # 346.0


# 3. train_test_split : 70% vs 30%
X_train,X_test,y_train,y_test = train_test_split(X, y, 
                                    test_size=0.3, 
                                    random_state=123) 

# data split check
X_train.shape # (309, 10)
X_test.shape # (133, 10)

y_train.shape # (309,)
y_test.shape # (133,)




# 4. model 생성 
lm = LinearRegression() # object
model = lm.fit(X=X_train, y=y_train) # train데이터 학습 


# 5. model 평가  
y_pred = model.predict(X=X_test)  
y_true = y_test 


# 1) 평균제곱오차(MSE) :y의 정규화, 0의 수렴정도
MSE = mean_squared_error(y_true, y_pred)
print('MSE =', MSE)
# MSE = 2926.8196257936324

err = y_true - y_pred
err_squared = np.square(err)
err_squared.mean() # 0.11633863200224709 -> 오차 제곱 평균 


# 2) 결정계수(R-제곱) : y의 정규화(x)의 수렴정도 
score = r2_score(y_true, y_pred)
print('r2 score =', score) 
# r2 score = 0.5078253552814805


















=======
# -*- coding: utf-8 -*-
"""
step01_datasets_linearRegression.py

sklearn 패키지 
 - python 기계학습 관련 도구 제공 
"""

import sklearn
print(sklearn.__all__) # 패키지의 하위 모듈 확인 

from sklearn.datasets import load_diabetes # dataset 
from sklearn.linear_model import LinearRegression # model
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score  
import numpy as np

# 1. dataset load 
diabetes = load_diabetes() # 객체 반환 : X변수, y변수, 관련자료 
X = diabetes.data 
y = diabetes.target
X, y = load_diabetes(return_X_y = True) # X변수, y변수 반환 


# 2. X, y변수 특징 
X.mean() # -1.6638274468590581e-16
X.min() # -0.137767225690012
X.max() # 0.198787989657293


# y변수 
y.mean() # 152.13348416289594
y.min() # 25.0
y.max() # 346.0


# 3. train_test_split : 70% vs 30%
X_train,X_test,y_train,y_test = train_test_split(X, y, 
                                    test_size=0.3, 
                                    random_state=123) 

# data split check
X_train.shape # (309, 10)
X_test.shape # (133, 10)

y_train.shape # (309,)
y_test.shape # (133,)




# 4. model 생성 
lm = LinearRegression() # object
model = lm.fit(X=X_train, y=y_train) # train데이터 학습 


# 5. model 평가  
y_pred = model.predict(X=X_test)  
y_true = y_test 


# 1) 평균제곱오차(MSE) :y의 정규화, 0의 수렴정도
MSE = mean_squared_error(y_true, y_pred)
print('MSE =', MSE)
# MSE = 2926.8196257936324

err = y_true - y_pred
err_squared = np.square(err)
err_squared.mean() # 0.11633863200224709 -> 오차 제곱 평균 


# 2) 결정계수(R-제곱) : y의 정규화(x)의 수렴정도 
score = r2_score(y_true, y_pred)
print('r2 score =', score) 
# r2 score = 0.5078253552814805


















>>>>>>> 32f7d70641783dee9a7f41f16c0d9a0ed6467ceb
