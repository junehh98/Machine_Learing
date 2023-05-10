<<<<<<< HEAD
﻿'''
문2) load_boston() 함수를 이용하여 보스턴 시 주택 가격 예측 회귀모델 생성 
  조건1> train/test - 7:3비울
  조건2> y 변수 : boston.target
  조건3> x 변수 : boston.data
  조건4> 모델 평가 : MSE, r2_score
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score 

import seaborn as sns
import numpy as np
import pandas as pd

# 1. data load
boston = fetch_openml(name='boston', version=1)
print(boston)


# 2. 변수 선택  
X = boston.data
y = boston.target


X.shape
y.shape


# 3. train/test split
X_train,X_test, y_train,y_test = train_test_split(X, y, 
                 test_size=0.3, random_state=123)



# 4. 회귀모델 생성 : train set
model = LinearRegression().fit(X=X_train, y=y_train) 

model.summary()
model.params















=======
﻿'''
문2) load_boston() 함수를 이용하여 보스턴 시 주택 가격 예측 회귀모델 생성 
  조건1> train/test - 7:3비울
  조건2> y 변수 : boston.target
  조건3> x 변수 : boston.data
  조건4> 모델 평가 : MSE, r2_score
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score 

import seaborn as sns
import numpy as np
import pandas as pd

# 1. data load
boston = fetch_openml(name='boston', version=1)
print(boston)


# 2. 변수 선택  
X = boston.data
y = boston.target


X.shape
y.shape


# 3. train/test split
X_train,X_test, y_train,y_test = train_test_split(X, y, 
                 test_size=0.3, random_state=123)



# 4. 회귀모델 생성 : train set
model = LinearRegression().fit(X=X_train, y=y_train) 

model.summary()
model.params















>>>>>>> 32f7d70641783dee9a7f41f16c0d9a0ed6467ceb
