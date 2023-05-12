# -*- coding: utf-8 -*-
"""
 - XGBoost 회귀트리 예
"""

from xgboost import XGBRegressor # 회귀트리 모델 
from xgboost import plot_importance # 중요변수 시각화 

from sklearn.datasets import load_boston # dataset
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 


# 1. dataset load
boston = load_boston()

x_names = boston.feature_names
x_names


X = boston.data
y = boston.target 
X.shape 


#  2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# 3. model 생성 
model = XGBRegressor().fit(X = X_train, y = y_train)
print(model)


# 4. 중요변수 확인 
fscore = model.get_booster().get_fscore()
print(fscore)


# 중요변수 시각화 
plot_importance(model, max_num_features=13) # f0 ~ f12 










