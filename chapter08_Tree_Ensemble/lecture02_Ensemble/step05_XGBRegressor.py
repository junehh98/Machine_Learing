# -*- coding: utf-8 -*-
"""
 - XGBoost 회귀트리 예
"""
import pandas as pd
from xgboost import XGBRegressor # 회귀트리 모델 
from xgboost import plot_importance # 중요변수 시각화 


from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 
from sklearn.datasets import fetch_openml


# 1. Dataset load
boston_df = pd.read_csv(r'C:\ITWILL\5_Python_ML\data\BostonHousing.csv')

# Print the shape of the DataFrame
print(boston_df.shape)

# Check the column names of the DataFrame
print(boston_df.columns)

# Assign the correct target variable column name
target_column = 'CAT. MEDV'

X = boston_df.drop(target_column, axis=1).values
y = boston_df[target_column].values
X.shape  # (506, 13)



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


# 5. model 평가
y_pred = model.predict(X_test)
y_pred

mse = mean_squared_error(y_true = y_test, y_pred = X_test)
print('mse :', mse)


score = r2_score(y_true = y_test, y_pred = X_test)
print('score :', score)


