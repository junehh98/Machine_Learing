# -*- coding: utf-8 -*-
"""
 범주형 변수를 X변수로 사용 - 가변수(dummy) 변환 
"""

import pandas as pd # csv file, 가변수 
from sklearn.model_selection import train_test_split # split 
from sklearn.linear_model import LinearRegression # model 
from sklearn.preprocessing import minmax_scale # 정규화(0~1)
from scipy.stats import zscore

# 1. csv file load 
path = r'C:\ITWILL\5_Python_ML\data'
insurance = pd.read_csv(path + '/insurance.csv')
insurance.info()



# 2. 불필요한 칼럼 제거 : region
new_df = insurance.drop(['region'], axis= 1)
new_df.info()
'''
 0   age      1338 non-null   int64  
 1   sex      1338 non-null   object   -> 가변수
 2   bmi      1338 non-null   float64
 3   children  1338 non-null   int64
 4   smoker   1338 non-null   object  -> 가변수
 5   charges  1338 non-null   float64 -> y변수 
'''



# 3. X, y변수 선택 
X = new_df.drop('charges', axis= 1)
X.shape #  (1338, 5)

y = new_df['charges']


# 4. 명목형(범주형) 변수 -> 가변수(dummy) 변환 : k-1개 
X.info()
X_dummy = pd.get_dummies(X, columns=['sex', 'smoker'],
               drop_first=True)

X_dummy.info()


# 5. 이상치 확인  
X_dummy.describe() 
# age : 최대값 이상치(100세이상)
# bmi : 최소값 이상치(0 미만)

# 이상치 제거 
X_new = X_dummy[(X_dummy.age < 100) & (X_dummy.bmi > 0)]
X_dummy.shape # (1338, 5)
X_new.shape # (1332, 5)


# x변수 표준화
X_new = zscore(X_new)
idx = X_dummy[(X_dummy.age < 100) & (X_dummy.bmi > 0)].index
len(idx) # 1332

# x_new와 y와 행의 개수가 맞지 않음 

# 제외된 관측치 : 6개 index
X_dummy[~((X_dummy.age < 100) & (X_dummy.bmi > 0))].index
# Int64Index([12, 16, 48, 82, 114, 180]) -> 지워진 6개 관측치의 index 

y_new = y[idx]
len(y_new) # 1332


# 6. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
   X_new, y_new, test_size=0.3, random_state=123)


# 7. model 생성 & 평가 
model = LinearRegression().fit(X=X_train, y=y_train)

model.score(X=X_train, y=y_train)
model.score(X=X_test, y=y_test)

'''
이상치 처리 전 
train : 0.6837522970202745
test  : 0.7236336310934985

이상치 처리 후
train : 0.7425695212639727
test  : 0.7612276881341357

스케일링 후 
train : 0.7425695212639727
test  : 0.7612276881341358
'''








