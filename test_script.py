# -*- coding: utf-8 -*-
"""
문) cars 데이터셋을 대상으로 결측치, 이상치, 변수 제거, 스케일링, 인코딩
    등의 전처리 방식을 적용하여 과적합을 고려한 가장 최적의 성능을 
    제공하는 회귀 model을 생성하시오.    
    
    훈련셋(train) : 80%
    평가셋(test) : 20%
"""

import pandas as pd # csv file load 
import numpy as np
from sklearn.linear_model import LinearRegression #선형회귀분석
from sklearn.model_selection import train_test_split # 데이터 split
from sklearn.metrics import mean_squared_error, r2_score 



## 단계1 : 데이터 준비 
cars = pd.read_csv(r'C:\ITWILL\5_Python_ML\data\cars.csv')
cars.info()
'''
RangeIndex: 40 entries, 0 to 38
Data columns (total 11 columns):    
 0   mpg     40 non-null     float64 : 연비   --> 종속변수로 설정 
 
 1   cyl     40 non-null     int64   : 실린더수 
 2   disp    40 non-null     float64 : 엔진크기
 3   hp      40 non-null     int64   : 출력 
 4   drat    40 non-null     float64 : 뒷바퀴축 기어 비율 -> 제거
 5   wt      40 non-null     float64 : 중량 
 6   qsec    40 non-null     float64 : 1/4 소요시간 -> 제거
  # 정지 상태에서 1/4마일 거리를 이동하는 데 걸리는 시간
 7   vs      40 non-null     object  : 엔진모양(V' or Straight) -> 제거
 8   am      40 non-null     int64   : 변속기   -
 9   gear    40 non-null     object  : 기어 -> 기어가 높을수록 연비가 안좋을 것으로 예상
 10  carb    40 non-null     int64   : 카뷰레터 
 11  car_id  40 non-null     int64   : 자동차 구분번호  -> 제거
'''
cars.shape #  (40, 12)


# 1) 결측치, 이상치, 변수 제거
cars.isnull().sum()
'''
mpg       0
cyl       0
disp      0
hp        0
drat      0
wt        0
qsec      0
vs        0
am        0
gear      0
carb      0
car_id    0
dtype: int64  -> 결측치 없음 확인
'''

# 변수 제거
new_cars = cars.drop(['drat', 'qsec', 'car_id', 'vs'], axis=1)
new_cars

new_cars.describe()
new_cars.columns

# gear가 f인 행 삭제하고 ,object형을 int 형으로 변경 
new_cars = new_cars[new_cars['gear'] != 'f']
new_cars.shape # (38, 8)


new_cars['gear'] = new_cars['gear'].astype('int64')
new_cars.info() # 변경된것 확인 

# 상관관계 보기 
corr = new_cars.corr()




# 독립, 종속변수 설정 
X = new_cars.drop('mpg', axis=1)

y = new_cars.mpg


########################### 스케일링 X########################################
X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size = 30, random_state=1)

# 모델 생성, 학습
lm = LinearRegression() 
model = lm.fit(X=X_train, y=y_train)

# 예측
y_pred = model.predict(X_test)


# 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R-squared:', r2)
'''
Mean squared error: 7716.637054787997
R-squared: -219.5946756375258
'''





#################################### 스케일링 ################################
# scaling -> 변수마다 단위가 다르기 때문에 변경 
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scale = scaler.fit_transform(X)
y_scale = np.log1p(y)


X_train,X_test,y_train,y_test = train_test_split(
    X_scale, y_scale, test_size = 30, random_state=1998 )

# 모델 생성, 학습
lm = LinearRegression() 
model = lm.fit(X=X_train, y=y_train)



# 6. model 생성하기
model = LinearRegression().fit(X=X_train, y=y_train)  


# 4. model 평가하기
train_score = model.score(X_train, y_train) 
test_score = model.score(X_test, y_test) 
print('train score :', train_score)
print('test score :', test_score)
















