# -*- coding: utf-8 -*-
"""
문) cars 데이터셋을 대상으로 결측치, 이상치, 변수 제거, 스케일링, 인코딩
    등의 전처리 방식을 적용하여 과적합을 고려한 가장 최적의 성능을 
    제공하는 회귀 model을 생성하시오.    
    
    훈련셋(train) : 80%
    평가셋(test) : 20%
    
    평가셋 기준으로 모델 성능 평가 : r2 score or MSE 
"""
import numpy as np 
import pandas as pd # csv file load 
from sklearn.linear_model import LinearRegression #선형회귀분석
from sklearn.model_selection import train_test_split # 데이터 split
from sklearn.preprocessing import StandardScaler # 표준화 

## 단계1 : 데이터 준비 
cars = pd.read_csv(r'C:\ITWILL\5_Python_ML\data\cars.csv')
#cars.info()
'''
RangeIndex: 40 entries, 0 to 38
Data columns (total 11 columns):    
 0   mpg     40 non-null     float64 : 연비
 1   cyl     40 non-null     int64   : 실린더수 
 2   disp    40 non-null     float64 : 엔진크기
 3   hp      40 non-null     int64   : 출력 
 4   drat    40 non-null     float64 : 뒷바퀴축 기어 비율
 5   wt      40 non-null     float64 : 중량 
 6   qsec    40 non-null     float64 : 1/4 소요시간
 7   vs      40 non-null     object  : 엔진모양(V' or Straight)
 8   am      40 non-null     int64   : 변속기 
 9   gear    40 non-null     object  : 기어
 10  carb    40 non-null     int64   : 카뷰레터 
 11  car_id  40 non-null     int64   : 자동차 구분번호  
'''

# 1. 변수 탐색 
# 1) 숫자형 변수 스케일링 여부 확인 
#cars.mean(axis=0)
'''
mpg        20.007500
cyl         5.975000
disp      225.042500
hp        142.575000
drat        3.567000
wt          3.244425
qsec       18.113250
am          0.350000
carb        2.725000
car_id    119.500000
'''

# 2) object형 변수 확인 
cars.vs.unique() # ['V', 'S'] : 더미변수 사용 
cars.gear.unique() # ['4', '3', 'f', '5'] # 이산변수 사용 

# 3) 구분자 칼럼 : car_id 제외 
new_cars = cars.drop('car_id', axis = 1)
new_cars.shape # (40, 11)


# 2. 자료형 변환 : 문자형 -> 숫자형
new_cars.gear = new_cars.gear.replace('f', np.nan)
new_cars.isnull().sum() # gear    2

new_cars.dropna(subset=['gear'], axis = 0, inplace=True)
new_cars.shape # (38, 11)

# 정수형 변환 
new_cars.gear = new_cars.gear.astype('int')
#new_cars.info()


# vs 더미변수(k-1개)
new_cars_dummy = pd.get_dummies(data=new_cars, columns=['vs'], 
               drop_first=True)

#new_cars_dummy.info()
new_cars_dummy.shape # (38, 11)

new_cars_dummy.drop('vs_V', axis = 1, inplace=True) 

# 상관계수 
new_cars_dummy.corr()['mpg']
'''
cyl    -0.698238
disp   -0.827004
hp     -0.772298
drat    0.669016
wt     -0.863502
qsec    0.426284
am      0.586822
gear    0.476334
carb   -0.571019
'''

# 상관계수 낮은 변수 제거 
new_cars_dummy.drop(['qsec','gear'], axis = 1, inplace=True) 

cols = list(new_cars_dummy.columns)
cols # 11개 변수명 


# 3. X,y변수 선택
X = new_cars_dummy[cols[1:]]
y = new_cars_dummy[cols[0]]
X.shape # (38, 10)
y.shape # (38,)

# 4. X변수 스케일링 
X_scaled = StandardScaler().fit_transform(X)


# 5. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=34)


# 6. model 생성하기
model = LinearRegression().fit(X=X_train, y=y_train)  


# 4. model 평가하기
train_score = model.score(X_train, y_train) 
test_score = model.score(X_test, y_test) 
print('train score :', train_score)
print('test score :', test_score)
'''
train score : 0.8502729395823185
test score : 0.8425493136054849
'''

'''
test score 기준 
0.5875433678606834 : 11개 변수 
0.6557191822275628 : 스케일링 후 
0.6870985651429969 : 10개 변수(더미변수 제거) 
0.6990214610569532 : 8개 변수(상관계수 낮은 변수 2개 제거) 
0.8425493136054849 : seed값 조정(훈련세 표본 수정)    
'''

