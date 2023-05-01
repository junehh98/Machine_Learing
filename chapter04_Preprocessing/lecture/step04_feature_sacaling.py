#########################################
### 4. 피처 스케일링(feature scaling) 
#########################################

"""
피처 스케일링 : 서로 다른 크기(단위)를 갖는 X변수(feature)를 대상으로 일정한 범위로 조정하는 전처리 작업 
 - 방법 : 표준화, 최소-최대 정규화, 로그변환    
 
 1. 표준화 : X변수를 대상으로 정규분포가 될 수 있도록 평균=0, 표준편차=1로 통일 시킴 
   -> 회귀모델, SVM 계열은 X변수가 정규분포라고 가정하에 학습이 진행되므로 표준화를 적용   
 2. 최소-최대 정규화 : 서로 다른 척도(값의 범위)를 갖는 X변수를 대상으로 최솟값=0, 최댓값=1로 통일 시킴 
   -> 트리모델 계열(회귀모델 계열이 아닌 경우)에서 서로 다른 척도를 갖는 경우 적용 
 3. 로그변환 : log()함수 이용하여 로그변환   
   -> 비선형(곡선) -> 선형(직선)으로 변환
   -> 왜곡을 갖는 분포 -> 좌우대칭의 정규분포로 변환   
   -> 회귀모델에서 Y변수 적용(X변수를 표준화 또는 정규화할 경우 Y변수는 로그변환) 
"""

# 1. 함수형 스케일링 도구  
from sklearn.preprocessing import scale # 표준화 
from sklearn.preprocessing import minmax_scale # 정규화
import numpy as np # 로그변환 + 난수

# 실습 data 생성 : 난수 정수 생성  
np.random.seed(12) # 시드값 
X = np.random.randint(-10, 100, (5, 4)) # -10~100


# 1) 표준화 
X_zscore = scale(X)



# 2) 정규화 
X_nor = minmax_scale(X) # 0~1 조정 


# 3) 로그변환 
X_log = np.log(X) # RuntimeWarning 발생


# 2. 클래스형 스케일링 도구 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

import pandas as pd
iris = pd.read_csv(r"C:\ITWILL\5_Python_ML\data\iris.csv")
iris.info()

# 1) DataFrame 표준화 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X=iris.drop('Species', axis=1))# Species 칼럼 제외 


# 2) DataFrame 정규화 
scaler2 = MinMaxScaler()
X_scaled2 = scaler2.fit_transform(X = iris.drop('Species', axis = 1))

