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
import pandas as pd # matrix -> DataFrame 
from sklearn.preprocessing import LabelEncoder



# 실습 data 생성 : 난수 정수 생성  
np.random.seed(12) # 시드값 
X = np.random.randint(-10, 100, (5, 4)) # -10~100
'''
array([[ 65,  17,  -4,  -8],  --> 5행 4열의 난수정수
       [ -7,  57,  66,  38],
       [ 12,  39,  42,  -5],
       [  3,  79,  24,  65],
       [ 64, -10,  94,  66]])
'''

X.shape # (5, 4)
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4'])
df


# 1) 표준화  
X_zscore = pd.DataFrame(scale(df, axis=0),
                        columns=['x1', 'x2', 'x3', 'x4']) # mean = 0, st = 1로 표준화 
'''
         x1        x2        x3        x4
0  1.217447 -0.627756 -1.434592 -1.210107
1 -1.113835  0.666586  0.640231  0.209917
2 -0.498635  0.084132 -0.071137 -1.117497
3 -0.790045  1.378475 -0.604663  1.043409
4  1.185068 -1.501437  1.470161  1.074279
'''
X_bar = X.mean() # 모평균
sigma = X.std() # 모표준편차 
(X - X_bar) / sigma # 약간 차이가 있지만 X_zscore와 비슷한 값을 가짐 

X_zscore.describe()


# 2) 정규화 
X_nor = pd.DataFrame(minmax_scale(df, axis=0),
                     columns=['x1', 'x2', 'x3', 'x4']) # 0~1 조정 
'''
X - min(X) / max(X) - min(X)
'''


# 3) 로그변환 
X_log = np.log(X) # RuntimeWarning 발생
# 입력자료에 음수나 0이 포함되어 있을때 RuntimeWarning 발생 
np.log(-1) # nan
np.log(0)  # -inf
np.log(0+1) # 0.0

np.log1p(0)
np.log1p(abs(-1))

X_log = np.log1p(abs(X))
X_log
'''
array([[4.18965474, 2.89037176, 1.60943791, 2.19722458],
       [2.07944154, 4.06044301, 4.20469262, 3.66356165],
       [2.56494936, 3.68887945, 3.76120012, 1.79175947],
       [1.38629436, 4.38202663, 3.21887582, 4.18965474],
       [4.17438727, 2.39789527, 4.55387689, 4.20469262]])
'''




# 2. 클래스형 스케일링 도구 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

import pandas as pd
iris = pd.read_csv(r"C:\ITWILL\5_Python_ML\data\iris.csv")
iris.info()

# object -> 10진수로 인코딩 
y = LabelEncoder().fit_transform(y=iris['Species'])


# 1) DataFrame 표준화 : 회귀계열모델(X변수:표준화, y변수:인코딩)
scaler = StandardScaler()

# DF -> numpy array
X_scaled = scaler.fit_transform(X=iris.drop('Species', axis=1))# Species 칼럼 제외 
X_scaled.shape # (150, 4)
type(X_scaled) # numpy.ndarray


# new_df = X_scaled + y
new_df = pd.DataFrame(X_scaled, columns=list(iris.columns[:4]))
new_df.info()
new_df['y'] = y



# 2) DataFrame 정규화 : 트리계열모델(X변수:정규화, y변수:레이블인코딩)
scaler2 = MinMaxScaler()
X_scaled2 = scaler2.fit_transform(X = iris.drop('Species', axis = 1))



# new_df = X_scaled + y
new_df2 = pd.DataFrame(X_scaled2, columns=list(iris.columns[:4]))
new_df2['y']=y

new_df2














