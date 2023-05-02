######################################
### 결측치 처리
######################################

'''
x변수 : 문자형 -> 숫자형 변환 
x변수 : 결측치 처리 
y변수 레이블 인코딩 
'''

import pandas as pd 
pd.set_option('display.max_columns', 50) # 최대 50 칼럼수 지정

# 데이터셋 출처 : https://www.kaggle.com/uciml/breast-cancer-wisconsin-data?select=data.csv
cencer = pd.read_csv(r'C:\ITWILL\5_Python_ML\data\brastCencer.csv')
cencer.info()
'''
RangeIndex: 699 entries, 0 to 698
Data columns (total 11 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   id               699 non-null    int64  -> 제거 
 1   clump            699 non-null    int64 
 2   cell_size        699 non-null    int64 
 3   cell_shape       699 non-null    int64 
 4   adhesion         699 non-null    int64 
 5   epithlial        699 non-null    int64 
 6   bare_nuclei      699 non-null    object -> x변수(노출원자핵) : 숫자형 변환
 7   chromatin        699 non-null    int64 
 8   normal_nucleoli  699 non-null    int64 
 9   mitoses          699 non-null    int64 
 10  class            699 non-null    int64 -> y변수 
'''


# 1. 변수 제거 
df = cencer.drop(['id'], axis = 1) # 열축 기준 : id 칼럼 제거  


# 2. x변수 숫자형 변환 : object -> int형 변환  
df['bare_nuclei'] = df['bare_nuclei'].astype('int') # error 발생 



# 3. 특수문자 결측치 처리 & 자료형 변환 

# 1) 특수문자 결측치 대체   
import numpy as np 
df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan) # 결측치로 교체 


# 2) 전체 칼럼 단위 결측치 확인 
df.isnull().any() # 1개 이상 결측치 : bare_nuclei         True


# 3) 결측치 제거  
new_df = df.dropna(subset=['bare_nuclei']) # bare_nuclei 기준 결측치 제거   
new_df.shape # (683, 10) : 16개 제거 


# 4) int형 변환 
new_df['bare_nuclei'] = new_df['bare_nuclei'].astype('int64') 



# 4. y변수 레이블 인코딩 : 10진수 변환 
from sklearn.preprocessing import LabelEncoder # class 

# 인코딩 객체 
encoder = LabelEncoder().fit(df['class']) # object 

# data변환 
labels = encoder.transform(df['class'])  
labels # 0 or 1
 
