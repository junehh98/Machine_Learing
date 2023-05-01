######################################
### 결측치 처리
######################################

'''
 - 결측치 확인 및 처리(제거 및 채우기) 
'''

import pandas as pd
path = r'C:\ITWILL\5_Python_ML\data'
data = pd.read_csv(path +'/dataset.csv') 
data.info()
'''
 0   resident  217 non-null    int64  
 1   gender    217 non-null    int64  
 2   job       205 non-null    float64
 3   age       217 non-null    int64  
 4   position  208 non-null    float64
 5   price     217 non-null    float64
 6   survey    217 non-null    int64  
'''
 

# 1. 결측치(NaN) 확인  
data.isnull().any() # 결측치 칼럼 : job, position, 
data.isnull().sum() # job : 12, postion : 9
data.shape # (217, 7)  


# 2. 전체 칼럼 기준 결측치 제거 
new_data = data.dropna() # 7개 컬럼 전체 
new_data.shape # (198, 7) # 21개의 행 삭제 


# 3. 특정 칼럼 기준 결측치 제거   
new_data2 = data.dropna(subset ='job') # job 칼럼 기준 
new_data2.shape # (205, 7) 

# 2개 이상 칼럼 기준
new_data2 = data.dropna(subset =['job', 'position'])
new_data2.shape # (198, 7)
new_data.isnull().sum()
'''
resident    0
gender      0
job         0
age         0
position    0
price       0
survey      0
dtype: int64
'''


# 4. 모든 결측치 다른값으로 채우기 : 상수 or 통계  
new_data3 = data.fillna(0.0) 
new_data3.shape  
new_data3.isnull().sum() # 없음


# 5-1. 특정변수 결측치 채우기 : 숫자변수(상수 or 통계 대체) 
new_data4 = data.copy() # 내용복제
new_data4.isna().sum() # position 결측치 확인 

# position 결측치 평균 대체  
new_data4['position'].fillna(new_data4['position'].mean(), inplace=True)
new_data4.isna().sum()  # position 결측치 확인 


# 5-2. 특정변수 결측치 채우기 : 범주형변수(빈도수가 높은 값으로 대체)  
new_data5 = data.copy() # 내용복제 
new_data5['job'].fillna(3.0, inplace=True) # 현재 객체 반영 


# 6. 결측치 비율 40% 이상인 경우 해당 컬럼 제거 
data.isna().sum() # job 칼럼 제거 

new_data6 = data.drop(['job'], axis = 1) 
new_data6.shape # (217, 6) : 변수 1개 제거 

