# -*- coding: utf-8 -*-
"""
kMeans 알고리즘 
 - 확인적 군집분석 
 - 군집수 k를 알고 있는 분석방법 
"""

import pandas as pd # DataFrame 
from sklearn.cluster import KMeans # model 
import matplotlib.pyplot as plt # 군집결과 시각화 
import numpy as np # array 


# 1. text file -> dataset 생성 
file = open('C:/ITWILL/5_Python_ML/data/testSet.txt')
lines = file.readlines() # list 반환 

print(lines)

dataset = [] # 2차원(80x2)

for line in lines : 
    cols = line.split('\t') # tab키 기준 분리 
    
    rows = [] # 1줄 저장 
    for col in cols : # 칼럼 단위 추가 
        rows.append(float(col)) # 문자열 -> 실수형 변환  
        
    dataset.append(rows) 
      
    
print(dataset) # 중첩 list   
dataset[0] # [1.658985, 4.285136]
dataset[-1] # [-4.905566, -2.91107]


# list -> numpy(array) print(dataset_arr)
#
# print()
dataset_arr = np.array(dataset)


dataset_arr.shape # (80, 2)

# 2. numpy -> DataFrame(column 지정)
data_df = pd.DataFrame(dataset_arr, columns=['x', 'y'])
data_df.info()
'''
RangeIndex: 80 entries, 0 to 79
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   x       80 non-null     float64
 1   y       80 non-null     float64
'''


# 3. KMeans model 생성 
obj = KMeans(n_clusters=4, max_iter=300, algorithm='auto')

model = obj.fit(data_df) # 학습 수행 


# 예측치 생성 
pred = model.labels_ # 예측된 레이블 번호 
# pred = model.predict(data_df) # test set 
print(pred) # 0 ~ 3


# 군집 중앙값 
centers = model.cluster_centers_
print(centers)
'''
      x            y
[[ 2.80293085 -2.7315146 ]
 [-2.46154315  2.78737555]
 [-3.38237045 -2.9473363 ]
 [ 2.6265299   3.10868015]]
'''

# clusters 시각화 : 예측 결과 확인 
data_df['predict'] = pred # 칼럼추가 


# 산점도 
plt.scatter(x=data_df['x'], y=data_df['y'], 
            c=data_df['predict'])

# 중앙값 추가 
plt.scatter(x=centers[:,0], y=centers[:,1], 
            c='r', marker='D')
plt.show()










