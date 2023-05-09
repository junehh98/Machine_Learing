
# -*- coding: utf-8 -*-
"""
random 모듈
 - 난수 생성 함수 제공 
"""

import numpy as np # 별칭 
import matplotlib.pyplot as plt # 그래프

# 1. 난수 실수와 정수  

# 1) 난수 실수 : [0, 1)
data = np.random.rand(5, 3) # (행, 열)
print(data)

# 차원 모양
print(data.shape) # (5, 3) -> (300, 400)

# 난수 통계
print(data.min()) # 0.08132439151228676
print(data.max()) # 0.9462959398267136
print(data.mean()) # 0.63050758946005

# 2) 난수 정수 : [a, b) 
data = np.random.randint(165, 175, size=50) # (행, 열)
print(data)

# 차원 모양
print(data.shape) # (50,)

# 난수 통계
print(data.min()) # 165
print(data.max()) # 174
print(data.mean()) # 169.78




# 2. 정규분포
height = np.random.normal(173, 5, 2000) # N(173, 5^2)
print(height) # (2000,)

height2 = np.random.normal(173, 5, (500, 4))# N(173, 5^2)
print(height2) # (500, 4)


# 난수 통계
print(height.mean()) # 173.64868062947306
print(height2.mean()) # 173.38566887645658

# 정규분포 시각화 
plt.hist(height, bins=100, density=True, histtype='step')
plt.show()




# 3. 표준정규분포 
standNormal = np.random.randn(500, 3)
print(standNormal.mean()) # -0.04444361993656145

# normal 함수 이용 
standNormal2 = np.random.normal(0, 1, (500, 3))
print(standNormal2.mean())


# 정규분포 시각화 
plt.hist(standNormal[:,0], 
         bins=100, density=True, histtype='step', label='col1')
plt.hist(standNormal[:,1], 
         bins=100, density=True, histtype='step', label='col2')
plt.hist(standNormal[:,2], 
         bins=100, density=True, histtype='step',label='col3')
plt.legend(loc='best')
plt.show()



# 4. 균등분포
uniform = np.random.uniform(low=-1, high=0, size=1000)
uniform

plt.hist(uniform, bins = 10, density = True)
plt.show()





import pandas as pd

path = r'C:\ITWILL\5_Python_ML\data'



# 5. DataFrame sampling

## csv file 가져오기
wdbc = pd.read_csv(path + '/wdbc_data.csv')
print(wdbc.info())
wdbc.shape

# 1) seed값 적용 
np.random.seed(123)

# 2) pandas sample() 이용  
wdbc_df = wdbc.sample(400)
print(wdbc_df.shape) #  (400, 32)
print(wdbc_df.head())


# 3) training vs test sampling
idx = np.random.choice(a=len(wdbc), size=int(len(wdbc) * 0.7), replace = False)
'''
a : 전체관측치 수
size : 선택할 sample 수
'''

len(idx) # 398

# training dataset : 70%
train_set = wdbc.iloc[idx] # iloc 속성 이용
print(train_set.shape) # 차원 모양
print(train_set.head()) # training set 확인


# test dataset : 30%
 





















