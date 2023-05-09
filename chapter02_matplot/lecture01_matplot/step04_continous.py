# -*- coding: utf-8 -*-
"""
step04_continous.py

 연속형 변수 시각화 : 산점도, 히스토그램, 박스 플롯 
"""

import numpy as np # 수치 data 생성 
import matplotlib.pyplot as plt # data 시각화 

# 차트에서 한글 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
# 음수 부호 지원 
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# 차트 자료 생성 
data1 = np.arange(-3, 7) # -3~6
data2 = np.random.randn(10) # 난수 실수 

# 1. 산점도 : 단일 색상  
plt.scatter(x=data1, y=data2, c='r', marker='o')
plt.title('scatter plot')
plt.show()
# c='색상', marker='모양'

# 군집별 산점도 : 군집별 색상 적용 
cdata = np.random.randint(1, 4, 10) # 난수 정수 : (1, 4] = 1~3
cdata # [3, 3, 3, 2, 2, 1, 2, 1, 2, 2]

plt.scatter(x=data1, y=data2, c=cdata, marker='o')
plt.title('scatter plot')
plt.show()


# 2. 히스토그램 : 대칭성 
data3 = np.random.normal(0, 1, 2000) # N(0,1)
data4 = np.random.normal(0, 1, 2000) # N(0,1)

data3.mean() # 0.03640860285417842
data3.std() # 1.0101026633581638

data4.mean() # -0.013604202823745224
data4.std() # 1.000497316180641

# 정규분포 시각화 
plt.hist(data3, bins=100, density=True, histtype='step', label='data3')
plt.hist(data4, bins=100, density=True, histtype='step', label='data4')
plt.legend(loc = 'best') # 범례 
plt.show()
'''
loc 속성
best 
lower left/right
upper left/right
center 
'''

# 3. 박스 플롯(box plot)
data5 = np.random.random(100) # 0~1 
print(data5)
dir(data5)

data5.max()
data5.mean() # 0.5021406560506729
data5.min()

import pandas as pd 

# numpy -> pandas 
ser = pd.Series(data5)
ser.describe() # 요약통계량 

plt.boxplot(data5)
plt.show()


# 4. 이상치(outlier) 발견
import pandas as pd 

path = r'C:\ITWILL\5_Python_ML\data'

insurance = pd.read_csv(path + '/insurance.csv')
insurance.info()
'''
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   region    1338 non-null   object 
 6   charges   1338 non-null   float64
'''

# 1) subset 만들기 
df = insurance[['age','bmi']]
df.head()

# 2) 이상치 발견 
df.describe() # # 요약통계량 

# 3) 이상치 시각화 
plt.boxplot(df)
plt.show()

# 4) 이상치 처리 

# age 이상치 처리 : 100세 이하 
new_df = df[df.age <= 100]

plt.boxplot(new_df)
plt.show()

# bmi 이상치 처리 
new_df['bmi'].describe()
q1 = 26.2725
q3 = 34.7
iqr = q3 - q1
step = iqr * 1.5
minval = 0 
maxval = q3 + step # 47.34125

new_df2 = new_df[(new_df.bmi > minval) & (new_df.bmi < maxval)]

plt.boxplot(new_df2)
plt.show()


# 5. 레이블 인코딩 : 문자열 -> 10진수 변환  
from sklearn.preprocessing import LabelEncoder # class 

## 범주형 변수 빈도수 
insurance.region.unique() 
# ['southwest(3)', 'southeast(2)', 'northwest(1)', 'northeast(0)']

# 레이블 인코딩  
encoder = LabelEncoder().fit(insurance.region) # 데이터셋 적용 
dir(encoder)
region_encoding = encoder.transform(insurance.region) # 레이블 인코딩 변환 
region_encoding # [3, 2, 2, ..., 2, 3, 1]

# 칼럼 추가 
insurance['region2'] = region_encoding
insurance.head()

# x축 : 행번호, y축 : 의료비, col : 4개 지역 
plt.scatter(x=insurance.index, y=insurance.charges, 
            c=insurance.region2, marker='o')
plt.title('charges vs  region')
for idx, value in enumerate(region_encoding) : 
    plt.annotate(text=value, 
                 xy=(insurance.index[idx],insurance.charges[idx]))
plt.show()


















