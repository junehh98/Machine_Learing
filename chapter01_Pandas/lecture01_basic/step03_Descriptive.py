# -*- coding: utf-8 -*-
"""
step03_Descriptive.py

1. DataFrame의 요약통계량 
2. 상관계수
"""

import pandas as pd 

path = r'C:\ITWILL\5_Python_ML\data'

product = pd.read_csv(path + '/product.csv')


# DataFrame 정보 보기 
product.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 264 entries, 0 to 263
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   a       264 non-null    int64
 1   b       264 non-null    int64
 2   c       264 non-null    int64
'''

# 앞부분/뒷부분 관측치 5개 보기 
product.head()
product.tail()


# 1. DataFrame의 요약통계량 
dir(product)

summ = product.describe() # 숫자변수 대상 
print(summ)
summ.shape # (8, 3)

# 명칭으로 통계 추출 
q1 = summ.loc['25%']
q3 = summ.loc['75%']

IQR = q3 - q1 
IQR
'''
a    2.0
b    1.0
c    1.0
'''

# 행/열 통계
product.shape # (264, 3)
product.sum(axis = 0) # 행축 : 열단위 합계  
product.sum(axis = 1) # 열축 : 행단위 합계

# 산포도 : 분산, 표준편차 
product.var() # axis = 0
product.std() # axis = 0

# 빈도수 : 집단변수 
product['a'].value_counts() # Series.value_counts()
'''
3    126
4     64
2     37
1     30
5      7
'''

# 유일값 
product['b'].unique() # Series.unique()
type(product['b']) # pandas.core.series.Series
dir(product['b']) # unique

# 2. 상관관계 
cor = product.corr()
print(cor) # 상관계수 행렬 
'''
          a         b         c
a  1.000000  0.499209  0.467145
b  0.499209  1.000000  0.766853
c  0.467145  0.766853  1.000000
'''







