# -*- coding: utf-8 -*-
"""
lecture02 > step02 관련문제

문8) iris 데이터셋을 대상으로 아래와 같이 단계별로 처리하시오. 
"""

import seaborn as sns # seaborn 데이터셋 이용 

iris = sns.load_dataset('iris')
iris.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object
'''
 

# 단계1 : species 칼럼를 대상으로 범주별 빈도수 확인
'''
setosa        50
versicolor    50
virginica     50
'''
iris_group = iris.groupby('species')
iris_group.size()



# 단계2 : species 칼럼을 대상으로 아래와 같이 범주별로 인코딩 변환 후 y 칼럼 추가
'''
각 범주별 인코딩 정보 
setosa        [1, 0, 0]
versicolor    [0, 1, 0]
virginica     [0, 0, 1]
'''
encoding = {'setosa':[1,0,0], 'versicolor':[0,1,0], 'virginica':[0,0,1]}

iris['y'] = iris['species'].map(lambda x : encoding[x])
iris



# 단계3 : iris의 상위 5개와 하위 5개를 관측치 확인  
iris.head()
'''
   sepal_length  sepal_width  petal_length  petal_width species          y
0           5.1          3.5           1.4          0.2  setosa  [1, 0, 0]
1           4.9          3.0           1.4          0.2  setosa  [1, 0, 0]
2           4.7          3.2           1.3          0.2  setosa  [1, 0, 0]
3           4.6          3.1           1.5          0.2  setosa  [1, 0, 0]
4           5.0          3.6           1.4          0.2  setosa  [1, 0, 0]
'''

iris.tail()
'''
     sepal_length  sepal_width  petal_length  petal_width    species          y
145           6.7          3.0           5.2          2.3  virginica  [0, 0, 1]
146           6.3          2.5           5.0          1.9  virginica  [0, 0, 1]
147           6.5          3.0           5.2          2.0  virginica  [0, 0, 1]
148           6.2          3.4           5.4          2.3  virginica  [0, 0, 1]
149           5.9          3.0           5.1          1.8  virginica  [0, 0, 1]
'''