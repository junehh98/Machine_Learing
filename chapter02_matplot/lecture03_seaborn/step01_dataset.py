# -*- coding: utf-8 -*-
"""
step01_dataset.py

Seaborn : Matplotlib 기반 다양한 배경 테마, 통계용 차트 제공 
커널 밀도(kernel density), 카운트 플롯, 다차원 실수형 데이터,2차원 카테고리 데이터
2차원 복합 데이터(box-plot), heatmap, catplot 
"""

import seaborn as sn # 별칭 

# 1. 데이터셋 확인 
names = sn.get_dataset_names()
print(names)


# 2. 데이터셋 로드 
iris = sn.load_dataset('iris')
type(iris) # pandas.core.frame.DataFrame
print(iris.info())
'''
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object 
'''

flights = sn.load_dataset('flights')
flights.info()
'''
0   year        144 non-null    int64   
1   month       144 non-null    category
2   passengers  144 non-null    int64
'''



