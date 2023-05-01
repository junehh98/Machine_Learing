# -*- coding: utf-8 -*-
"""
step01_line_bar_pie.py

형식) object.plot(kind='유형', 속성) 
"""

import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 

# 1. 기본 차트 시각화 

# 1) Series 객체 시각화 
ser = pd.Series(np.random.randn(10),
          index = np.arange(0, 100, 10))

ser.plot() # 선 그래프 
plt.show()

# 2) DataFrame 객체 시각화
df = pd.DataFrame(np.random.randn(10, 4),
                  columns=['one','two','three','fore'])

# 기본 차트 : 선 그래프 
df.plot()  
plt.show()


# 2. dataset 이용 
path = r'C:\ITWILL\5_Python_ML\data'

tips = pd.read_csv(path + '/tips.csv')
tips.info()


# 행사 요일별 : 파이 차트 
cnt = tips['day'].value_counts()
cnt.plot(kind = 'pie')
plt.show()






