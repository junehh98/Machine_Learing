# -*- coding: utf-8 -*-
"""
step06_multi_line.py

 - marker, color, line style, label 이용 
"""

import numpy as np # 수치 data 생성 
import matplotlib.pyplot as plt # data 시각화 
plt.style.use('ggplot') # 차트내 격차 제공 


# 1. data 생성 : 정규분포
data1 = np.random.normal(0.5, 0.3, 100) 
data2 = np.random.normal(0.7, 0.2, 100) 
data3 = np.random.normal(0.1, 0.9, 100) 
data4 = np.random.normal(0.4, 0.3, 100) 

# 2. Fugure 객체 
fig = plt.figure(figsize = (12, 5)) 
chart = fig.add_subplot() # 1개 격자 

# 3. plot : 시계열 시각화 
#help(chart.plot)
chart.plot(data1, marker='o', color='blue', linestyle='-', label='data1')
chart.plot(data2, marker='+', color='red', linestyle='--', label='data2')
chart.plot(data3, marker='*', color='green', linestyle='-.', label='data3')
chart.plot(data4, marker='s', color='orange', linestyle=':', label='data4')
plt.title('Line plots : marker, color, linestyle')
plt.xlabel('index')
plt.ylabel('random number')
plt.legend(loc='best')
plt.show()











