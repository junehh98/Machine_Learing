# -*- coding: utf-8 -*-
"""
step05_subplot.py

 subplot 차트 시각화 
"""

import numpy as np # 수치 data 생성 
import matplotlib.pyplot as plt # data 시각화 


# 1. subplot 생성 
fig = plt.figure(figsize = (10, 5)) # 차트 size 지정 
x1 = fig.add_subplot(2,2,1) # 2행2열 1번 
x2 = fig.add_subplot(2,2,2) # 2행2열 1번 
x3 = fig.add_subplot(2,2,3) # 2행2열 1번 
x4 = fig.add_subplot(2,2,4) # 2행2열 1번 

# 2. 각 격차 차트 그리기
data1 = np.random.randn(100)
data2 = np.random.randint(1, 100, 100)
cdata = np.random.randint(1, 4, 100)


