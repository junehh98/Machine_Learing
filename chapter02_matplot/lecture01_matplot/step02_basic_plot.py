# -*- coding: utf-8 -*-
"""
step02_basic_plot.py

 - 기본 차트 그리기 
"""

import numpy as np # 수치 data 생성 
import matplotlib.pyplot as plt # data 시각화 

# 차트에서 한글과 음수 부호 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# 1. 차트 자료 생성 
data = np.arange(-3, 7) 
print(data) # [-3 -2 -1  0  1  2  3  4  5  6]
len(data) # 10

# 2. 기본 차트 
plt.plot(data) # 선색 : 파랑, 스타일 : 실선 
plt.title('선 색 : 파랑, 선 스타일 : 실선 ')
plt.show()


# 3. 색상 : 빨강, 선스타일(+)
plt.plot(data, 'r+')
plt.title('선 색 : 빨강, 선 스타일 : +')
plt.show()

help(plt.plot)

# 4. x, y축 사용, 색상과 마커 기호 
data2 = np.random.randn(10)
data2

plt.plot(data, data2) # 선그래프

plt.plot(data, data2, 'bo') # 산점도
plt.show()




