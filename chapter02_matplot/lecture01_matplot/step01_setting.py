# -*- coding: utf-8 -*-
"""
step01_setting.py

 - matplotlib 사용법 
 - 한글/음수 부호 처리방법 
"""
import numpy as np # 숫자 data 생성
import matplotlib.pyplot as plt 


# 1. 차트 dataset 생성 
data = np.random.randn(100) # N(0,1)
data

# 2. 정규분포 시각화 
plt.plot(data)  
plt.title('vaisulize the normal dist') 
plt.xlabel('index')
plt.ylabel('random number')
plt.show() 


# 차트에서 한글 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'


# 음수 부호 지원 
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# 3. 정규분포 시각화 : 한글 적용  
plt.plot(data)  # 시각화(선 그래프)
plt.title('정규분포 난수 시각화') 
plt.xlabel('색인')
plt.ylabel('난수')
plt.show() 













