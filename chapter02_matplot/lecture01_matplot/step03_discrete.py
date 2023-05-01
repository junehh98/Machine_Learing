# -*- coding: utf-8 -*-
"""
step03_discrete.py

- 이산형 변수 시각화 : 막대차트, 원차트 
"""
import numpy as np # 수치 data 생성 
import matplotlib.pyplot as plt # data 시각화 

# 차트에서 한글과 음수 부호 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


# 차트 자료 생성 
data = [127, 90, 201, 150, 250] # 국가별 수출현황 
idx = np.arange(len(data)) # 색인
labels = ['싱가폴','태국','한국','일본','미국'] 

# 1. 세로막대 
plt.bar(x = idx+2000, height=data) # x:년도, y:수출현황 
plt.title('국가별 수출현황')
plt.xlabel('년도별')
plt.ylabel('수출현황(단위 : $)')
plt.show()


# 2. 가로막대 
plt.barh(y= idx+2000, width = data)# x:년도, y:수출현황 
plt.title('국가별 수출현황')
plt.xlabel('수출현황(단위 : $)')
plt.ylabel('년도별')
plt.show()


# 3. 원차트
plt.pie(x = data, labels = labels)
plt.show()


# 백분율
arr_dat = np.array(data)
data_ratio = arr_dat / arr_dat.sum()
data_ratio = np.round(data_ratio * 100, 2)
data_ratio # array([15.53, 11. , 24.57, 18.34, 30.56])



# 파이차트에 비율 반영 
new_labels = []
for i in range(len(data)):
    new_labels.append(labels[i] + '\n' + str(data_ratio[i])+'%')

plt.pie(x=data_ratio, labels=new_labels)
plt.show()

















