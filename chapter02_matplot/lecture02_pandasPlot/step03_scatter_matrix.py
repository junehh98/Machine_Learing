# -*- coding: utf-8 -*-
"""
 - 산점도 행렬과 3차원 산점도 
"""

import pandas as pd # object
import matplotlib.pyplot as plt # chart

path = r'C:\ITWILL\5_Python_ML\data'

# 1. 산점도 행렬 
from pandas.plotting import scatter_matrix

# 3) iris.csv
iris = pd.read_csv(path + '/iris.csv')
cols = list(iris.columns)

x = iris[cols[:4]]
print(x.head())

# 산점도 matrix 
scatter_matrix(x)
plt.show()


# 2. 3차원 산점도 
from mpl_toolkits.mplot3d import Axes3D

col_x = iris[cols[0]]
col_y = iris[cols[1]]
col_z = iris[cols[2]]

cdata = [] # color data 
for s in iris[cols[-1]] : # 'Species'
    if s == 'setosa' :
        cdata.append(1)
    elif s == 'versicolor' :
        cdata.append(2)
    else :
        cdata.append(3)

fig = plt.figure()
chart = fig.add_subplot(projection='3d')

chart.scatter(col_x, col_y, col_z, c = cdata)
chart.set_xlabel('Sepal.Length')
chart.set_ylabel('Sepal.Width')
chart.set_zlabel('Petal.Length')
plt.show()






 
