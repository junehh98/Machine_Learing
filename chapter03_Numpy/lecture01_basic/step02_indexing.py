# -*- coding: utf-8 -*-
"""
indexing/slicing 
 - 1차원 indexing 
 - 2,3차원 indexing 
 - boolean indexing  
"""

import numpy as np

# 1. 색인(indexing) : object의 자료 참조  

# 1) list 배열 색인
ldata = [0,1,2,3,4,5]
print(ldata[:]) # 전체 원소 
print(ldata[3]) # 특정 원소 1개 
print(ldata[:3]) # 범위 선택 (0~n-1)
print(ldata[-1]) # 오른쪽 기준(-)

# 2) numpy 다차원 배열 색인 : list 동일 
arr = np.arange(10) # 0~9
print(arr[:])
print(arr[3])
print(arr[:3])
print(arr[-1])


# 2. slicing : 특정 부분을 잘라서 new object
arr = np.arange(10) # 0~9

# 주소 복사 
arr_obj = arr[1:4] # 주소 반환 
print(arr_obj) # [1 2 3]

arr_obj[:] = 100 # 전체 수정(o)
print(arr_obj) # [100 100 100]

print(arr) # 원본 변경 


# 3. 고차원 색인(indexing)

# 1) 2차원 indexing : ppt. 21참조 
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 중첩list
print(arr2d)

# 행 index(기본)
print(arr2d[0, :]) # 1행 전체 
print(arr2d[0]) # 행 index(기본)
print(arr2d[1:,1:])
print(arr2d[::2]) # 홀수행 선택 

# 비연속 행렬
print(arr2d[[0,2]])

print(arr2d[[0,2], [0,2]]) # 1행1열, 3행3열 


# 2) 3차원 indexing 
arr3d = np.array([[[1,2,3],[4,5,6]], [[7,8,9], [10,11,12]]])
print(arr3d)
print(arr3d.shape) # (2, 2, 3) - (면,행,열)

# 면 index(기본)
print(arr3d[0]) # 1면 전체 
print(arr3d[0, 1])  # 1면의 1행 전체 : [4 5 6]
print(arr3d[0, 1, 1:])  # 1면 1행 2~3열 : [5 6]

# image: [size. height, weight, c(color=3)] -> [100, 150, 120, 3]




# 4. 조건식 색인(boolean index)
dataset = np.random.randn(3, 4) # 12개 
print(dataset)
'''
[[ 1.31757489 -0.38784468 -0.08494841  0.57136138]
 [-1.85763682  0.00219376  0.98089281 -2.00417053]
 [-0.61041334 -0.71765377 -1.09149542 -0.1069251 ]]
'''

# 0.7 이상 경우 
print(dataset[dataset >= 0.7])
# [1.31757489 0.98089281]


# numpy 논리식 함수 
'''
np.logical_and() # 논리곱  :&
np.logical_or() # 논리합 : |
np.logical_not() # 부정 : ~
np.logical_xor() # 배타적 논리합 
'''

#0.1 ~ 1.5 범위 : &
result = dataset[np.logical_and(dataset >= 0.1, dataset <= 1.5)]
print('0.1 ~ 1.5 : 범위 조건식')
print(result) # [1.31757489 0.57136138 0.98089281]


result2 = dataset[np.logical_not(dataset >= 0.1)]
result2 # 0.1 이하의 값만





















