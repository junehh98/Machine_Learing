# -*- coding: utf-8 -*-
"""
1. 행렬곱
2. 연립방식의 해 
"""

import numpy as np 

# 1. 행렬곱 : 행렬 vs 행렬 곱셈 연산
'''
  - 행렬의 선형변환(선형결합)  : 차원축소 
  - 회귀방정식 : X변수와 기울기 곱셈 
  - 딥러닝 : 입력변수(x)와 가중치 곱셈 
'''
  
A = np.array([1, 2, 3]).reshape(3,1) # 행렬A
'''
array([[1],
       [2],
       [3]])
'''
B = np.array([2, 3, 4]).reshape(3,1) # 행렬B
'''
array([[2],
       [3],
       [4]])
'''

# A, B 모양 동일 
A.shape # (3, 1)
B.shape # (3, 1)
A @ B # ValueError
'''
 행렬곱 연산 시 주의사항
 1. A, B : 행렬구조
 2. 수 일치 : A(열수) = B(행수)
 3. 행렬곱 연산 : C[Arow, Bcolumns]
'''
A.T.shape # (1, 3)
B.shape # (3, 1)


# 1) 행렬내적 내적 : 상수(scala) 반환 
# A.T @ B
dot1 = A.T.dot(B) 
# array([[20]])


# 2) 행렬내적 외적 : 행렬 반환 
dot2 = A.dot(B.T) 
'''
array([[ 2,  3,  4],
       [ 4,  6,  8],
       [ 6,  9, 12]])
'''


'''
회귀방정식 예
'''
X = np.array([[5,3,2]]) # 입력변수x(3개)
X.shape # (1, 3)

a = np.array([[1.2, 2.0, 0.5]]) # 기울기(가중치)
a.shape # (1, 3)

y_pred = X @ a.T
y_pred # array([[13.]])

y_pred.shape # (1, 1) = X[r], a[c]



# 2. 연립방정식 해(Solve a linear matrix equation): np.linalg.solve(a, b)
'''
연립방정식 : 2개 이상의 방정식을 묶어놓은 것
다음과 같은 연립방정식의 해(x, y) 구하기 
3*x + 2*y = 53
-4*x + 3*y = -35
'''

a = np.array([[3, 2], [-4, 3]])
b = np.array([53, -35])

x, y = np.linalg.solve(a, b)
# (13.470588235294118, 6.294117647058823)


3*x + 2*y # 53.0
-4*x + 3*y # -35.0






















