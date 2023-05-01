'''
문5) 다음 같은 가중치(weight)와 입력(X)를 이용하여 히든 노드(hidden node)를 구하시오.      
    <단계1> weight(3,3) 행렬 만들기 : 표준정규분포 난수 자료 이용   
    <단계2> X(3,1) 행렬 만들기 : 1,2,3 자료 이용      
    <단계3> weight행렬과  X행렬 대상으로 행렬곱 연산   
           계산식 : hidden(3,1) = weight(3,3) @ X(3,1)   
'''

import numpy as np

print('단계1 : weight 행렬 만들기')
weight = np.random.randn(3,3)
weight.shape # (3, 3)
'''
array([[ 0.86147794,  0.42846273, -1.74303551],
       [-1.22456488,  2.89521052,  1.24882847],
       [-1.88959923,  0.66755347, -2.36664691]])
'''

print('단계2 : X 행렬 만들기')
X = np.array([[1],[2],[3]])
X.shape # (3, 1)
'''
array([[1],
       [2],
       [3]])
'''

print('단계3 : hidden node 계산')

hidden = weight @ X
hidden.shape
hidden
'''
array([[-3.51070313],
       [ 8.31234158],
       [-7.65443302]])
'''



