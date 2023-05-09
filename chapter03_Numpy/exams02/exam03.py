'''
문3) 다음과 같이  단계별로 자료구조를 생성하시오.
    단계1 : 1~84 정수를  이용하여 벡터 생성
    단계2 : 벡터를 대상으로 7x3x4 구조의 3차원 배열로 모양 변경
    단계3 : 3차원 배열을 대상으로 (행,면,열) 축의 순서로 구조 변경
'''

import numpy as np

# 1. vector 생성 
vec = list(range(1, 85))
vec1 = np.array(vec)


# 2. 3차원 배열 
vec2 = vec1.reshape(7,3,4)
vec2.shape # (7, 3, 4)

# 3. transpose(행,면,열)

vec3 = vec2.transpose()
vec3.shape # (4, 3, 7)

