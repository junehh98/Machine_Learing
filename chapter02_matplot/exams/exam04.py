# -*- coding: utf-8 -*-
'''
문4) seaborn의  titanic 데이터셋을 이용하여 다음과 같이 단계별로 시각화하시오.
  <단계1> 'survived','pclass', 'age','fare' 칼럼으로 서브셋 만들기
  <단계2> 'survived' 칼럼을 집단변수로 하여 'pclass', 'age','fare' 칼럼 간의 산점도행렬 시각화
  <단계3> 산점도행렬의 시각화 결과 해설하기              

문5) seaborn의 tips 데이터셋을 이용하여 다음과 같이 단계별로 시각화하시오.
   <단계1> 'total_bill','tip','sex','size' 칼럼으로 서브셋 만들기 
   <단계2> 성별(sex) 칼럼을 집단변수로 하여 total_bill, tip, size 칼럼 간의 산점도행렬 시각화 
   <단계3> 산점도행렬의 시각화 결과 해설하기 
'''

import matplotlib.pyplot as plt
import seaborn as sn


# 문4) seaborn의  titanic 데이터셋을 이용하여 다음과 같이 단계별로 시각화하시오.
titanic = sn.load_dataset('titanic')
print(titanic.info())

# 칼럼명 수정 : 'survived','pclass', 'age','fare'
#  <단계1> 'survived','pclass', 'age','fare' 칼럼으로 서브셋 만들기  


# <단계2> 'survived' 칼럼을 집단변수로 하여 'pclass', 'age','fare' 칼럼 간의 산점도행렬 시각화


# <단계3> 산점도행렬에서 pclass, age, fare와 survived 변수의 관계 해설



# 문5) seaborn의 tips 데이터셋을 이용하여 다음과 같이 단계별로 시각화하시오.
tips = sn.load_dataset('tips')
print(tips.info())

# <단계1> 'total_bill','tip','sex','size' 칼럼으로 서브셋 만들기


# <단계2> 성별(sex) 칼럼을 집단변수로 산점도행렬 시각화 


# <단계3> 산점도행렬에서 total_bill과 tip의 관계를 설명하고 추가로 sex 변수의 관계 해설



