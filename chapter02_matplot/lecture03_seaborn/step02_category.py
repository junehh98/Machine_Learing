# -*- coding: utf-8 -*-
"""
step02_category.py

1. Object vs Category 
  - 공통점 : 자료형이 모두 문자열 
  - object : 문자열 순서 변경 불가 
  - category : 문자열 순서 변경 가능 
  
2. 범주형 자료 시각화 
"""

import matplotlib.pyplot as plt
import seaborn as sn


# 1. Object vs Category 

# dataset load
titanic = sn.load_dataset('titanic')

print(titanic.info())

# subset 만들기 
df = titanic[['survived','age','class','who']]
df.info()
'''
 0   survived  891 non-null    int64   
 1   age       714 non-null    float64 
 2   class     891 non-null    category
 3   who       891 non-null    object 
'''
df.head()
'''
   survived   age  class    who
0         0  22.0  Third    man
1         1  38.0  First  woman
2         1  26.0  Third  woman
3         1  35.0  First  woman
4         0  35.0  Third    man
'''


# category형 정렬 
dir(df)
'''
'sort_index' : index 기준 정렬
'sort_values' : 특정 컬럼 기준 정렬 
'''
df.sort_values(by = 'class') # category 오름차순
# First > Second > Third

# object형 정렬 
df.sort_values(by = 'who') # object 오름차순 
# child > man > woman


# category형 변수 순서 변경 : Third > Second > First 
df['class_new'] = df['class'].cat.set_categories(['Third', 'Second', 'First'])
df.info()

df.sort_values(by = 'class_new')
# Third > Second > First


# object -> category 형 변환
df['who_new'] = df['who'].astype('category')
df.info()

df['who_new'].value_counts()
'''
man      537
woman    271
child     83
'''


# 2. 범주형 자료 시각화 

# 1) 배경 스타일 
sn.set_style(style='darkgrid')
tips = sn.load_dataset('tips')
print(tips.info())

# 2) category형 자료 시각화 
sn.countplot(x = 'smoker', data = tips) 
plt.title('smoker of tips')
plt.show()


tips.smoker.value_counts()
'''
No     151
Yes     93
'''

tips.day.value_counts()
'''
Sat     87
Sun     76
Thur    62
Fri     19
'''
sn.countplot(x = 'day', data = tips) 
plt.title('day of tips')
plt.show()


titanic['class'].value_counts()
'''
Third     491
First     216
Second    184
'''

sn.countplot(x = 'class', data = titanic) 
plt.title('class of titanic')
plt.show()


















