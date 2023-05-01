# -*- coding: utf-8 -*-
"""
step01_DF_merge.py
"""

import pandas as pd 
pd.set_option('display.max_columns', 100) # 콘솔에서 보여질 최대 칼럼 개수 

path = r'C:\ITWILL\5_Python_ML\data'

wdbc = pd.read_csv(path + '/wdbc_data.csv')
wdbc.info()

'''
RangeIndex: 569 entries, 0 to 568
Data columns (total 32 columns):
'''

# 전체 칼럼 가져오기 
cols = list(wdbc.columns)
len(cols) # 32


# 1. DF 병합(merge) : 공통컬럼을 기준으로 병합 

DF1 = wdbc[cols[:16]] 
DF2 = wdbc[cols[16:]] 
DF2['id'] = wdbc.id # id 칼럼 추가 

DF3 = pd.merge(left=DF1, right=DF2, on='id') 
DF3.info()


# 2. DF 결합(concat)
DF1.shape # (569, 16)
DF2 = wdbc[cols[16:]]
DF2.shape # (569, 16)


# cbine(v1, v2)
DF4 = pd.concat(objs=[DF1, DF2], axis = 1) # 열축 기준 결합
DF4.shape # (569, 32)



# 3. Inner join과 Outer join 
name = ['hong','lee','park','kim']
age = [35, 20, 33, 50]

df1 = pd.DataFrame(data = {'name':name, 'age':age}, 
                   columns = ['name', 'age'])
df1
'''
   name  age
0  hong   35
1   lee   20
2  park   33
3   kim   50
'''

name2 = ['hong','lee','kim']
age2 = [35, 20, 50]
pay = [250, 350, 250]

df2 = pd.DataFrame(data = {'name':name2, 'age':age2,'pay':pay}, 
                   columns = ['name', 'age', 'pay'])

df2
'''
   name  age  pay
0  hong   35  250
1   lee   20  350
2   kim   50  250
'''

# 내부조인
inner = pd.merge(left=df1, right=df2, how='inner')
inner


# 외부조인 
outer = pd.merge(left=df1, right=df2, how='outer')
outer


# 결측치 총합 개수
outer.isna().sum()

# 결측치 처리 : 0으로  채우기
outer.fillna(0)


# 현재 객체 반영  -> 변수 할당하지 않고 바로 반영 
outer.fillna(value=0, inplace = True)
'''
   name  age    pay
0  hong   35  250.0
1   lee   20  350.0
2  park   33    0.0 -> 0이 실제 데이터에 반영 
3   kim   50  250.0
'''





























