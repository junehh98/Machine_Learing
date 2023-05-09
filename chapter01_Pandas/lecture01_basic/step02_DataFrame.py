# -*- coding: utf-8 -*-
"""
step02_DataFrame.py

DataFrame 자료구조 특징 
 - 2차원 행렬구조(DB의 Table 구조와 동일함)
 - 칼럼 단위 상이한 자료형 
"""

import pandas as pd # 별칭 
from pandas import DataFrame 
from pandas import Series

# 1. DataFrame 객체 생성 

# 1) list와 dict 이용 
names = ['hong', 'lee', 'kim', 'park']
ages = [35, 45, 55, 25]
pays = [250, 350, 450, 250]

# key -> 칼럼명, value -> 칼럼값 
frame = pd.DataFrame({'name':names, 'age': ages, 'pay': pays})

# Seires(1차원) <-> DataFramo(2차원)
gender = pd.Series(['M', 'F', 'F','M'])

# df에 칼럼 추가
frame['gender'] = gender
frame


# DataFrame(2차원) -> Series(1차원)
pays = frame['pay']
type(pays) # pandas.core.series.Series

print('급여평균 =', pays.mean())




# 객체 정보 
frame.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   name    4 non-null      object
 1   age     4 non-null      int64 
 2   pay     4 non-null      int64 
dtypes: int64(2), object(1)
memory usage: 224.0+ bytes
'''
frame.shape # (4, 3)


# 2) numpy 객체 이용
import numpy as np

data = np.arange(12).reshape(3, 4) # 1d -> 2d
print(data) 

# numpy -> pandas
frame2 = DataFrame(data, columns=['a','b','c','d'])
frame2


# 2. DF 칼럼 참조 
path = r'C:/ITWILL/5_Python_ML/data' # 경로 지정
emp = pd.read_csv(path + "/emp.csv", encoding='utf-8')
print(emp.info())
type(emp) # pandas.core.frame.DataFrame
emp.head() # 컬럼명 + 상위관측치 5개 
emp.tail() # 컬럼명 + 하위관측치 5개 




# 1) 단일 칼럼 : 단일 list
no = emp.No # 방법1 : DF.칼럼명 
name = emp['Name'] # 방법2 : DF['칼럼명']
# 방법2 : 칼럼명에 특수문자 or 공백 있는 경우 

print(no)
print(name)

# 2) 복수 칼럼  
df = emp[['No','Pay']]
print(df)




# 3. subset 만들기  : old DF -> new DF

# 1) 특정 칼럼 선택(칼럼수가 작은 경우)
subset1 =  emp[['Name', 'Pay']]
print(subset1)


# 2) 특정 행 제외 
subset2 = emp.drop([1,3]) # 2,4행 제외 : 현재 객체 변경 없음 
print(subset2)


# 3) 조건식으로 행 선택 : 비교연산자, 논리연산자  
subset3 = emp[emp.Pay > 350] # 비교연산자 : 급여 350 이하 제외 
print(subset3)


# 논리연산자 이용 : &(and), |(or), ~(not)
emp[(emp.Pay >= 300) & (emp.Pay <= 400)] # 급여 300 ~ 400 사이  

# 사번 짝수 or 홍길동 선택
emp[(emp.No % 2 == 0) | (emp.Name == '홍길동')]
'''
    No Name  Pay
0  101  홍길동  150
1  102  이순신  450
3  104  유관순  350
'''

# 홍길동 사원의 제외한 나머지 사원
subset4 = emp[~(emp.Name == '홍길동')]


# 칼럼값 이용 : 칼럼.isin([목록])
two_name = emp[emp.Name.isin(['홍길동', '유관순'])] # 2명 선택 
two_name



# 4) columns 이용 : 칼럼이 많은 경우 칼럼명 이용  
iris = pd.read_csv(path + '/iris.csv')

names = list(iris.columns) # 전체 칼럼명 list 반환 
names # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

# 방법1 : Error 발생
# iris.Sepal.Length

# 첫번째 칼럼 : 방법2 
iris['Sepal.Length']



# 변수 선택 
names[:4] # 앞부분 4개 선택 
names[-1] # 끝부분 1개 선택 
names[1:-1] # 중간 3개 선택 

# 변수 제외 
names.remove('Petal.Length') # (value) : 반환값 없음 
names # ['Sepal.Length', 'Sepal.Width', 'Petal.Width', 'Species']


# x변수 : 1,2,4 컬럼
iris_x = iris[names[:3]]
iris_x.head()
 
# y변수 : 5컬럼
iris_y = iris[names[-1]]


# x 변수 : 1~4
iris_x = iris[names[:4]]
iris_x.shape # (150, 4)

iris_y = iris[names[-1]]

# 열단위 평균 통계
iris_x.mean(axis = 0)
'''
Sepal.Length    5.843333
Sepal.Width     3.057333
Petal.Length    3.758000
Petal.Width     1.199333
'''

# 빈도분석
iris_y.value_counts()
'''
setosa        50
versicolor    50
virginica     50
'''



# 4. DataFrame 행열 참조 
'''
DF.loc[행label, 열label]
DF.iloc[행integer, 열integer]
연속 : 콜론(:)
'''

print(emp)
'''
칼럼 : 명칭(label)
행 : 숫자(integer)
'''
# 1) loc 속성 : 명칭 기반 
emp.loc[0, :] # 1행 전체 
emp.loc[0] # 열 생략 가능 
emp.loc[0:2] # 1~3행 전체 

# 2) iloc 속성 : 숫자 위치 기반 
emp.loc[0] # 1행 전체 
emp.iloc[0:2] # 1~2행 전체 
emp.iloc[:,1:] # 2번째 칼럼 이후 연속 칼럼 선택























