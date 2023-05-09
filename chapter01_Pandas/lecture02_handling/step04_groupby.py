# -*- coding: utf-8 -*-
"""
step04_groupby
   - 범주형 변수(집단변수)를 이용한 자료처리 
1. 범주형 변수 기준 subset 만들기 
2. 범주형 변수 기준 그룹 & 통계량 
3. apply() 함수 : DataFrame 객체에 외부함수 적용
4. map()함수 : Series 객체에 외부함수 적용  
"""

import pandas as pd 

 
path = r'C:\ITWILL\5_Python_ML\data'

# dataset load & 변수 확인
wine = pd.read_csv(path  + '/winequality-both.csv')
print(wine.info())
'''
RangeIndex: 6497 entries, 0 to 6496
Data columns (total 13 columns):
0   type                  6497 non-null   object : 집단변수    
 :
12  quality               6497 non-null   int64 : 와인품질 
'''    

# 칼럼 공백 -> '_' 교체 
wine.columns = wine.columns.str.replace(' ', '_')
wine.head()
print(wine.info())

wine['type'] # python명령어랑 겹칠경우 명칭기반으로 하는게 좋음 
wine['type'].value_counts()
'''
white    4898
red      1599
'''
wine['type'].unique() # array(['red', 'white']



# 5개 변수 선택 : subset 만들기 
wine_df = wine.iloc[:, [0,1,4,11,12]] # 위치 기반
print(wine_df.info()) 


# 특정 칼럼명 수정  : {'old','new'}
columns = {'fixed_acidity':'acidity', 'residual_sugar':'sugar'} # {'old','new'} 
wine_df = wine_df.rename(columns = columns) 
# rename 부분 수정할때 효율이 좋음 
wine_df.info() # 칼럼명 변경된것 확인 

# 전체 칼럼명 수정 : DF.columns = ['칼럼1', '칼럼2'...'칼럼n]
    

# 집단변수 확인 : 와인유형   
print(wine_df.type.unique()) # ['red' 'white']
print(wine_df.type.nunique()) # 2


# 이산변수 확인 : 와인 품질    
print(wine_df.quality.unique()) # [5 6 7 4 8 3 9]
print(wine_df.quality.value_counts())




# 1. 범주형 변수 기준 subset 만들기 

# 1) 1개 집단 기준  
red_wine = wine_df[wine['type']=='red']  
red_wine.shape # (1599, 5)

white_wine = wine_df[wine['type']=='white']
white_wine.shape # (4898, 5)



# 2) 2개 이상 집단 기준 : 
two_wine_type = wine_df[wine_df['type'].isin(['red','white'])] 
two_wine_type.shape # (6497, 5)



# 3) 범주형 변수 기준 특정 칼럼 선택 : 1차원 구조
red_wine_quality = wine.loc[wine['type']=='red', 'quality']  
white_wine_quality = wine.loc[wine['type']=='white', 'quality'] 

red_wine_quality.shape # (1599,)
white_wine_quality.shape # (4898,)



# 2. 범주형 변수 기준 group & 통계량

# 1) 범주형변수 1개 이용 그룹화 
type_group = wine_df.groupby('type')
print(type_group)
dir(type_group)

for group in type_group :
    print(group)


# 각 집단별 빈도수 
type_group.size()  
'''
type
red      1599
white    4898
dtype: int64
'''

# 그룹객체에서 그룹 추출 
red_df = type_group.get_group('red')
white_df = type_group.get_group('white')

    
# 그룹별 통계량 
print(type_group.sum()) 
print(type_group.mean())

print(type_group['alcohol'].sum()) 


# 2) 집단변수 2개 이용 : 나머지 변수(3개)가 그룹 대상 
# wine_df.groupby(['집단변수1', '집단변수2']) # 1차 -> 2차
wine_group = wine_df.groupby(['type','quality']) # 2개 x 7개 = 최대 14  

# 각 집단별 빈도수
wine_group.size() # 2개:그룹만드는데 사용, 3개(acidity,sugar,alcohol)



# 그룹 통계 시각화 
grp_mean = wine_group.mean()

grp_mean.plot() # kind='bar' -> bar graph




# 3. apply() 함수 : DataFrame객체에 외부함수 적용 

# 1) 사용자 함수 : 0 ~ 1 사이 정규화 
def normal_df(x):
    nor = ( x - min(x) ) / ( max(x) - min(x) )
    return nor


# 2) 2차원 data 준비 : wine 데이터 적용 
wine_df.info()
wine_x = wine_df.iloc[:, 1:] # 숫자변수만 선택 
wine_x.info()
wine_x.head()
dir(wine_x)


# 3) apply 함수 적용 : 열(칼럼) 단위로 실인수 전달   
wine_nor = wine_x.apply(normal_df) 
print(wine_nor.describe()) # 정규화 확인 

# 평소 함수호출(DF)
# normal_df(wine_x) -> error 

#####################################################################
# lambda : 한 줄 함수 (lambda 인수 : 명령문)
wine_nor2 = wine_x.apply(lambda x : ( x - min(x) ) / ( max(x) - min(x) ))
print(wine_nor2.describe()) 




# 4. map() 함수  : Series객체에 외부함수 적용 

# 1) 인코딩 함수 : 문자열 -> 2진수  
def encoding_df(x):
    encoding = {'red':[1,0], 'white':[0,1]}
    return encoding[x]


# 2) 1차원 data 준비 
wine_type = wine_df['type']
wine_type
type(wine_type) # pandas.core.series.Series


# 3) map 함수 적용 
label = wine_type.map(encoding_df)
label
'''
3       [1, 0]
4       [1, 0]
         ...
6492    [0, 1]
6493    [0, 1]
'''

# 4) lambda 적용
encoding = {'red':[1,0], 'white':[0,1]} # dict : map table

wine_df['type'].map(lambda x : encoding[x])



