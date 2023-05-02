# -*- coding: utf-8 -*-
"""
포루투갈의 2차교육과정에서 학생들의 음주에 영향을 미치는 요소는 무엇인가? 
"""

import pandas as pd

## data source : https://www.kaggle.com/uciml/student-alcohol-consumption  
path = r'C:\ITWILL\5_Python_ML\data'
student = pd.read_csv(path + '/student-mat.csv')
student.info()

'''
학생들의 음주에 미치는 영향을 조사하기 위해 6가지의 변수 후보 선정
독립변수 : sex(성별), age(15~22), Pstatus(부모거주여부), failures(수업낙제횟수), famrel(가족관계), grade(G1+G2+G3 : 연간성적) 
          grade : 0~60(60점이 고득점), Alcohol : 0~500(100:매우낮음, 500:매우높음)으로 가공
종속변수 : Alcohol = (Dalc+Walc)/2*100 : 1주간 알코올 섭취정도  
'''

# 1. subset 만들기 
df = student[['sex','age','Pstatus','failures','famrel','Dalc','Walc','G1','G2','G3']]
df.info()
'''
RangeIndex: 395 entries, 0 to 394
Data columns (total 10 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   sex       395 non-null    object : 성별(F, M)
 1   age       395 non-null    int64  : 나이(15 ~ 22)
 2   Pstatus   395 non-null    object : 부모거주여부(T, A)
 3   failures  395 non-null    int64  : 수업낙제횟수(0,1,2,3)
 4   famrel    395 non-null    int64  : 가족관계(1,2,3,4,5)
 5   Dalc      395 non-null    int64  : 1일 알콜 소비량(1,2,3,4,5)   
 6   Walc      395 non-null    int64  : 1주일 알콜 소비량(1,2,3,4,5)  
 7   G1        395 non-null    int64  : 첫번째 학년(0~20)
 8   G2        395 non-null    int64  : 두번째 학년(0~20) 
 9   G3        395 non-null    int64  : 마지막 학년(0~20) 
'''

# 1. 문자형 변수의 빈도수와 숫자형 변수 통계량 확인  
df.sex.value_counts() # 문자형 변수 
'''
F    208
M    187
'''

df.Pstatus.value_counts() # 문자형 변수 
'''
T    354
A     41
'''

df.describe() # 숫자형 변수 


# 2. 파생변수 만들기 
grade = df.G1 + df.G2 + df.G3 # 성적 
grade.describe() # 4 ~ 58 

Alcohol = (df.Dalc + df.Walc) / 2 * 100 # 알콜 소비량 
Alcohol.describe() # 100 ~ 500(100:매우낮음, 500:매우높음)


# 1) 파생변수 추가 
df['grade'] = grade # 1~3학년 성적 
df['Alcohol'] = Alcohol # 알콜소비량 


# 2) 기존 변수 제거
new_df = df.drop(['Dalc','Walc','G1','G2','G3'], axis = 1) # 칼럼 기준 제거 
new_df.info()
'''
0   sex       395 non-null    object : 범주형(F, M)
1   age       395 non-null    int64  : 연속형(15 ~ 22)
2   Pstatus   395 non-null    object : 범주형(T, A)
3   failures  395 non-null    int64  : 이산형(0,1,2,3)
4   famrel    395 non-null    int64  : 이산형(1,2,3,4,5)
5   grade     395 non-null    int64  : 연속형(4~58)
6   Alcohol   395 non-null    float64 : 연속형(100~500)
'''

import seaborn as sn
import matplotlib.pyplot as plt # data 시각화 


# 3. EDA : 종속변수(Alcohol) vs 독립변수 탐색 

### 연속형(y) vs 범주형(x)  

# 1) Alcohol vs sex
sn.countplot(x='sex',  data=new_df) # 빈도수 시각화 
sn.barplot(x='sex', y='Alcohol', data=new_df)  # 오차막대 5%

new_df.groupby('sex').mean() # sex를 기준으로 평균
'''
           age  failures    famrel      grade     Alcohol
sex                                                      
F    16.730769  0.302885  3.894231  30.975962  160.576923
M    16.657754  0.368984  4.000000  33.219251  219.786096 -> 남학생이 알코올 섭취평균이 높음
'''


# 2) Alcohol vs Pstatus
sn.countplot(x='Pstatus',  data=new_df) # 빈도수 시각화
sn.barplot(x='Pstatus', y='Alcohol', data=new_df)
 # 혼자사는 학생이 알코올 섭취가 조금 더 많음, 거의 차이가 없음 
new_df.groupby('Pstatus').mean()



### 연속형(y) vs 이산형(x) 

# 1) Alcohol vs failures
sn.countplot(x='failures',  data=new_df) 
sn.barplot(x='failures', y='Alcohol', data=new_df)
# 수업 낙제가 많을수록 알코올 섭취가 증가함  


# 2) Alcohol vs famrel
sn.barplot(x='famrel', y='Alcohol', data=new_df)
# 가족관계가 좋을수록 알코올 섭취가 줄어듬 




### 연속형(y) vs 연속형(x) 

# 1) Alcohol vs age  
sn.scatterplot(x="age", y="Alcohol", data=new_df) 
# 각 연령대별 알콜 평균으로 시각화


# 2) Alcohol vs grade  
sn.scatterplot(x="grade", y="Alcohol", data=new_df) 
# 각 성적대별 알콜 평균으로 시각화 


sn.barplot(x="grade", y="Alcohol", data=new_df, errwidth=0) 


















