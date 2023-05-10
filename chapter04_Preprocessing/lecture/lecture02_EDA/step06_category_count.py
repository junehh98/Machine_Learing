<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
object형 변수를 대상으로 빈도수 확인    
 - object형 변수 중에서 범주형으로 사용할 변수 확인    
"""
 
import seaborn as sns # dataset, 시각화 
import matplotlib.pyplot as plt # 시각화

# 텍시 관련 데이터셋 
taxis = sns.load_dataset('taxis')
taxis.info()
'''
RangeIndex: 6433 entries, 0 to 6432
Data columns (total 14 columns):
 0   pickup           6433 non-null   object 
 1   dropoff          6433 non-null   object 
 2   passengers       6433 non-null   int64  
 3   distance         6433 non-null   float64
 4   fare             6433 non-null   float64
 5   tip              6433 non-null   float64
 6   tolls            6433 non-null   float64
 7   total            6433 non-null   float64
 8   color            6433 non-null   object 
 9   payment          6389 non-null   object 
 10  pickup_zone      6407 non-null   object 
 11  dropoff_zone     6388 non-null   object 
 12  pickup_borough   6407 non-null   object 
 13  dropoff_borough  6388 non-null   object 
'''


# 칼럼 단위 결측치 확인 
taxis.isna().any() #  False : 결측치 유무 
taxis.isna().sum() # 0 : 결측치 개수 확인


# object 자료형 
taxis['pickup'].dtype 


# object 범주 및 유일값 확인 
taxis['pickup'].unique()


# object형 -> 범주형 변수 선택(범주 9개 이하)  
def object_count(column, df) :    
    if df[column].dtype == 'O' and df[column].nunique() <= 9:        
        print('칼럼명 : ', column)
        print('유일값 : ', df[column].unique())
        print('유일값 개수 :', df[column].nunique())
        
        # 범주별 빈도수 시각화
        sns.countplot(x=column, data=df)
        plt.title(f'{column} name')
        plt.ylabel(f'{column}개수')
        plt.show()
        

colnames = list(taxis.columns)
colnames

for col in colnames :
    object_count(col, taxis)






























=======
# -*- coding: utf-8 -*-
"""
object형 변수를 대상으로 빈도수 확인    
 - object형 변수 중에서 범주형으로 사용할 변수 확인    
"""
 
import seaborn as sns # dataset, 시각화 
import matplotlib.pyplot as plt # 시각화

# 텍시 관련 데이터셋 
taxis = sns.load_dataset('taxis')
taxis.info()
'''
RangeIndex: 6433 entries, 0 to 6432
Data columns (total 14 columns):
 0   pickup           6433 non-null   object 
 1   dropoff          6433 non-null   object 
 2   passengers       6433 non-null   int64  
 3   distance         6433 non-null   float64
 4   fare             6433 non-null   float64
 5   tip              6433 non-null   float64
 6   tolls            6433 non-null   float64
 7   total            6433 non-null   float64
 8   color            6433 non-null   object 
 9   payment          6389 non-null   object 
 10  pickup_zone      6407 non-null   object 
 11  dropoff_zone     6388 non-null   object 
 12  pickup_borough   6407 non-null   object 
 13  dropoff_borough  6388 non-null   object 
'''


# 칼럼 단위 결측치 확인 
taxis.isna().any() #  False : 결측치 유무 
taxis.isna().sum() # 0 : 결측치 개수 확인


# object 자료형 
taxis['pickup'].dtype 


# object 범주 및 유일값 확인 
taxis['pickup'].unique()


# object형 -> 범주형 변수 선택(범주 9개 이하)  
def object_count(column, df) :    
    if df[column].dtype == 'O' and df[column].nunique() <= 9:        
        print('칼럼명 : ', column)
        print('유일값 : ', df[column].unique())
        print('유일값 개수 :', df[column].nunique())
        
        # 범주별 빈도수 시각화
        sns.countplot(x=column, data=df)
        plt.title(f'{column} name')
        plt.ylabel(f'{column}개수')
        plt.show()
        

colnames = list(taxis.columns)
colnames

for col in colnames :
    object_count(col, taxis)






























>>>>>>> 32f7d70641783dee9a7f41f16c0d9a0ed6467ceb
