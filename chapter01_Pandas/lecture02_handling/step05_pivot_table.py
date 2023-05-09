# -*- coding: utf-8 -*-
"""
step05_pivot_table.py

피벗테이블(pivot table) 
  - DF 객체를 대상으로 행과 열 그리고 교차 셀에 표시될 칼럼을 지정하여 만들어진 테이블 
   형식) pivot_table(DF, values='교차셀 칼럼',
                index = '행 칼럼', columns = '열 칼럼'
                ,aggFunc = '교차셀에 적용될 함수')  
"""

import pandas as pd 

path = r'C:\ITWILL\5_Python_ML\data'

# csv file 가져오기 
pivot_data = pd.read_csv(path + '/pivot_data.csv')
pivot_data.info()
'''
 0   year     8 non-null      int64  : 년도 
 1   quarter  8 non-null      object : 분기 
 2   size     8 non-null      object : 매출규모
 3   price    8 non-null      int64  : 매출액 
'''
# 1. 핏벗테이블 작성
ptable = pd.pivot_table(data=pivot_data, 
               values='price', 
               index=['year','quarter'], 
               columns='size', aggfunc='sum')
 
print(ptable)


# 2. 핏벗테이블 시각화 : 누적형
ptable.plot(kind='barh', stacked=True)
