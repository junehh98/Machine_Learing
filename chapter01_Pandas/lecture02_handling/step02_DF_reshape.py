# -*- coding: utf-8 -*-
"""
step02_DF_reshape.py

- DataFrame 모양 변경 
"""

import pandas as pd 

path = r'C:\ITWILL\5_Python_ML\data'

buy = pd.read_csv(path + '/buy_data.csv')

print(buy.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 22 entries, 0 to 21
Data columns (total 3 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   Date         22 non-null     int64
 1   Customer_ID  22 non-null     int64
 2   Buy          22 non-null     int64
'''

buy.shape # (22, 3) : 2차원(행, 열)구조
buy.head()
'''
       Date  Customer_ID  Buy
0  20150101            1    3 -> 1행 -> 3개원소
1  20150101            2    4
2  20150102            1    2
3  20150101            2    3
4  20150101            1    2
'''

# 1. 2차원 -> 1차원 구조 변경
buy_long = buy.stack() 

buy_long.shape # (66,) -> 22*3


# 2. 1차원 -> 2차원 구조 변경 
buy_wide = buy_long.unstack()
buy_wide.shape # (22, 3) : 2차원 
buy_long.unstack(0) # 3행 22열로 반환 


# 3. 전치행렬 
buy_tran = buy.T
buy_tran


# 4. 중복 행 제거 
buy.duplicated() # 중복 행 여부 확인 
buy2 = buy.drop_duplicates() # 중복 행 제거
buy2.shape # (20, 3) # (20, 3)
# 특정 컬럼 기준 중복 확인
buy['Date'].duplicated


# 5. 특정 칼럼을 index 지정 
buy.index # RangeIndex(start=0, stop=22, step=1)
new_buy = buy.set_index('Date')

new_buy.shape # (22, 2)
new_buy

# 2015.1.3 구매정보
new_buy.loc[20150103] # 색인 명칭
new_buy.iloc[0] 


# 기업 주가 정보 
stock = pd.read_csv(path + '/stock_px.csv')
stock.info()
'''
RangeIndex: 2214 entries, 0 to 2213
Data columns (total 5 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Unnamed: 0  2214 non-null   object 
 1   AAPL        2214 non-null   float64
 2   MSFT        2214 non-null   float64
 3   XOM         2214 non-null   float64
 4   SPX         2214 non-null   float64
'''

# 칼럼명 수정 
stock.columns = ['Date', 'AAPL', 'MSFT', 'XOM', 'SPX']

stock.columns
# ['Date', 'AAPL', 'MSFT', 'XOM', 'SPX']

stock.head()


# 2) object형 -> 날짜형 변환
stock['Date']=pd.to_datetime(stock['Date'])
stock.info() 
# 0   Date    2214 non-null   datetime64[ns]


# 3) Date 칼럼 -> index 지정
new_stock = stock.set_index('Date') # 날짜 색인 변경 
new_stock # 날짜가 인덱스로 넘어감 
new_stock.head()

aapl = new_stock.AAPL
aapl.shape # (2214,)
type(aapl) # pandas.core.series.Series


# 주가 트렌드
aapl.plot() # 2003 ~ 2011


# 년단위 2008 ~ 2009년 
aapl['2008':'2010'].plot()

# 월단위
aapl['2008-09':'2010-05'].plot()

# 일단위
aapl['2008-09-15':'2008-10-15'].plot()













