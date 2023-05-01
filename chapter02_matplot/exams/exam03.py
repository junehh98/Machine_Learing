# -*- coding: utf-8 -*-
'''
문3) stock 데이터셋을 이용하여 다음과 같은 단계로 시계열 자료를 시각화하시오.
  <단계1> Date 칼럼을 날짜시간 자료형 변환
  <단계2> 2007년 10월 ~ 2010년 10월 까지 subset 만들기
  <단계3> AAPL사와 XOM사 주가를 대상으로 시계열 자료 시각화              
'''

import pandas as pd
import matplotlib.pyplot as plt

path = r'C:\ITWILL\5_Python_ML\data'

# 4대 기업 주가정보 
stock = pd.read_csv(path + '/stock_px.csv')
stock.info()

# 칼럼명 수정 
stock.columns = ['Date','AAPL', 'MSFT', 'XOM', 'SPX']
stock.info()
'''
RangeIndex: 2214 entries, 0 to 2213
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Date    2214 non-null   object 
 1   AAPL    2214 non-null   float64
 2   MSFT    2214 non-null   float64
 3   XOM     2214 non-null   float64
 4   SPX     2214 non-null   float64
'''


# <단계1> Date 칼럼을 날짜시간 자료형 변환  
stock['Date'] = pd.to_datetime(stock['Date'])
stock.info()


# <단계2> 2007년 10월 ~ 2010년 10월 까지 subset 만들기 
df = stock.set_index('Date') # index 지정
stock_subset = df['2007-10':'2010-10']
stock_subset


# <단계3>  AAPL사와 XOM사 주가를 대상으로 시계열 자료 시각화
stock_subset[['AAPL','XOM']].plot()
plt.show




















