'''
시계열 데이터 시각화 
'''

import pandas as pd
import matplotlib.pyplot as plt


# 1. 날짜형식 수정(다국어)
cospi = pd.read_csv("C:/ITWILL/5_Python_ML/data/cospi.csv")
print(cospi.info())

# object -> Date형 변환 
cospi['Date'] = pd.to_datetime(cospi['Date'])
print(cospi.info())


# 2. 시계열 데이터/시각화

# 1개 칼럼 추세그래프 
cospi['High'].plot(title = "Trend line of High column")
plt.show()

# 2개 칼럼(중첩list) 추세그래프
cospi[['High', 'Low']].plot(color = ['r', 'b'],
        title = "Trend line of High and Low column")
plt.show() 


# index 수정 : Date 칼럼 이용  
new_cospi = cospi.set_index('Date')
print(new_cospi.info())
print(new_cospi.head())

# 날짜형 색인 
new_cospi.index #  DatetimeIndex(['2016-02-26', '2016-02-25',
print(new_cospi['2016']) # 년도 선택 
print(new_cospi['2016-02']) # 월 선택 
print(new_cospi['2016-02':'2016-01']) # 범위 선택 

# 2016년도 주가 추세선 시각화 
new_cospi_HL = new_cospi[['High', 'Low']]
new_cospi_HL['2016'].plot(title = "Trend line of 2016 year")
plt.show()

new_cospi_HL['2016-02'].plot(title = "Trend line of 2016 year")
plt.show()


# 3. 이동평균(평활) : 지정한 날짜 단위 평균계산 -> 추세그래프 스무딩  

# 5일 단위 평균계산 : 평균계산 후 5일 시작점 이동 
# pd.Series.rolling(column, window=5,center=False).mean()
roll_mean5 = pd.Series.rolling(new_cospi.High,
                               window=5, center=False).mean()

roll_mean10 = pd.Series.rolling(new_cospi.High,
                               window=10, center=False).mean()

roll_mean20 = pd.Series.rolling(new_cospi.High,
                               window=20, center=False).mean()
print(roll_mean5)


# 1) High 칼럼 시각화 
new_cospi['High'].plot(color = 'blue', label = 'High column')


# 2) rolling mean 시각화 : subplot 이용 - 격자 1개  
fig = plt.figure(figsize=(12,4))
chart = fig.add_subplot()
chart.plot(new_cospi['High'], color = 'blue', label = 'High column')
chart.plot(roll_mean5, color='red',label='5 day rolling mean')
chart.plot(roll_mean10, color='green',label='10 day rolling mean')
chart.plot(roll_mean20, color='orange',label='20 day rolling mean')
plt.legend(loc='best')
plt.show()















