# -*- coding: utf-8 -*-
"""
scipy 패키지 이용 
 1. 단순선형회귀분석 
 2. 다중선형회귀분석 
"""

from scipy import stats # 통계분석 
import pandas as pd

path = r'C:\ITWILL\5_Python_ML/data'
score_iq = pd.read_csv(path + '/score_iq.csv')
score_iq.info()



# 1. 단순선형회귀분석 : X(독립변수) -> Y(종속변수)에 어떠한 영향을 미치는가 

# 1) 변수 생성 
x = score_iq['iq'] # 독립변수 
y = score_iq['score'] # 종속변수 


# 2) model 생성 
model = stats.linregress(x, y)
print(model)
'''
LinregressResult(
    slope=0.6514309527270075, : x 기울기 
    intercept=-2.8564471221974657, : y 절편 
    rvalue=0.8822203446134699, : 설명력(1에 가까울수록 좋음)
    pvalue=2.8476895206683644e-50, : F검정 : 유의성검정 
    stderr=0.028577934409305443) : 표준오차 
'''

a = model.slope # x 기울기
b = model.intercept # y 절편 


# 회귀방정식으로 y 예측치 
X = 140; Y = 90  

y_pred = (X*a) + b
print(y_pred) # 88.34388625958358

# 모델 오차 
err = Y - y_pred
print('err=', err) # err= 1.6561137404164157


# 전체 x변수를 대상으로 예측치
len(x) # 150
y_pred = (x*a) + b # 150
y_pred # 각각의 예측치가 출력 


# 대표값으로 예측 
y.mean() #  77.77333333333333
y_pred.mean() # 77.77333333333334

y[:10]
y_pred[:10]

df = pd.DataFrame({'y_pred':y_pred, 'y_true':y}) # # 두개의 결과를 비교하기 쉬움
df.head(10)



# 2. 회귀모델 시각화 
import matplotlib.pyplot as plt

# 산점도 
plt.plot(score_iq['iq'], score_iq['score'], 'b.')
# 회귀선 
plt.plot(score_iq['iq'], y_pred, 'r.-')
plt.title('line regression') # 제목 
plt.legend(['x y scatter', 'line regression']) # 범례 
plt.show()
# x가 증가하면 y도 증가하는 그림 


# 3. 다중선형회귀분석 : formula 형식 
from statsmodels.formula.api import ols


# 상관계수 행렬 
corr = score_iq.corr()
corr['score']
'''
sid       -0.014399
score      1.000000
iq         0.882220 : x1
academy    0.896265 : x2
game      -0.298193
tv        -0.819752 : x3
'''

# 최소자승법 
ols_obj = ols(formula='score ~ iq + academy + tv', data = score_iq)

model = ols_obj.fit()

dir(model)

'''
 summary() : 회귀모델 결과 확인 
 params() -> 모델이 가지고 있는 회귀계수를 넘겨줌 
 fittedvalues : 적합치(예측치)
'''


# 회귀모델 결과 확인  
print(model.summary()) 


# 회귀계수값 반환 
print('회귀 계수값\n%s'%(model.params))
'''
회귀 계수값
Intercept    24.722251
iq            0.374196
academy       3.208802
tv            0.192573
dtype: float64
'''


# 다중회귀방정식 : 첫번째 관측치 적용 예 
score_iq.head()
y = 90
x1 = 140
x2 = 2
x3 = 0


y_pred = (x1*0.374196 + x2*3.208802 + x3*0.192573) + 24.722251
print('예측치 :', y_pred) # 예측치 : 83.527295
print('관측치 :', y) # 관측치 : 90
print('오차(잔차) :', y - y_pred) # 6.472705000000005


# model의 적합치 
print('model 적합치')
print(model.fittedvalues)

