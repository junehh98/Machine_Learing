'''  
문1) car_crashes 데이터셋을 이용하여 각 단계별로 다중선형회귀모델을 생성하시오.  
'''

import statsmodels.formula.api as sm # 다중회귀모델 
import seaborn as sn # 데이터셋 로드 
from statsmodels.formula.api import ols

# 미국의 51개 주 자동차 사고 관련 데이터셋 
car = sn.load_dataset('car_crashes')  

car.info()
'''
 0   total           51 non-null     float64 : 치명적 충돌사고 운전자 수 
 1   speeding        51 non-null     float64 : 과속 운전자 비율 
 2   alcohol         51 non-null     float64 : 음주 운전자 비율 
 3   not_distracted  51 non-null     float64 : 주시태만이 아닌 경우 충돌에 연루된 비율  
 4   no_previous     51 non-null     float64 : 이전 사고기록 없는 경우 충돌에 연루된 비율  
 5   ins_premium     51 non-null     float64 : 자동차보험료 
 6   ins_losses      51 non-null     float64 : 보험사가 입은 손해 
 7   abbrev          51 non-null     object : 주이름 
''' 


# 단계1 : object형 변수 제거하여 new_df 만들기  
new_df = car.drop('abbrev', axis=1)
new_df.info() # 제거된것 확인 


# 단계2 : 종속변수(total)과 비교하여 상관계수가 0.2미만의 변수들 제거 후 new_df 적용
corr = new_df.corr()
corr['total'] # 기준변수와 상관계수 비교 
'''
total             1.000000
speeding          0.611548
alcohol           0.852613
not_distracted    0.827560
no_previous       0.956179
ins_premium      -0.199702 -> 제거
ins_losses       -0.036011 -> 제거
'''
new_df.drop(['ins_premium', 'ins_losses'], axis=1, inplace = True)
new_df.info()


# 단계3 new_df에서 total 변수를 종속변수, 나머지 변수를 독립변수로 다중회귀모델 생성  
ols_obj = ols(formula='total ~ speeding + alcohol + not_distracted + no_previous', 
              data = new_df)

# 4. 회귀모델 결과 확인 & 해설 
model = ols_obj.fit()
print(model.summary()) 
model.params
'''
Intercept         1.043064
speeding         -0.035533
alcohol           0.485698
not_distracted    0.177615
no_previous       0.724064
'''















