# -*- coding: utf-8 -*-
"""
step03_PCA_regression.py

주성분 분석(PCA : Principal Component Analysis)
 1. 다중공선성의 진단 :  다중회귀분석모델 문제점 발생  
 2. 차원 축소 : 특징 수를 줄여서 다중공선성 문제 해결 
"""

from sklearn.decomposition import PCA # 주성분 분석 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols # 다중회귀모델('y~x1+x2') 
import pandas as pd 

  
# 1.iris dataset load      
iris = load_iris()

X = iris.data
y = iris.target
X.shape # (150, 4)
'''
array([[1. , 5.1, 3.5, 1.4, 0.2],
       [1. , 4.9, 3. , 1.4, 0.2],
       [1. , 4.7, 3.2, 1.3, 0.2],
       [1. , 4.6, 3.1, 1.5, 0.2],
       [1. , 5. , 3.6, 1.4, 0.2],
'''       

df = pd.DataFrame(X, columns= ['x1', 'x2', 'x3', 'x4'])
df.info()

corr = df.corr() # 상관행렬 
print(corr)

df['y'] = y 
df.columns  # ['x1', 'x2', 'x3', 'x4', 'y']


# 2. 다중선형회귀분석 
ols_obj = ols(formula='y ~ x1 + x2 + x3 + x4', data = df)
model = ols_obj.fit()

# 회귀분석 결과 제공  
print(model.summary()) 
'''
Model:                            OLS   Adj. R-squared:                  0.928
Method:                 Least Squares   F-statistic:                     484.5
Date:                Thu, 05 May 2022   Prob (F-statistic):           8.46e-83
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.1865      0.205      0.910      0.364      -0.218       0.591
x1            -0.1119      0.058     -1.941      0.054      -0.226       0.002
x2            -0.0401      0.060     -0.671      0.503      -0.158       0.078
x3             0.2286      0.057      4.022      0.000       0.116       0.341
x4             0.6093      0.094      6.450      0.000       0.423       0.796
==============================================================================
'''


#  3. 다중공선성의 진단
'''
분산팽창요인(VIF, Variance Inflation Factor) : 다중공선성 진단  
통상적으로 10보다 크면 다중공선성이 있다고 판단
''' 
from statsmodels.stats.outliers_influence import variance_inflation_factor

dir(ols_obj) # ols 객체 
'''
endog : 모형식에서 사용되는 종속변수 값 
exog : 모형식에서 사용되는 독립변수 값
exog_names : 독립변수 이름  
'''
ols_obj.exog_names # ['Intercept', 'x1', 'x2', 'x3', 'x4']

# 형식) variance_inflation_factor(exog, exog_idx)
exog = ols_obj.exog # 엑소(exog)
exog # [1. , 5.1, 3.5, 1.4, 0.2]

# 다중공선성 진단  
for idx in range(1,5) : # 1~4
    print(variance_inflation_factor(exog, idx)) # idx=1~4
'''
7.072722013939533   : x1
2.1008716761242523  : x2
31.26149777492164   : x3
16.090175419908462  : x4
'''    

    
# 4. 주성분분석(PCA)

# 1) 주성분분석 모델 생성 
pca = PCA() # random_state=123
X_pca = pca.fit_transform(X)
print(X_pca)

X.shape # (150, 4)
X_pca.shape # (150, 4)


# 2) 고유값이 설명가능한 분산비율(분산량) : 85% 이상 가능, 95% 권장(손실율 감안) 
var_ratio = pca.explained_variance_ratio_
print(var_ratio) # [0.92461872 0.05306648 0.01710261 0.00521218]

# 3) 스크리 플롯 : 주성분 개수를 선택할 수 있는 그래프(Elbow Point : 완만해지기 이전 선택)
plt.bar(x = range(4), height=var_ratio)
plt.plot(var_ratio, color='r', linestyle='--', marker='o') ## 선 그래프 출력
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.xticks(range(4), labels = range(1,5))
plt.show()


# 4) 주성분 결정 : 분산비율(분산량) 95%에 해당하는 지점
print(X_pca[:, :2]) # 주성분분석 2개 차원 선정  

X_new = X_pca[:, :2]

X.shape # (150, 4)
X_new.shape # (150, 2)

# 5. 주성분분석 결과를 회귀분석과 분류분석의 독립변수 이용 

from sklearn.linear_model import LinearRegression # 선형회귀모델  
from sklearn.linear_model import LogisticRegression # 로지스틱회귀모델  

from sklearn.metrics import r2_score, mean_squared_error,accuracy_score # 평가 
from sklearn.model_selection import train_test_split # datasets split 


def regression(X, y) :
    # 1. train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.3, random_state=123)
    # 2. model 생성 : train 
    model = LinearRegression().fit(X=X_train, y=y_train)
    
    # 3. model 평가 : test 
    y_pred = model.predict(X = X_test)
    
    # r2_score or mean_squared_error
    score = r2_score(y_true=y_test, y_pred=y_pred)
    print('r2 score : ', score)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    print('MSE : ', mse)
    
# 원본 자료 
regression(X = X, y = y)    
# 차원 축소 자료 
regression(X = X_new, y = y)        
'''
r2 score :  0.9424492525070314
MSE :  0.04473327237231242
r2 score :  0.9230166602115436
MSE :  0.059837914482484156
'''

def classifier(X, y) :
    # 1. train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.3, random_state=123)
    # 2. model 생성 : train 
    model = LogisticRegression().fit(X=X_train, y=y_train)
    
    # 3. model 평가 : test 
    y_pred = model.predict(X = X_test)
    
    # accuracy_score
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print('accuracy score : ', score)

# 원본 자료 
classifier(X = X, y = y)    
# 차원 축소 자료 
classifier(X = X_new, y = y)
'''
accuracy score :  0.9333333333333333
accuracy score :  0.9333333333333333
'''






