# -*- coding: utf-8 -*-
"""
문5) wine dataset을 이용하여 다음과 같이 다항분류 모델을 생성하시오. 
 <조건1> tree model 200개 학습
 <조건2> tree model 학습과정에서 조기 종료 100회 지정
 <조건3> model의 분류정확도와 리포트 출력   
"""
from xgboost import XGBClassifier # model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine # 다항분류
from sklearn.metrics import accuracy_score, classification_report


#################################
## 1. XGBoost Hyper Parameter
#################################

# 1. dataset load


# 2. train/test 생성 


# 3. model 생성 : 다항분류 
'''
xgb = XGBClassifier(objective='multi:softprob') # softmax 함수 
model = xgb.fit(X_train, y_train, eval_metric='merror')
'''

# 4. model 학습 조기종료 


# 5. model 평가 

