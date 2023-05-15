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
wine = load_wine()

X = wine.data
y = wine.target

X.shape # (178, 13)
y.shape # (178,)


# 2. train/test 생성 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)



# 3. model 생성 : 다항분류 
'''
xgb = XGBClassifier(objective='multi:softprob') # softmax 함수 
model = xgb.fit(X_train, y_train, eval_metric='merror')
'''
xgb = XGBClassifier(objective='multi:softprob', n_estimators=200)



# 4. model 학습 조기종료 
eval_set = [(X_test, y_test)]  

model = xgb.fit(X=X_train, y=y_train, # 훈련셋
                eval_set=eval_set,    # 평가셋 
                eval_metric='merror',  # 평가방법 
                early_stopping_rounds=100, # 조기종료 라운드수
                verbose=True)
'''
[122]	validation_0-merror:0.00000
[123]	validation_0-merror:0.00000
[124]	validation_0-merror:0.00000
'''

# 5. model 평가 
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(acc) # 1.0


report = classification_report(y_test, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      1.00      1.00        19
           2       1.00      1.00      1.00        19

    accuracy                           1.00        54
   macro avg       1.00      1.00      1.00        54
weighted avg       1.00      1.00      1.00        54
'''



# Best Parameter
from sklearn.model_selection import GridSearchCV 


xgb = XGBClassifier()

params = {'colsample_bytree': [0.5, 0.7, 1],
          'learning_rate' : [0.01, 0.3, 0.5],
          'max_depth' : [5, 6, 7],
          'min_child_weight' : [1, 3, 5],
          'n_estimators' : [100, 200, 300]} # dict


gs = GridSearchCV(estimator = xgb, 
             param_grid = params,  cv = 5)

model = gs.fit(X=X_train, y=y_train, eval_metric='merror',
       eval_set = eval_set, verbose=True)


print('best score =', model.best_score_)
# best score = 0.9723101265822786

print('best parameters :', model.best_params_)




# Best model
best_model = XGBClassifier(colsample_bytree=0.5, 
                    learning_rate= 0.5,
                    max_depth= 5,
                    min_child_weight= 3,
                    n_estimators= 100).fit(X_train, y_train)


y_pred = best_model.predict(X_test)

acc = accuracy_score(y_true =y_test,y_pred= y_pred)
print(acc) 


report = classification_report(y_test, y_pred)
print(report)






























