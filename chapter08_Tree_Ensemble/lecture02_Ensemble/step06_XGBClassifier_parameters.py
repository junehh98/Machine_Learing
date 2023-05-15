# -*- coding: utf-8 -*-
"""
1. XGBoost Hyper parameters 
2. model 학습 조기 종료 
3. Best Hyper parameters
"""

from xgboost import XGBClassifier # model 
from sklearn.datasets import load_breast_cancer # 이항분류 dataset 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import accuracy_score, classification_report # 평가


# 1. XGBoost Hyper parameters 

# 1) dataset load 
cancer = load_breast_cancer()

x_names = cancer.feature_names
print(x_names, len(x_names)) # 30
y_labels = cancer.target_names
print(y_labels) # ['malignant' 'benign'] : 이항

X, y = load_breast_cancer(return_X_y=True)


# 2) train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# 3) model 생성 
model = XGBClassifier().fit(X_train, y_train) # default 


print(model) # default parameters
'''
1.colsample_bytree=1 : 트리 모델 생성 시 훈련셋 샘플링 비율(보통 : 0.6 ~ 1)

2.learning_rate=0.3 : 학습율(보통 : 0.01~0.1) = 0의 수렴속도 

3.max_depth=6 : 트리의 깊이(클 수록 성능이 좋아짐, 과적합 영향)

4.min_child_weight=1 : 자식 노드 분할을 결정하는 가중치(Weight)의 합
  - 값을 작게하면 더 많은 자식 노드 분할(과적합 영향)
  
5. n_estimators=100 결정 트리 개수(default=100), 많을 수록 고성능

6. objective='binary:logistic', 'multi:softprob'
과적합 조절 : max_depth 작게, min_child_weight 크게 
'''               


## 2. model 학습 조기 종료
xgb = XGBClassifier(colsample_bytree=1,
                    learning_rate=0.3,
                    max_depth=6,
                    min_child_weight=1,
                    n_estimators=500) 
print(xgb)


eval_set = [(X_test, y_test)]  


model = xgb.fit(X=X_train, y=y_train, # 훈련셋
                eval_set=eval_set,    # 평가셋 
                eval_metric='error',  # 평가방법 
                early_stopping_rounds=80, # 조기종료 라운드수
                verbose=True) # 학습과정 출력 여부 
'''
[138]	validation_0-error:0.03509
[139]	validation_0-error:0.03509
[140]	validation_0-error:0.03509 -> 140개 tree : 조기종료 
'''


# 3. model 평가 
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(acc) # 0.9649122807017544


report = classification_report(y_test, y_pred)
print(report)
'''

              precision    recall  f1-score   support

           0       1.00      0.91      0.95        68
           1       0.94      1.00      0.97       103

    accuracy                           0.96       171
   macro avg       0.97      0.96      0.96       171
weighted avg       0.97      0.96      0.96       171
'''



# 4. Best Hyper parameters
from sklearn.model_selection import GridSearchCV # class 

# default parameters 
xgb = XGBClassifier()

params = {'colsample_bytree': [0.5, 0.7, 1],
          'learning_rate' : [0.01, 0.3, 0.5],
          'max_depth' : [5, 6, 7],
          'min_child_weight' : [1, 3, 5],
          'n_estimators' : [100, 200, 300]} # dict


gs = GridSearchCV(estimator = xgb, 
             param_grid = params,  cv = 5)

model = gs.fit(X=X_train, y=y_train, eval_metric='error',
       eval_set = eval_set, verbose=True)


print('best score =', model.best_score_)
# best score = 0.9723101265822786

print('best parameters :', model.best_params_)
'''
best parameters : {'colsample_bytree': 0.5, 
                   'learning_rate': 0.5, 
                   'max_depth': 5, 
                   'min_child_weight': 3, 
                   'n_estimators': 100}
'''

# Best model
best_model = XGBClassifier(colsample_bytree=0.5, 
                    learning_rate= 0.5,
                    max_depth= 5,
                    min_child_weight= 3,
                    n_estimators= 100).fit(X_train, y_train)


y_pred = best_model.predict(X_test)

acc = accuracy_score(y_true =y_test,y_pred= y_pred)
print(acc) # 0.9415204678362573


report = classification_report(y_test, y_pred)
print(report)
