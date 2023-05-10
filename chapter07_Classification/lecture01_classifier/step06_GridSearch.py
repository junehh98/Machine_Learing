# -*- coding: utf-8 -*-
"""
step06_SVM_GridSearch.py
 - Grid Search : best parameger 찾기 
"""

from sklearn.svm import SVC # svm model 
from sklearn.datasets import load_breast_cancer # dataset 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score # 평가 

# 1. dataset load 
X, y = load_breast_cancer(return_X_y= True)
X.shape # (569, 30)


# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. 비선형 SVM 모델 
obj = SVC(C=1.0, kernel='rbf', gamma='scale')

'''
기본 parameter
 C=1.0 : cost(오분류) 조절 : 결정경계 위치 조정
 kernel='rbf' : 커널트릭 함수 
  -> kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
 gamma='scale' : 결정경계 모양 조절 조정 
  -> {'scale', 'auto'} or float
  -> gamma='scale' : 1 / (n_features * X.var())
  -> gamma='auto' : 1 / n_features
'''

model = obj.fit(X=X_train, y=y_train)


# model 평가 
y_pred = model.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy =',acc)


# 4. 선형 SVM : 선형분류 
obj2 = SVC(C=1.0, kernel='linear', gamma='scale')

model2 = obj2.fit(X=X_train, y=y_train)

# model 평가 
y_pred = model2.predict(X = X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy =',acc)


###############################
### Grid Search 
###############################

from sklearn.model_selection import GridSearchCV 

parmas = {'kernel' : ['rbf', 'linear'],
          'C' : [0.01, 0.1, 1.0, 10.0, 100.0],
          'gamma': ['scale', 'auto']} # dict 정의 

# 5. GridSearch model   
grid_model = GridSearchCV(model, param_grid=parmas, 
                   scoring='accuracy',cv=5, n_jobs=-1).fit(X, y)

# 1) Best score 
print('best score =', grid_model.best_score_)

# 2) Best parameters 
print('best parameters =', grid_model.best_params_)
