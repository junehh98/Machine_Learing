'''
k겹 교차검정(cross validation)
 - 전체 dataset을 k등분 
 - 검정셋과 훈련셋을 서로 교차하여 검정하는 방식 
'''

from sklearn.datasets import load_digits # 0~9 손글씨 이미지 
from sklearn.ensemble import RandomForestClassifier # RM
from sklearn.metrics import accuracy_score # 평가 
from sklearn.model_selection import cross_validate # 교차검정 

# 1. dataset load 
digits = load_digits()

X = digits.data
y = digits.target

X.shape # (1797, 64) 
y.shape # (1797,)
y 


# 2. model 생성 : tree 100개 학습 
model = RandomForestClassifier().fit(X, y) 


# 3. Test set 선정 
import numpy as np 

idx = np.random.choice(a=len(X), size=int(len(X) * 0.3), replace = False)
X_test = X[idx]
y_test = y[idx]


# 4. model 평가 
y_pred = model.predict(X = X_test) # 예측치  
y_pred 

# model 평가 
acc = accuracy_score(y_test, y_pred)
print(acc) 


# 4. k겹 교차검정 (estimator=model, X, y) 
score = cross_validate(model, X_test, y_test, cv=5) # 5겹 교차검정 
print(score)

'''
{'fit_time': array([0.15618896, 0.15621281, 0.15214348, 0.15628815, 0.15613914]), 
 'score_time': array([0.  , 0.01562166, 0.00432587, 0.   , 0.   ]),
 'test_score': array([0.91666667, 0.97222222, 0.92592593, 0.96296296, 0.97196262])}
'''
scores = score['test_score']
scores.mean() #  0.9499307718933888 --> 약 95%의 예측력


