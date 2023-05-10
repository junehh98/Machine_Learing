# -*- coding: utf-8 -*-
"""
step01_logisticRegression_ROC.py

 - 이항분류기(binary class classifier) & ROC 평가 
"""

from sklearn.datasets import load_breast_cancer # dataset
from sklearn.linear_model import LogisticRegression # model 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import confusion_matrix, accuracy_score # model 평가 




################################
### 이항분류(binary class) 
################################


# 1. dataset loading 
X, y = load_breast_cancer(return_X_y=True)

print(X.shape) # (569, 30)
print(y) # 0 or 1 : 양성, 악성 


# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=1)



# 3. model 생성 
lr = LogisticRegression(solver='lbfgs', max_iter=100, random_state=1)  
# 반복학습을 많이할수록 정확도가 올라가지만 과적합 문제 발생 가능성이 올라감 

help(LogisticRegression())
'''
solver='lbfgs', : 최적화에 사용되는 기본 알고리즘(solver) 
max_iter=100,  : 반복학습횟수 
random_state=None, : 난수 seed값 지정 
'''

# 모델 학습
model = lr.fit(X=X_train, y=y_train) # 훈련셋 이용 

dir(model)
'''
 score() : 분류정확도(accuracy)
 predict() : class 예측
 predict_proba() : y를 확률로 예측 
'''


# 4. model 평가 
y_pred = model.predict(X = X_test) # class 예측치 
y_true = y_test # 관측치 




# 1) 혼동행렬(confusion_matrix)
con_max = confusion_matrix(y_true, y_pred) # (y의 정답, y의 예측치)
print(con_max)
'''
[[ 59   4] : 양성인경우 59개 예측 성공
 [  5 103]] : 악성인경우 103개 예측 성공 
'''

# 2) 분류정확도 
acc = accuracy_score(y_true, y_pred) #(59+103)/(59+4+5+103)
print('accuracy =',acc) # accuracy = 0.9473684210526315



# 3) 과적합 유무(통상적으로 학습데이터가 높게 나옴)
# train과 test 정확도가 많이 차이나지 않으면 과적합이 없다고 봄 
model.score(X = X_train,y = y_train) # 0.9522613065326633
model.score(X = X_test,y = y_test) # 0.9473684210526315



#############################
# ROC curve 시각화
#############################

# 1) 확률 예측치
y_pred_proba = model.predict_proba(X = X_test) # 확률 예측 
y_pred_proba.shape # (171, 2) # 0, 1
print(y_pred_proba)
'''
0(양성)이 나올 확률   1(악성)이 나올 확률
[[1.39326020e-01 8.60673980e-01]  = [0.139 + 0.86] -> 1이 나옴 
 [6.32838435e-01 3.67161565e-01]
 [1.58068049e-03 9.98419320e-01]
 [9.99932587e-01 6.74127803e-05]
'''

y_pred_proba = y_pred_proba[:, 1] # 1 예측확률 추출  


# 2) ROC curve 
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 

fpr, tpr, _ = roc_curve(y_true, y_pred_proba) # (관측치, 1의 예측확률)
'''
 fpr : x축, 위양성비율(1-특이도)
 tfp : y축, 민감도
 # _ : 임계값
'''
fpr # 0 ~1 사이의 값
tpr # 실제 양성일때 양성, 0 ~ 1 사이의 값 


plt.plot(fpr, tpr, color = 'red', label='ROC curve')
plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='AUC')
plt.legend()
plt.show()
# 면적이 클수록 예측력이 좋음 


'''
ROC curve FPR vs TPR  

ROC curve x축 : FPR(False Positive Rate) - 실제 음성을 양성으로 잘못 예측할 비율 = FP / FP + TN 
ROC curve y축 : TPR(True Positive Rate) - 실제 양성을 양성으로 정상 예측할 비율  = TP / TP + FN 
'''







