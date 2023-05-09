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
print(y) # 0 or 1 : label encoding 


# 2. train/test split : 70% vs 30%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=1)


# 3. model 생성 
lr = LogisticRegression(solver='lbfgs', max_iter=2000, 
                        random_state=1)  
help(LogisticRegression) # Ctrl+click : 소스 확인 
'''
solver='lbfgs', : 최적화에 사용되는 기본 알고리즘(solver) 
max_iter=100,  : 반복학습횟수 
random_state=None, : 난수 seed값 지정 
'''

model = lr.fit(X=X_train, y=y_train) # 훈련셋 이용  

dir(model)
'''
predict() : class 예측 
predict_proba() : 확률 예측 
score() : 분류정확도(accuracy) 
'''

# 4. model 평가 
y_pred = model.predict(X = X_test) # class 예측치(0 or 1) 
y_true = y_test # 관측치 

dir(y_true)



# 1) 혼동행렬(confusion_matrix)
con_max = confusion_matrix(y_true, y_pred)
print(con_max)
'''
0 [[ 58   5]  = 63
1  [  6 102]] = 108
'''
# 63 vs 108


# 2) 분류정확도 
acc = accuracy_score(y_true, y_pred)
print('accuracy =',acc) # accuracy = 0.935672514619883
# accuracy = 0.9473684210526315

# 3) 과적합 유무 
model.score(X=X_train, y=y_train) # 0.9623115577889447
model.score(X=X_test, y=y_test) # 0.9473684210526315


#############################
# ROC curve 시각화
#############################

# 1) 확률 예측치
y_pred_proba = model.predict_proba(X = X_test) # 확률 예측 
y_pred_proba.shape # (171, 2) # 0, 1
print(y_pred_proba)
'''
       0(양성)        1(악성)
[[6.36001684e-01 3.63998316e-01] = [0.636,  0.363] 
 [6.38112684e-01 3.61887316e-01]
 [9.03780581e-04 9.99096219e-01]
 [9.92921056e-01 7.07894358e-03]
'''
 
y_pred_proba = y_pred_proba[:, 1] # 1 예측확률 추출  


# 2) ROC curve 
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 

fpr, tpr, _ = roc_curve(y_true, y_pred_proba) # (관측치, 1의 예측확률)
'''
fpr : x축, 위양성비율(1-특이도)
tfp : y축, 민감도 
 _ : 임계값 
'''
fpr # 0 ~ 1
tpr # 0 ~ 1


plt.plot(fpr, tpr, color = 'red', label='ROC curve')
plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='AUC')
plt.legend()
plt.show()

'''
ROC curve FPR vs TPR  

ROC curve x축 : FPR(False Positive Rate) - 실제 음성을 양성으로 잘못 예측할 비율 = FP / FP + TN 
ROC curve y축 : TPR(True Positive Rate) - 실제 양성을 양성으로 정상 예측할 비율  = TP / TP + FN 
'''







