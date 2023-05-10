# -*- coding: utf-8 -*-
"""
문4) 다음 조건에 맞게 비선형 SVM 모델과 선형 SVM 모델을 생성하시오. 
  <조건1> 비선형 SVM 모델과 선형 SVM 모델 생성
  <조건2> GridSearch model을 이용하여 best score와 best parameters 구하기  
"""

from sklearn.svm import SVC # svm model 
from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score # 평가 

# 1. dataset load 
X, y = load_iris(return_X_y= True)
X.shape # (569, 30)


# 2. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. 비선형 SVM 모델 : kernel='rbf'
rbf_model = None


# 4. 선형 SVM 모델 : kernel='linear' 
linear_model = None



# 5. Grid Search : 분류정확도가 낮은 model 대상, 5겹 교차검정 수행 









