'''
 문1) load_breast_cancer 데이터 셋을 이용하여 다음과 같이 Decision Tree 모델을 생성하시오.
  <조건1> 75:25비율 train/test 데이터 셋 구성 
  <조건2> y변수 : cancer.target, x변수 : cancer.data
  <조건3> tree 최대 깊이 : 5 
  <조건4> decision tree 시각화 & 중요변수 확인
'''
import pandas as pd 
from sklearn import model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
# tree 시각화 
from sklearn.tree import export_graphviz
from graphviz import Source 
import matplotlib.pyplot as plt



# 데이터 셋 load 
cancer = load_breast_cancer()
type(cancer)

cancer.feature_names
cancer.target_names


# <단계1> y변수 : cancer.target, x변수 : cancer.data 
y = cancer.target
X = cancer.data



# <단계2> 75:25비율 train/test 데이터 셋 구성
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123)


X_train.shape
X_test.shape



# <단계3> tree 최대 깊이 : 5
classifier= DecisionTreeClassifier(max_depth=5, random_state=123)
dir(classifier)

# 모델 학습 
model = classifier.fit(X=X_train, y=y_train) # 훈련셋 적용 


# 모델 평가
y_pred = model.predict(X = X_test)
print(y_pred) 

acc = accuracy_score(y_test, y_pred) # (관측치, 예측치)
print(acc) # 0.972027972027972



# <단계4> decision tree 시각화 & 중요변수 확인 

export_graphviz(decision_tree = model,  # model
                out_file='exam1_tree.dot', 
                feature_names=cancer.feature_names,
                class_names=cancer.target_names,
                filled=True)


file = open("exam1_tree.dot", mode = 'r') 
dot_graph = file.read()
  
Source(dot_graph) 









