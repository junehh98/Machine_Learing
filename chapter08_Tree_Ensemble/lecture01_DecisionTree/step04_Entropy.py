# -*- coding: utf-8 -*-
"""
step04_Entropy.py

지니불순도(Gini-impurity), 엔트로피(Entropy)
  Tree model에서 중요변수 선정 기준
 확률 변수 간의 불확실성을 나타내는 수치
 무질서의 양의 척도, 작을 수록 불확실성이 낮다.

  지니불순도와 엔트로피 수식  
 Gini-impurity = sum(p * (1-p))
 Entropy = -sum(p * log(p))

  지니계수와 정보이득  
  gini_index = base - Gini-impurity # 0.72
  info_gain = base - Entropy
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier # model 
from sklearn.metrics import confusion_matrix # 평가 
# 시각화 도구 
from sklearn.tree import plot_tree, export_graphviz 
from graphviz import Source 

##########################
### dataset 적용 
##########################

# 1. data set 생성 함수
def createDataSet():
    dataSet = [[1, 1, 'yes'],
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1, 'no']]
    columns = ['dark_clouds','gust'] # X1,X2,label
    return dataSet, columns


# 함수 호출 
dataSet, columns = createDataSet()

# list -> numpy 
dataSet = np.array(dataSet)
dataSet.shape # (5, 3)
print(dataSet)
print(columns) # ['dark_clouds', 'gust']

# 변수 선택 
X = dataSet[:, :2]
y = dataSet[:, -1]

# 레이블 인코딩 : 'yes' = 1 or 'no' = 0
label = [1 if i == 'yes' else 0 for i in y] 


# model 생성 
obj = DecisionTreeClassifier(criterion='entropy')
model = obj.fit(X = X, y = label)

y_pred = model.predict(X)

# 혼동행렬 
con_mat = confusion_matrix(label, y_pred)
print(con_mat)

# tree 시각화 : Plots 출력  
plot_tree(model, feature_names=columns)

# tree graph 
export_graphviz(decision_tree=model, 
                out_file='tree_graph.dot',
                max_depth=3,
                feature_names=columns,
                class_names=True)

# # dot(화소) file load 
file = open("tree_graph.dot")
dot_graph = file.read()

# tree 시각화 : Console 출력  
Source(dot_graph)






















