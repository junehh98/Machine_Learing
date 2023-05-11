'''
step01_DecisionTree.py

Decision Tree 모델 
 - 중요변수 선택 기준 : GINI, Entropy
 - 의사결정트리 시각화 
"""

'''
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier # 분류기 
from sklearn.metrics import accuracy_score # model 평가 
# tree 시각화 
from sklearn.tree import plot_tree, export_graphviz # 시각화도구 
from graphviz import Source # 외부시각화도구 : pip install graphviz

# 1. dataset load 
path = 'c:/ITWILL/5_Python_ML/data'
dataset = pd.read_csv(path + "/tree_data.csv")
print(dataset.info())
'''
iq         6 non-null int64 - iq수치
age        6 non-null int64 - 나이
income     6 non-null int64 - 수입
owner      6 non-null int64 - 사업가 유무
unidegree  6 non-null int64 - 학위 유무
smoking    6 non-null int64 - 흡연 유무 - y변수 
'''

# 2. 변수 선택 
cols = list(dataset.columns)
X = dataset[cols[:-1]]
y = dataset[cols[-1]]

# 3. model & 평가 : gini계수 기준, 트리깊이=3
model = DecisionTreeClassifier(criterion='gini',
                               max_depth=3,
                               random_state=123).fit(X, y)
'''
criterion : {"gini", "entropy"}, default="gini" : 중요변수 선정 기준 
max_depth : int, default=None
'''
dir(model)
model.get_depth() # 3

y_pred = model.predict(X)
print(y_pred) 

acc = accuracy_score(y, y_pred)
print(acc) 


# 4. tree 시각화 
feature_names = cols[:-1]  # x변수 
plot_tree(model, feature_names = feature_names) # Plots 출력 
'''
 level1 : 중요변수 income
   left 분류조건 : income 24이하(no : 2) 
   right 분류조건 : income 24이상(no : 1, yes : 3) 
 level2 : 중요변수 iq 
   left 분류조건 :   incom 24 이상 & iq 105이상(yes : 2) 
''' 

# 외부 파일 내보내기 & 콘솔 출력
class_names = ['no', 'yes'] # y변수 class 

# 그래프 설정
graph = export_graphviz(model,
                out_file="tree_graph.dot", # file명 
                feature_names = feature_names, # X변수명 
                class_names = class_names, # y 클래스명 
                rounded=True, # 모서리 라운드 
                impurity=True, # gini 계수 
                filled=True) # 색상 채우기 

# dot file load 
file = open("tree_graph.dot", mode = 'r') 
dot_graph = file.read()
  
Source(dot_graph) # tree 시각화 : Console 출력
'''
시각화 오류 해결법 
1. spyder restart
2. path 확인 
'''

