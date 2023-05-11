'''
step02_DecisionTree_parameter.py

DecisionTreeClassifier 관련 문서 : 
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

DecisionTree Hyper parameter 
'''
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# tree 시각화 
from sklearn.tree import export_graphviz
from graphviz import Source  

############################
### Hyper parameter 
############################
iris = load_iris()
x_names = iris.feature_names # x변수 이름 
labels = iris.target_names # ['setosa', 'versicolor', 'virginica']

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123)

'''
criterion='gini' : 중요변수 선정 기준, 
 -> criterion : {"gini", "entropy"}, default="gini"
splitter='best' : 각 노드에서 분할을 선택하는 데 사용되는 전략, 
max_depth=None : tree 최대 깊이, 
 -> max_depth : int, default=None
 -> max_depth=None : min_samples_split의 샘플 수 보다 적을 때 까지 tree 깊이 생성 
 -> 과적합 제어 역할 : 값이 클 수록 과대적합, 적을 수록 과소적합 
min_samples_split=2 : 내부 노드를 분할하는 데 필요한 최소 샘플 수(기본 2개)
 -> int or float, default=2    
 -> 과적합 제어 역할 : 값이 클 수록 과소적합, 적을 수록 과대적합 
'''

# model : default parameter
tree = DecisionTreeClassifier(criterion='gini',
                               random_state=123, 
                               max_depth=4, # None -> 4
                               min_samples_split=2)
# 가지치기 : max_depth=None -> max_depth=4

model = tree.fit(X=X_train, y=y_train) # 훈련셋 
X_train.shape # (n=105, 4)

model.get_depth() # 5 -> 4

# tree 시각화 
graph = export_graphviz(model,
                out_file="tree_graph.dot",
                feature_names=x_names,
                class_names=labels,
                rounded=True,
                impurity=True,
                filled=True)


# dot file load 
file = open("tree_graph.dot") 
dot_graph = file.read()

# tree 시각화 : Console 출력  
Source(dot_graph) 

# 과적합(overfitting) 유무 확인  
model.score(X=X_train, y=y_train)
model.score(X=X_test, y=y_test)
'''
0.9904761904761905
0.9333333333333333
'''










