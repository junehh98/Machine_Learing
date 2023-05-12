# -*- coding: utf-8 -*-
"""
RandomForest 앙상블 모델 
"""

from sklearn.ensemble import RandomForestClassifier # model 
from sklearn.datasets import load_wine # dataset 
# 평가 도구 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# 1. dataset load
wine = load_wine()

X, y = wine.data, wine.target
X.shape # (178, 13)
y # class3(0 ~ 2)

# y를 분류하는 X변수의 개수 
import numpy as np 
np.sqrt(X.shape[1]) # 3.605551275463989 : 3 ~ 4 



# 2. model 생성 
'''
주요 hyper parameter(default)
 n_estimators=100 : tree 개수 
 criterion='gini' : 중요변수 선정 기준 
 max_depth=None : 트리 깊이 
 min_samples_split=2 : 내부 노드 분할에 필요한 최소 샘플 수
'''

# full dataset 적용 
model = RandomForestClassifier(random_state=123).fit(X = X, y = y) 

dir(model)



# 3. test set 만들기 
import numpy as np

# 178 -> 100개 선정 
idx = np.random.choice(a=len(X), size=100, replace=False)
X_test, y_test = X[idx], y[idx]



# 4. model 평가 
y_pred = model.predict(X = X_test)

# 혼동행렬
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)

# 분류정확도 
print(accuracy_score(y_test, y_pred))

# 분류 리포트 
print(classification_report(y_test, y_pred))



# 5. 중요변수 시각화 
print('중요도 : ', model.feature_importances_)
'''
 중요도 :  [0.10968347 0.0327231  0.0122728  0.02942002 0.02879057 0.05790394
 0.13386676 0.00822129 0.02703471 0.158535   0.07640667 0.13842681
 0.18671486]
'''

x_names = wine.feature_names # x변수 이름 
# ['alcohol',  ... 'proline']

x_size = len(x_names) # x변수 개수  

import matplotlib.pyplot as plt 

# 가로막대 차트 
plt.barh(range(x_size), model.feature_importances_) # (y, x)
plt.yticks(range(x_size), x_names)   
plt.xlabel('feature_importances') 
plt.show()


### 가로막대 차트 : 중요점수 순으로 정렬

# 중요점수 오름차순 정렬 
importances_sorted = sorted(model.feature_importances_)

# 중요점수 오름차순 색인 정렬 
idx = model.feature_importances_.argsort()
idx # [ 7,  2,  8,  4,  3,  1,  5, 10,  0,  6, 11,  9, 12]

# numpy array 변환 
x_names = np.array(x_names)

# X변수 중요점수 순으로 정렬 
sorted_x_names = x_names[idx]

 
plt.barh(range(x_size), importances_sorted) # (y, x)
plt.yticks(range(x_size), sorted_x_names)   
plt.xlabel('feature_importances') 
plt.show()



