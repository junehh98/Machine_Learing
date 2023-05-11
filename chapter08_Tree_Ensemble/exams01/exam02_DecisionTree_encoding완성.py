# -*- coding: utf-8 -*-
'''
 문2) 다음 데이터 셋을 이용하여 단계별로 Decision Tree 모델을 생성하시오.
'''

import pandas as pd # csv file read
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.tree import DecisionTreeClassifier # model
from sklearn.metrics import accuracy_score, confusion_matrix # model evaluation


path = r'C:\ITWILL\5_Python_ML\data'
data = pd.read_csv(path +'/dataset.csv') 
data.info()
'''
RangeIndex: 217 entries, 0 to 216
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   resident  217 non-null    int64    : x변수(1~5) -> (0~4)  
 1   gender    217 non-null    int64    : x변수 
 2   job       205 non-null    float64  : x변수 
 3   age       217 non-null    int64    : x변수 
 4   position  208 non-null    float64  : y변수 
 5   price     217 non-null    float64  : x변수 
 6   survey    217 non-null    int64    : x변수  
'''
 

# 단계1 : 결측치를 포함한 모든 행 제거 후 new_data 생성 
data.isnull().sum()
'''
job         12
position     9
'''

# job, postion 칼럼 기준 결측치 행 제거 
new_data = data.dropna(axis=0, subset=['job','position'])
new_data.shape # (198, 7)

217 - 198 # 19개 행 제거 
 
# 단계2 : resident, gender, job, position변수 대상  레이블인코딩
from sklearn.preprocessing import LabelEncoder # 인코딩 도구
new_data['resident'] = LabelEncoder().fit_transform(new_data['resident'])
new_data['gender'] = LabelEncoder().fit_transform(new_data['gender'])
new_data['job'] = LabelEncoder().fit_transform(new_data['job'])
new_data['position'] = LabelEncoder().fit_transform(new_data['position'])


# 단계3 : new_data에서 X, y변수 선택 
X = new_data[['resident','gender','job','age','price','survey']] # 'resident','gender','job','age','price','survey'
y = new_data['position'] # 'position'

# 직급(position) 범주 
y.unique() # [3, 0, 2, 4, 1]


# 단계4 : 훈련 데이터 75, 테스트 데이터 25 나누기 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 123)



# 단계5 : 기본모델 만들기 
model = DecisionTreeClassifier(random_state=123).fit(X_train, y_train)



# 단계6. 기본모델 평가 : 분류정확도, 혼동행렬
y_pred = model.predict(X_test)  

accuracy = accuracy_score(y_test, y_pred) # 분류정확도
print(accuracy) # 0.86

conf_mat = confusion_matrix(y_test, y_pred) # 혼동 행렬
print(conf_mat)
'''
[[16  0  0  0  3]
 [ 0  9  0  0  1]
 [ 0  0  7  0  0]
 [ 0  1  0  5  0]
 [ 0  2  0  0  6]]
'''

# chapter07 > lecture01 > step06_GridSearch.py 참고 

# 단계7 : best parameter 찾기 : 기본모델을 대상으로 최적의 파라미터 찾기  
from sklearn.model_selection import GridSearchCV # best parameters

# Decision Tree 주요 파라미터 
parmas = {'criterion' : ['gini', 'entropy'], # 중요변수 선택 
          'max_depth' : [None, 3, 4, 5, 6],  # 트리 깊이 
          'min_samples_split': [2, 3, 4]}  # 내부노드 분할 최소 샘플 수 

# 1) 기본 model과 parmas을 이용하여 5겹 교차검정으로 Grid model 만들기 
grid_model = GridSearchCV(estimator=model, 
                          param_grid=parmas,
                          scoring='accuracy', 
                          cv=5, n_jobs=-1).fit(X, y)# full data 적용
'''
estimator : 기본 model
param_grid : 대상 파라미터 
scoring : model 평가방법 
cv : k겹 교차검정
n_jobs : 사용할 cpu 수 
verbose : 학습과정 출력 여부(1:출력, 0:미출력)
'''

# 2) Best score 확인 
print('best score =', grid_model.best_score_)
# best score = 0.9294871794871795

# 3) Best parameters 확인  
print('best parameters =', grid_model.best_params_)
# best parameters = {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 2}

# 4) best parameters 적용 : new model 생성 & 평가  
obj = DecisionTreeClassifier(criterion='gini',
                             max_depth=4,
                             random_state=123,
                             min_samples_split=2) # best parameters 적용 
new_model = obj.fit(X=X_train, y=y_train) 


# train/test score 확인 : 과적합 여부 확인 
train_score = new_model.score(X=X_train, y=y_train)
print(train_score) # 0.9662162162162162

test_score = new_model.score(X=X_test, y=y_test)
print(test_score) # 0.9

















