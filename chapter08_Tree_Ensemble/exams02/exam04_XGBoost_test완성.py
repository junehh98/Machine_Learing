'''
 문4) iris dataset을 이용하여 다음과 같은 단계로 XGBoost model을 생성하시오.
'''

import pandas as pd # file read
from xgboost import XGBClassifier # model 생성 
from xgboost import plot_importance # 중요변수 시각화  
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report # model 평가 


# 단계1 : data set load 
iris = pd.read_csv("C:/ITWILL/5_Python_ML/data/iris.csv")

# 변수명 추출 
cols=list(iris.columns)
col_x=cols[:4] # x변수명 : ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
col_y=cols[-1] # y변수명 : 'Species'


# 단계2 : 훈련/검정 데이터셋 생성
train_set, test_set = train_test_split(iris, 
                                       test_size=0.25)
train_set.shape # (112, 5) = X + y
test_set.shape # (38, 5)

from sklearn.preprocessing import LabelEncoder

# y변수 레이블 인코딩 : 문자열 -> 10진수 변환 
y_train =LabelEncoder().fit_transform(train_set[col_y])
y_test =LabelEncoder().fit_transform(test_set[col_y])

# 단계3 : model 생성 : train data 이용
model = XGBClassifier().fit(X = train_set[col_x], y=y_train)
dir(model)
print(model)

# 단계4 :예측치 생성 : test data 이용  
y_pred = model.predict(test_set[col_x])

# 단계5 : 중요변수 확인 & 시각화  
plot_importance(model)
# 중요변수 : Petal.Length 

# 단계6 : model 평가 : confusion matrix, accuracy, report
print(confusion_matrix(y_test, y_pred))
'''
[[ 9  0  0]
 [ 0  9  1]
 [ 0  2 17]]
'''
print(accuracy_score(y_test, y_pred))
# 0.9473684210526315
print(classification_report(y_test, y_pred))
'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         9
           1       0.82      0.90      0.86        10
           2       0.94      0.89      0.92        19

    accuracy                           0.92        38
   macro avg       0.92      0.93      0.93        38
weighted avg       0.92      0.92      0.92        38
'''
