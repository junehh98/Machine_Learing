'''
 문4) iris dataset을 이용하여 다음과 같은 단계로 XGBoost model을 생성하시오.
'''

import pandas as pd # file read
from xgboost import XGBClassifier # model 생성 
from xgboost import plot_importance # 중요변수 시각화  
import matplotlib.pyplot as plt # 중요변수 시각화 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report # model 평가 


# 단계1 : data set load 
iris = pd.read_csv("C:/ITWILL/5_Python_ML/data/iris.csv")
iris.info()


# 변수명 추출 
cols=list(iris.columns)
col_x=cols[:4] # x변수명 
col_y=cols[-1] # y변수명 


iris.Species.unique()
# array(['setosa', 'versicolor', 'virginica'], dtype=object) -> class가 3개임 



# 단계2 : 훈련/검정 데이터셋 생성
train_set, test_set = train_test_split(iris, test_size=0.25)
train_set.shape # (112, 5) = X + y
test_set.shape # (38, 5)

#  4   Species       150 non-null    object 

from sklearn.preprocessing import LabelEncoder

# y변수 라벨인코딩 
y_train = LabelEncoder().fit_transform(train_set)




# 단계3 : model 생성 : train data 이용
xgb = XGBClassifier(objective='multi:softprob')
model = xgb.fit(train_set, test_set, eval_metric='merror') 


# 단계4 :예측치 생성 : test data 이용  


# 단계5 : 중요변수 확인 & 시각화  


# 단계6 : model 평가 : confusion matrix, accuracy, report
