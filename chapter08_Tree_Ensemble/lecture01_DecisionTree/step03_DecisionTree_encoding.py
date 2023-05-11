# -*- coding: utf-8 -*-
"""
 Label Encoding
 - 일반적으로 y변수(대상변수)를 대상으로 인코딩 
 - 트리 계열 모델(의사결정트리, 랜덤포레스트)에서 x변수에 적용
"""

import pandas as pd  
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder # 인코딩 도구 
import matplotlib.pyplot as plt # 중요변수 시각화 

# 1. 화장품 데이터(skin.csv) 가져오기 
df = pd.read_csv(r"C:\ITWILL\5_Python_ML\data\skin.csv")
df.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 제외 
 1   gender       30 non-null     object -> x변수  
 2   age          30 non-null     int64  -> x변수 
 3   job          30 non-null     object -> x변수
 4   marry        30 non-null     object -> x변수
 5   car          30 non-null     object -> x변수
 6   cupon_react  30 non-null     object -> y변수(화장품 구입여부) 
'''


#  각 변수 범주 확인 함수  
'''
def category(df, col_names) : # (데이터프레임, 칼럼명)
    for name in col_names :
        print('{0} -> {1}'.format(name, df[name].unique()))
'''

 
# 2. X, y변수 선택 
X = df.drop(['cust_no', 'cupon_react'], axis = 1) # 사용한 변수와 y변수 제거 
y = df['cupon_react'] 

label_encoder = LabelEncoder()
X_encoded = X.copy()
for col in X.columns:
    if X[col].dtype == 'object':
        X_encoded[col] = label_encoder.fit_transform(X[col])





                                                                                     
# 3.훈련 데이터 75, 테스트 데이터 25으로 나눈다. 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 123)

# 4. DecisionTree 분류기 
model = DecisionTreeClassifier().fit(X_train, y_train) # ValueError
# ValueError: could not convert string to float: 'male'


# 5. 중요 변수 
print("중요도 : \n{}".format(model.feature_importances_))


x_size = X.shape[1] # x변수 개수 = 5
x_names = list(X.columns) # x변수명 추출 : 중요변수 시각화에 이용
 
# 중요변수 시각화 : 가로막대 차트 
plt.barh(range(x_size), model.feature_importances_) 
plt.yticks(range(x_size), x_names) # y축 눈금 : x변수명 적용  
plt.xlabel('feature_importances')
plt.show()



# 6. 모델 평가  
y_pred= model.predict(X_test) # 예측치

# 분류정확도 
accuracy = accuracy_score( y_test, y_pred)
print( accuracy) # 0.875 

# 혼동행렬
conf_matrix= confusion_matrix(y_test, y_pred)
print(conf_matrix)    

# 정밀도 , 재현율, f1 score 확인 
report = classification_report(y_test, y_pred)
print(report)


