# -*- coding: utf-8 -*-
"""
문1) 주어진 자료를 대상으로 조건에 맞게 단계별로 로지스틱 회귀모델(이항분류)를 생성하시오.  
    조건1> cust_no, cor 변수 제거 
    조건2> object형 X변수 : OneHotEncoding(k-1개)
    조건3> object형 y변수 : LabelEncoding 
    조건4> 모델 평가 : 혼돈행렬과 분류정확도
    조건5> ROC curve 시각화  
"""

from sklearn.linear_model import LogisticRegression # model 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import confusion_matrix, accuracy_score # model 평가 
from sklearn.preprocessing import LabelEncoder # 레이블 인코딩 도구 

import pandas as pd # pd.get_dummies() : 원-핫 인코딩 도구 

df = pd.read_csv(r"C:\ITWILL\5_Python_ML\data\skin.csv")
df.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 변수 제외 
 1   gender       30 non-null     object -> x변수(성별) 
 2   age          30 non-null     int64  -> x변수(나이)
 3   job          30 non-null     object -> x변수(직업유무)
 4   marry        30 non-null     object -> x변수(결혼여부)
 5   car          30 non-null     object -> 변수 제외 
 6   cupon_react  30 non-null     object -> y변수(쿠폰 반응) 
''' 


# 단계1. 변수 제거 : cust_no, car
new_df = df.drop(labels=['cust_no', 'car'], axis=1) # 칼럼 제거 
new_df.shape # (30, 5)


# 단계2. object형 변수 인코딩  

# 1) X변수 OneHotEncoding : k-1개 가변수 만들기 gender, job, marry 변수
X = pd.get_dummies(data = new_df, 
                   columns=['gender', 'job', 'marry'], # 대상 칼럼 
                   drop_first = True) # k-1개 더미변수 
X.info()
'''
 0   age          30 non-null     int64 
 1   cupon_react  30 non-null     object : y변수 제거 
 2   gender_male  30 non-null     uint8 
 3   job_YES      30 non-null     uint8 
 4   marry_YES    30 non-null     uint8
'''

# y변수 제거 후 현재 객체 반영 
X.drop(labels='cupon_react', axis=1, inplace=True)

 
# 2) y변수 LabelEncoding : cupon_react 변수
y = LabelEncoder().fit_transform(y = new_df['cupon_react'])
y # 0 or 1 

# 단계3. train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=34)


# 단계4. model 생성 : 기본 파라미터 이용 
model = LogisticRegression().fit(X=X_train, y=y_train)


# 단계5. model 평가 
y_pred = model.predict(X = X_test)

# 1) 혼동행렬 
con_mat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(con_mat)
'''
[[4 0]
 [1 4]]
'''

# 2) 분류정확도 
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print(acc) # 0.888888


# 단계6. ROC curve 시각화
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 

# y 확률 예측 
y_pred_prob = model.predict_proba(X = X_test)
y_pred_prob = y_pred_prob[:, 1] # class 1 확률 예측치 


fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred_prob)
fpr # [0.  , 0.  , 0.  , 0.75, 0.75, 1.  ]
tpr # [0. , 0.4, 0.8, 0.8, 1. , 1. ]

# ROC curve 시각화 
plt.plot(fpr, tpr, color = 'red', marker='o',
         label='ROC curve')
plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='AUC')
plt.legend()
plt.show()






