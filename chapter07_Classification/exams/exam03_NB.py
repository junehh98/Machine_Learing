'''
문3) 날씨 데이터(weather)을 이용하여 각 단계별로 Naive Bayes 모델을 생성하시오
'''

import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # 평가 


weather = pd.read_csv(r'C:\ITWILL\5_Python_ML\data\weatherAUS.csv')
print(weather.head())
print(weather.info())


# 단계1. 결측치(NaN)을 가진 모든 행(row) 제거 후 data 저장    
weather.isnull().sum()


data = weather.dropna(axis = 0)
data.isnull().sum()

weather.shape # (36881, 24)
data.shape # (17378, 24)



# 단계2. data에서 1,2,8,10,11,22,23 칼럼을 제외한 나머지 new_data 저장 
cols = list(data.columns) # 전체 칼럼이름 반환  
cols

colnames = [] # 사용할 칼럼 저장 

for i in range(len(cols)) : # 전체 칼럼 수 만큼 반복 
    if i not in [0,1,7,9,10,21,22] : # 해당 칼럼 제외 
        colnames.append(cols[i]) 
    
new_data = data[colnames]



# 모델에서 사용할 전체 관측치 36,881개와 변수 17개  
print(new_data.info()) 
'''
RangeIndex: 36881 entries, 0 to 36880
Data columns (total 17 columns):
 0   MinTemp        36543 non-null  float64
 1   MaxTemp        36639 non-null  float64
 2   Rainfall       36255 non-null  float64
 3   Evaporation    24035 non-null  float64
 4   Sunshine       23317 non-null  float64
 5   WindGustSpeed  33520 non-null  float64
 6   WindSpeed9am   36219 non-null  float64
 7   WindSpeed3pm   36235 non-null  float64
 8   Humidity9am    36311 non-null  float64
 9   Humidity3pm    36370 non-null  float64
 10  Pressure9am    33309 non-null  float64
 11  Pressure3pm    33329 non-null  float64
 12  Cloud9am       24381 non-null  float64
 13  Cloud3pm       23899 non-null  float64
 14  Temp9am        36394 non-null  float64
 15  Temp3pm        36437 non-null  float64
 16  RainTomorrow   36261 non-null  object  : y변수 
'''
 

# 단계3. new_data에서 변수 선택(y변수 : RainTomorrow, x변수 : 나머지 16개)
y = new_data.RainTomorrow
X = new_data.drop('RainTomorrow', axis=1)



# 단계4. y변수 레이블 인코딩 
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
y = le.fit_transform(y)



# 단계5. 70:30 비율 train/test 데이터셋 구성
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=100)


# 단계6. GaussianNB 모델 생성 
gnb = GaussianNB() 
model = gnb.fit(X_train, y_train)


# 단계7. model 평가 : accuracy, confusion matrix, classification_report
y_pred = model.predict(X_test)

# accuracy
y_true = y_test
acc = accuracy_score(y_true, y_pred)
print('accuracy =', acc) # accuracy = 0.8097429996164174


# confision matrix
con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)
'''
[[3410  643]
 [ 349  812]]
'''

# classification_report
report = classification_report(y_true, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       0.91      0.84      0.87      4053
           1       0.56      0.70      0.62      1161

    accuracy                           0.81      5214
   macro avg       0.73      0.77      0.75      5214
weighted avg       0.83      0.81      0.82      5214
'''








