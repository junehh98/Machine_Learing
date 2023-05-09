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
data = None


# 단계2. data에서 1,2,8,10,11,22,23 칼럼을 제외한 나머지 new_data 저장 
cols = list(data.columns) # 전체 칼럼이름 반환  

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
y = None
X = None



# 단계4. y변수 레이블 인코딩 
from sklearn.preprocessing import LabelEncoder 



# 단계5. 70:30 비율 train/test 데이터셋 구성


# 단계6. GaussianNB 모델 생성 
  

# 단계7. model 평가 : accuracy, confusion matrix, classification_report



