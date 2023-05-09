"""
기사 실기시험 유형 

문3) 다음 타이타닉(titanic) 데이터셋으로 로지스틱회귀모델을 만들고, 모델을 평가하시오.
     <작업절차>
      1. titanic_train.csv의 자료를 전처리한 후 75%을 이용하여 모델을 학습하고,
         25%로 학습된 모델을 검증한다.(검증 대상 : 모델 성능, 과적합 여부)
      2. 검증된 모델에 titanic_test.csv의 X변수를 적용하여 예측치를 구한다.
      
    훈련셋/검증셋 : titanic_train.csv(X, y변수 포함)  
    평가셋 : titanic_test.csv(X변수만 포함) 
"""

import pandas as pd
pd.set_option('display.max_columns',15) # 최대 15개 칼럼 출력 
from sklearn.linear_model import LogisticRegression # class - model
from sklearn.preprocessing import StandardScaler # 표준화
from sklearn.model_selection import train_test_split # split


### 1단계 : train.csv 가져오기 
train = pd.read_csv(r"/Users/junehh98/Desktop/itwill/5_Python_ML/data/titanic_train.csv")
train.info()
'''
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 0   PassengerId  891 non-null    int64   -> 제거 
 1   Survived     891 non-null    int64   -> y변수 
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object  -> 제거
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object  -> 제거
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object  
 11  Embarked     889 non-null    object
'''


### 2단계 : 불필요한 변수('PassengerId','Name','Ticket') 제거 후 new_df 만들기 
new_df = train.drop(['PassengerId','Name','Ticket'], axis=1)
new_df.columns


### 3단계 : 결측치 확인 및 처리 

# 1) 결측치(NaN) 확인
df1 = new_df.isnull().sum()
'''
Survived      0
Pclass        0
Sex           0
Age         177
SibSp         0
Parch         0
Fare          0
Cabin       687
Embarked      2
'''


# 2) 결측치가 50% 이상 칼럼 제거 후 new_df에 반영     

# 결측치가 50% 이상인 열 추출
columns_to_drop = df1[df1 / len(new_df) >= 0.5].index

# 열 제거 후 new_df에 반영
new_df = new_df.drop(columns=columns_to_drop)
new_df.columns



# 3) Age 칼럼의 결측치를 평균으로 대체 후 new_df에 반영 
new_df['Age'] = new_df['Age'].fillna(new_df['Age'].mean())

new_df.isnull().sum()



# 4) Embarked 칼럼의 결측치를 가장 많이 출현한 값으로 대체 후 new_df 적용
# 결측치 최빈값으로 대체
new_df['Embarked'] = new_df['Embarked'].fillna(new_df['Embarked'].mode().iloc[0])




### 단계4 : X변수, y변수 선정   

# 1) X변수 만들기 : new_df에서 'Survived' 칼럼을 제외한 나머지 칼럼 이용  
X = new_df[~new_df == 'Survived'] # 독립변수

# 2) y변수 만들기  : new_df에서 'Survived' 칼럼 이용   
y = None # 종속변수



### 단계5 : X변수 전처리     

# 1) object형 변수를 대상으로 k-1개 가변수 만들기

 
# 2) X변수 전체를 대상으로 표준화 방식으로 스케일링  



### 단계6 : 훈련셋(train)/검증셋(val) split(75% : 25%) 
X_train, y_train, X_val, y_val = train_test_split()



### 단계7 : 로지스틱회귀모델 생성 & 결정계수 model 평가 

# 1) 회귀모델 생성 : 훈련셋 이용(X_train, y_train)

# 2) 회귀모델 검증 : 검증셋 이용(X_val, y_val) 



### 8단계 테스트셋(titanic_test.csv)으로 model 평가 
test = pd.read_csv(r"C:\ITWILL\5_Python_ML\data\titanic_test.csv")
test.info() # y변수 없음 
'''
 0   PassengerId  418 non-null    int64   - 제거 
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object  - 제거 
 3   Sex          418 non-null    object  - 더미변수 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object  - 제거 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object  - 제거 
 10  Embarked     418 non-null    object  - 더미변수 
'''
 
### 1. 불필요한 변수('PassengerId','Name','Ticket','Cabin') 제거 new_test 만들기 


### 2. 결측치 확인 및 평균 대체  

# 1) 결측치(NaN) 확인


# 2) 모든 결측치 평균으로 대체  



### 3. X변수 전처리     

# 1) object형 변수 대상으로 k-1개 가변수 만들기

# 2) 표준화 방식으로 스케일링  


### 4. 학습된 모델(model) X변수를 적용하여 예측치 만들기       

