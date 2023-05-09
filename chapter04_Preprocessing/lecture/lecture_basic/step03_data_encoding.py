<<<<<<<< HEAD:workspace/chapter04_Preprocessing/lecture/lecture_basic/step03_data_encoding.py
######################################
### 3. 데이터 인코딩 
######################################

"""
데이터 인코딩 : 머신러닝 모델에서 범주형변수를 대상으로 숫자형의 목록으로 변환해주는 전처리 작업
 - 방법 : 레이블 인코딩(label encoding), 원-핫 인코딩(one-hot encoding)   
 - 레이블 인코딩(label encoding) : 트리계열모형(의사결정트리, 앙상블)의 변수 대상 
 - 원-핫 인코딩(one-hot encoding) : 회귀계열모형(선형,로지스틱,SVM,신경망)의 변수 대상 
   -> 회귀모형에서는 인코딩값이 가중치로 적용되므로 원-핫 인코딩(더미변수)으로 변환  
"""


import pandas as pd 

data = pd.read_csv(r"C:\ITWILL\5_Python_ML\data\skin.csv")
data.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 변수 제외 
 1   gender       30 non-null     object -> x변수 
 2   age          30 non-null     int64 
 3   job          30 non-null     object
 4   marry        30 non-null     object
 5   car          30 non-null     object
 6   cupon_react  30 non-null     object -> y변수(쿠폰 반응) 
''' 


## 1. 변수 제거 : cust_no
df = data.drop('cust_no', axis = 1)
df.info() # cust_no 삭제됨 


### 2. 레이블 인코딩 : 트리모델 계열의 x, y변수 인코딩  
from sklearn.preprocessing import LabelEncoder # 인코딩 도구 


# 1) 쿠폰 반응 범주 확인 
df.cupon_react.unique() # array(['NO', 'YES'], dtype=object) 


# 2) 인코딩
encoder = LabelEncoder()
dir(encoder)

# fit() + transform() 
label = encoder.fit_transform(df['cupon_react']) # data 반영 
label # 0과 1로 인코딩 확인 
'''
array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 1])
'''


# 3) 칼럼 추가 
df['label'] = label



### 3. 원-핫 인코딩 : 회귀모델 계열의 x변수 인코딩  

# 1) k개 목록으로 가변수(더미변수) 만들기 
df_dummy = pd.get_dummies(data=df) # object형 기준변수 포함 
df_dummy # age  gender_female  gender_male  ...  marry_YES  car_NO  car_YES -> (1+8)
df_dummy.info()



# 2) 특정 변수 선택   
df_dummy2 = pd.get_dummies(data=df, columns=['label','gender','job','marry'])
df_dummy2.info() # Data columns (total 12 columns):
'''
-- NO, YES 컬럼 추가됨 -- 
4   job_NO           30 non-null     uint8
5   job_YES          30 non-null     uint8
6   marry_NO         30 non-null     uint8
7   marry_YES        30 non-null     uint8
8   car_NO           30 non-null     uint8
9   car_YES          30 non-null     uint8
10  cupon_react_NO   30 non-null     uint8
11  cupon_react_YES  30 non-null     uint8
'''
    

# 3) k-1개 목록으로 가변수(더미변수) 만들기   
df_dummy3 = pd.get_dummies(data=df, drop_first=True) # 기준변수 제외(권장)
df_dummy3  




###############################
## 가변수 기준(base) 변경하기  
###############################
import seaborn as sn 
iris = sn.load_dataset('iris')


# 1. 가변수(dummy) : k-1개 
iris_dummy = pd.get_dummies(data = iris, columns=['species'], drop_first=True)
# drop_first=True : 첫번째 범주 제외(기준변수)
iris_dummy.info()
iris_dummy[['species_versicolor','species_virginica']]


# 2. base 기준 변경 : 범주 순서변경('virginica' -> 'versicolor' -> 'setosa') 
# object -> category 변환 (그래야 범주의 순서 변경 가능함 )
iris['species'] = iris['species'].astype('category')
iris['species'].dtype

# CategoricalDtype(categories=['setosa', 'versicolor', 'virginica'], ordered=False)


iris['species'] = iris['species'].cat.set_categories(['virginica','versicolor','setosa'])


# 3. 가변수(dummy) : k-1개 
iris_dummy2 = pd.get_dummies(data=iris, columns=['species'], drop_first=True)

iris_dummy2.info()
iris_dummy2[['species_versicolor', 'species_setosa']]





















========
######################################
### 3. 데이터 인코딩 
######################################

"""
데이터 인코딩 : 머신러닝 모델에서 범주형변수를 대상으로 숫자형의 목록으로 변환해주는 전처리 작업
 - 방법 : 레이블 인코딩(label encoding), 원-핫 인코딩(one-hot encoding)   
 - 레이블 인코딩(label encoding) : 트리계열모형(의사결정트리, 앙상블)의 변수 대상 
 - 원-핫 인코딩(one-hot encoding) : 회귀계열모형(선형,로지스틱,SVM,신경망)의 변수 대상 
   -> 회귀모형에서는 인코딩값이 가중치로 적용되므로 원-핫 인코딩(더미변수)으로 변환  
"""


import pandas as pd 

data = pd.read_csv(r"C:\ITWILL\5_Python_ML\data\skin.csv")
data.info()
'''
RangeIndex: 30 entries, 0 to 29
Data columns (total 7 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   cust_no      30 non-null     int64  -> 변수 제외 
 1   gender       30 non-null     object -> x변수 
 2   age          30 non-null     int64 
 3   job          30 non-null     object
 4   marry        30 non-null     object
 5   car          30 non-null     object
 6   cupon_react  30 non-null     object -> y변수(쿠폰 반응) 
''' 


## 1. 변수 제거 : cust_no
df = data.drop('cust_no', axis = 1)
df.info() # cust_no 삭제됨 


### 2. 레이블 인코딩 : 트리모델 계열의 x, y변수 인코딩  
from sklearn.preprocessing import LabelEncoder # 인코딩 도구 


# 1) 쿠폰 반응 범주 확인 
df.cupon_react.unique() # array(['NO', 'YES'], dtype=object) 


# 2) 인코딩
encoder = LabelEncoder()
dir(encoder)

# fit() + transform() 
label = encoder.fit_transform(df['cupon_react']) # data 반영 
label # 0과 1로 인코딩 확인 
'''
array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 1])
'''


# 3) 칼럼 추가 
df['label'] = label



### 3. 원-핫 인코딩 : 회귀모델 계열의 x변수 인코딩  

# 1) k개 목록으로 가변수(더미변수) 만들기 
df_dummy = pd.get_dummies(data=df) # object형 기준변수 포함 
df_dummy # age  gender_female  gender_male  ...  marry_YES  car_NO  car_YES -> (1+8)
df_dummy.info()



# 2) 특정 변수 선택   
df_dummy2 = pd.get_dummies(data=df, columns=['label','gender','job','marry'])
df_dummy2.info() # Data columns (total 12 columns):
'''
-- NO, YES 컬럼 추가됨 -- 
4   job_NO           30 non-null     uint8
5   job_YES          30 non-null     uint8
6   marry_NO         30 non-null     uint8
7   marry_YES        30 non-null     uint8
8   car_NO           30 non-null     uint8
9   car_YES          30 non-null     uint8
10  cupon_react_NO   30 non-null     uint8
11  cupon_react_YES  30 non-null     uint8
'''
    

# 3) k-1개 목록으로 가변수(더미변수) 만들기   
df_dummy3 = pd.get_dummies(data=df, drop_first=True) # 기준변수 제외(권장)
df_dummy3  




###############################
## 가변수 기준(base) 변경하기  
###############################
import seaborn as sn 
iris = sn.load_dataset('iris')


# 1. 가변수(dummy) : k-1개 
iris_dummy = pd.get_dummies(data = iris, columns=['species'], drop_first=True)
# drop_first=True : 첫번째 범주 제외(기준변수)
iris_dummy.info()
iris_dummy[['species_versicolor','species_virginica']]


# 2. base 기준 변경 : 범주 순서변경('virginica' -> 'versicolor' -> 'setosa') 
# object -> category 변환 (그래야 범주의 순서 변경 가능함 )
iris['species'] = iris['species'].astype('category')
iris['species'].dtype

# CategoricalDtype(categories=['setosa', 'versicolor', 'virginica'], ordered=False)


iris['species'] = iris['species'].cat.set_categories(['virginica','versicolor','setosa'])


# 3. 가변수(dummy) : k-1개 
iris_dummy2 = pd.get_dummies(data=iris, columns=['species'], drop_first=True)

iris_dummy2.info()
iris_dummy2[['species_versicolor', 'species_setosa']]





















>>>>>>>> origin/main:chapter04_Preprocessing/lecture/lecture_basic/step03_data_encoding.py
