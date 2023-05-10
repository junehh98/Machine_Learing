<<<<<<< HEAD
'''
회귀분석용 sklearn dataset 정리 
'''
from sklearn import datasets # dataset 제공 library

######################################
# 선형회귀분석에 적합한 데이터셋
######################################

# 1. 붓꽃(iris) : 회귀와 분류 모두 사용 
'''
붓꽃(iris) 데이터
- 붓꽃 데이터는 통계학자 피셔(R.A Fisher)의 붓꽃의 분류 연구에 기반한 데이터

• 타겟 변수 : y변수
세가지 붓꽃 종(species) : setosa, versicolor, virginica

•특징 변수(4) : x변수
꽃받침 길이(Sepal Length)
꽃받침 폭(Sepal Width)
꽃잎 길이(Petal Length)
꽃잎 폭(Petal Width)
'''
iris = datasets.load_iris() # dataset load 
print(iris) 
print(iris.DESCR) # dataset 설명제공 : 변수특징, 요약통계 


# X, y변수 선택 
iris_X = iris.data # x변수 
iris_y = iris.target # y변수

# 객체형과 모양확인 
print(type(iris_X))
print(type(iris_y))

print(iris_X.shape)
print(iris_y.shape)


# X변수명과 y변수 범주명 
print(iris.feature_names)# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names) # ['setosa' 'versicolor' 'virginica']
 

# DataFrame 변환  
import pandas as pd
iris_df = pd.DataFrame(iris_X, columns=iris.feature_names)

# y변수 추가 
iris_df['species'] = iris.target 
iris_df.head()
iris_df.info() 


# 차트 분석 : 각 특징별 타겟변수의 분포현황  
import seaborn as sn
import matplotlib.pyplot as plt

# 변수 간 산점도 : hue = 집단변수 : 집단별 색상 제공 
sn.pairplot(iris_df, hue="species")
plt.show() 




# 2. 당료병 데이터셋
'''
- 442명의 당뇨병 환자를 대상으로한 검사 결과를 나타내는 데이터

•타겟 변수 : y변수
1년 뒤 측정한 당료병 진행상태 정량적화 자료(연속형)

•특징 변수(10: 모두 정규화된 값) : x변수
age : 나이 (세)
sex : 성별 
bmi : 비만도지수
bp : 평균혈압(Average blood pressure)
S1 ~ S6: 기타 당료병에 영향을 미치는 요인들 
'''

diabetes = datasets.load_diabetes() # dataset load 
print(diabetes.DESCR) # 컬럼 설명, url
'''
:Target: Column 11 -> 1년기준으로 질병 진행상태를 정량적(연속형)으로 측정 
:Attribute Information: Age ~ S6
'''    

print(diabetes.feature_names) # X변수명 
#print(diabetes.target_names) # None : 연속형 변수 이름 없음 

# X, y변수 동시 선택 
X, y = datasets.load_diabetes(return_X_y=True)

print(X.shape) # (442, 10) 
print(y.shape) # (442,) 



# 3. boston 주택가격 
'''
- 1978 보스턴 주택 가격에 미치는 요인을 기록한 데이터 

• 타겟 변수 : x변수
보스턴 주택 가격: 506개 타운의 주택 가격 중앙값(단위 1,000 달러)

•특징 변수(13) : y변수
CRIM: 범죄율
INDUS: 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율
'''
boston = datasets.load_boston()
print(boston.DESCR)

boston_X = boston.data 
boston_y = boston.target

print(boston_X.shape) # (506, 13) 
print(boston_y.shape) # (506,) 


# 4. california 주택가격 
'''
•타겟 변수 : y변수
1990년 캘리포니아의 각 행정 구역 내 주택 가격의 중앙값

•특징 변수(8) : x변수
MedInc : 행정 구역 내 소득의 중앙값
HouseAge : 행정 구역 내 주택 연식의 중앙값
AveRooms : 평균 방 갯수
AveBedrms : 평균 침실 갯수
Population : 행정 구역 내 인구 수
AveOccup : 평균 자가 비율
Latitude : 해당 행정 구역의 위도
Longitude : 해당 행정 구역의 경도
'''
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
print(california.DESCR)

# X변수 -> DataFrame 변환 
cal_df = pd.DataFrame(california.data, columns=california.feature_names)
# y변수 추가 
cal_df["MEDV"] = california.target
cal_df.tail()
cal_df.info() 

=======
'''
회귀분석용 sklearn dataset 정리 
'''
from sklearn import datasets # dataset 제공 library

######################################
# 선형회귀분석에 적합한 데이터셋
######################################

# 1. 붓꽃(iris) : 회귀와 분류 모두 사용 
'''
붓꽃(iris) 데이터
- 붓꽃 데이터는 통계학자 피셔(R.A Fisher)의 붓꽃의 분류 연구에 기반한 데이터

• 타겟 변수 : y변수
세가지 붓꽃 종(species) : setosa, versicolor, virginica

•특징 변수(4) : x변수
꽃받침 길이(Sepal Length)
꽃받침 폭(Sepal Width)
꽃잎 길이(Petal Length)
꽃잎 폭(Petal Width)
'''
iris = datasets.load_iris() # dataset load 
print(iris) 
print(iris.DESCR) # dataset 설명제공 : 변수특징, 요약통계 


# X, y변수 선택 
iris_X = iris.data # x변수 
iris_y = iris.target # y변수

# 객체형과 모양확인 
print(type(iris_X))
print(type(iris_y))

print(iris_X.shape)
print(iris_y.shape)


# X변수명과 y변수 범주명 
print(iris.feature_names)# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names) # ['setosa' 'versicolor' 'virginica']
 

# DataFrame 변환  
import pandas as pd
iris_df = pd.DataFrame(iris_X, columns=iris.feature_names)

# y변수 추가 
iris_df['species'] = iris.target 
iris_df.head()
iris_df.info() 


# 차트 분석 : 각 특징별 타겟변수의 분포현황  
import seaborn as sn
import matplotlib.pyplot as plt

# 변수 간 산점도 : hue = 집단변수 : 집단별 색상 제공 
sn.pairplot(iris_df, hue="species")
plt.show() 




# 2. 당료병 데이터셋
'''
- 442명의 당뇨병 환자를 대상으로한 검사 결과를 나타내는 데이터

•타겟 변수 : y변수
1년 뒤 측정한 당료병 진행상태 정량적화 자료(연속형)

•특징 변수(10: 모두 정규화된 값) : x변수
age : 나이 (세)
sex : 성별 
bmi : 비만도지수
bp : 평균혈압(Average blood pressure)
S1 ~ S6: 기타 당료병에 영향을 미치는 요인들 
'''

diabetes = datasets.load_diabetes() # dataset load 
print(diabetes.DESCR) # 컬럼 설명, url
'''
:Target: Column 11 -> 1년기준으로 질병 진행상태를 정량적(연속형)으로 측정 
:Attribute Information: Age ~ S6
'''    

print(diabetes.feature_names) # X변수명 
#print(diabetes.target_names) # None : 연속형 변수 이름 없음 

# X, y변수 동시 선택 
X, y = datasets.load_diabetes(return_X_y=True)

print(X.shape) # (442, 10) 
print(y.shape) # (442,) 



# 3. boston 주택가격 
'''
- 1978 보스턴 주택 가격에 미치는 요인을 기록한 데이터 

• 타겟 변수 : x변수
보스턴 주택 가격: 506개 타운의 주택 가격 중앙값(단위 1,000 달러)

•특징 변수(13) : y변수
CRIM: 범죄율
INDUS: 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율
'''
boston = datasets.load_boston()
print(boston.DESCR)

boston_X = boston.data 
boston_y = boston.target

print(boston_X.shape) # (506, 13) 
print(boston_y.shape) # (506,) 


# 4. california 주택가격 
'''
•타겟 변수 : y변수
1990년 캘리포니아의 각 행정 구역 내 주택 가격의 중앙값

•특징 변수(8) : x변수
MedInc : 행정 구역 내 소득의 중앙값
HouseAge : 행정 구역 내 주택 연식의 중앙값
AveRooms : 평균 방 갯수
AveBedrms : 평균 침실 갯수
Population : 행정 구역 내 인구 수
AveOccup : 평균 자가 비율
Latitude : 해당 행정 구역의 위도
Longitude : 해당 행정 구역의 경도
'''
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
print(california.DESCR)

# X변수 -> DataFrame 변환 
cal_df = pd.DataFrame(california.data, columns=california.feature_names)
# y변수 추가 
cal_df["MEDV"] = california.target
cal_df.tail()
cal_df.info() 

>>>>>>> 32f7d70641783dee9a7f41f16c0d9a0ed6467ceb
