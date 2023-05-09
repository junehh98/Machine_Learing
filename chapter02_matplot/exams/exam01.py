'''
문1) iris.csv 파일을 이용하여 다음과 같이 차트를 그리시오.
    <조건1> iris.csv 파일을 iris 변수명으로 가져온 후 파일 정보 보기
    <조건2> 1번 칼럼과 3번 칼럼을 대상으로 산점도 그래프 그리기
    <조건3> 1번 칼럼과 3번 칼럼을 대상으로 산점도 그래프 그린 후 5번 칼럼으로 색상 적용
            힌트) plt.scatter(x, y, c) 
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder # 인코딩 class


# <조건1> iris.csv 파일을 iris 변수명으로 가져온 후 파일 정보 보기
path = r'C:\ITWILL\5_Python_ML\data'
iris = pd.read_csv(path + '/iris.csv')

print(iris.info())
'''
 0   Sepal.Length  150 non-null    float64
 1   Sepal.Width   150 non-null    float64
 2   Petal.Length  150 non-null    float64
 3   Petal.Width   150 non-null    float64
 4   Species       150 non-null    object
'''


# <조건2> 1번 칼럼과 3번 칼럼을 대상으로 산점도 그래프 그리기
plt.scatter(iris['Sepal.Width'], iris['Petal.Width'])
plt.show()

# <조건3> 1번 칼럼과 3번 칼럼을 대상으로 산점도 그래프 그린 후 5번 칼럼으로 색상 적용
iris.Species.unique() 
# array(['setosa', 'versicolor', 'virginica']
encoder = LabelEncoder().fit(iris.Species) # 데이터셋 적용 

species_encoding = encoder.transform(iris.Species) # 레이블 인코딩 변환 
species_encoding

iris['species2'] = species_encoding
iris.head()


plt.scatter(iris['Sepal.Width'], iris['Petal.Width'], 
            c=iris.species2, marker='o')
plt.title('sepal vs  width')
for idx, value in enumerate(species_encoding) : 
    plt.annotate(text=value, 
                 xy=(iris.Sepal.Width[idx],iris.Petal.Width[idx]))
plt.show()


















