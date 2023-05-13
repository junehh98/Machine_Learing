'''
 문3) 당료병(diabetes.csv) 데이터 셋을 이용하여 다음과 같은 단계로 
     RandomForest 모델을 생성하시오.

  <단계1> 데이터셋 로드 & 칼럼명 적용 
  <단계2> x, y 변수 선택 : x변수 : 1 ~ 8번째 칼럼, y변수 : 9번째 칼럼
  <단계3> 500개의 트리를 이용하여 모델 생성   
  <단계4> 중요변수 시각화 : feature names 적용                    
'''
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt # 중요변수 시각화 

# 단계1. 테이터셋 로드  
dia = pd.read_csv('/Users/junehh98/Desktop/itwill/5_Python_ML/data/diabetes.csv', header=None) # 제목 없음 
print(dia.info())

# 칼럼명 추가 
dia.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
               'Insulin','BMI','DiabetesPedigree','Age','Outcome']
print(dia.info()) 
'''
 0   Pregnancies       759 non-null    float64
 1   Glucose           759 non-null    float64
 2   BloodPressure     759 non-null    float64
 3   SkinThickness     759 non-null    float64
 4   Insulin           759 non-null    float64
 5   BMI               759 non-null    float64
 6   DiabetesPedigree  759 non-null    float64
 7   Age               759 non-null    float64
 8   Outcome           759 non-null    int64  --> y변수
 (한글명 : 임신, 혈당, 혈압, 피부두께,인슐린,비만도지수,당료병유전,나이,결과)  
'''
dia.Outcome.unique() # [0, 1]



# 단계2. x,y 변수 생성 
X = dia.drop('Outcome', axis=1)

y = dia.Outcome


# 단계3. model 생성
model = RandomForestClassifier(random_state=0).fit(X=X, y=y)



# 단계4. 중요변수 시각화 
print('중요도 : ', model.feature_importances_)

x_names = X.columns # x변수 이름  

x_size = len(x_names) # x변수 개수  



### 가로막대 차트 : 중요점수 순으로 정렬

# 중요점수 오름차순 정렬 
importances_sorted = sorted(model.feature_importances_)

# 중요점수 오름차순 색인 정렬 
idx = model.feature_importances_.argsort()
idx 

# numpy array 변환 
x_names = np.array(x_names)

# X변수 중요점수 순으로 정렬 
sorted_x_names = x_names[idx]

 
plt.barh(range(x_size), importances_sorted) # (y, x)
plt.yticks(range(x_size), sorted_x_names)   
plt.xlabel('feature_importances') 
plt.show()

















