# 문2) mtcars 자료를 이용하여 다음과 같은 단계로 이상치를 처리하시오.

import pandas as pd 
import seaborn as sn # 데이터셋 로드 
pd.set_option('display.max_columns', 50) # 최대 50 칼럼수 지정
import matplotlib.pyplot as plt # boxplot 시각화 


# 데이터셋 로드 
data = sn.load_dataset('mpg')
data.info()
print(data)


# 단계1. boxplot으로 'acceleration' 칼럼 이상치 탐색 


# 단계2. IQR 방식으로 이상치 탐색

# 1) IQR 수식 작성 


# 2) 이상치 확인 


# 단계3. 이상치 대체 : 정상범주의 하한값과 상한값 대체 


# 단계4. boxplot으로 'acceleration' 칼럼 이상치 처리결과 확인 



