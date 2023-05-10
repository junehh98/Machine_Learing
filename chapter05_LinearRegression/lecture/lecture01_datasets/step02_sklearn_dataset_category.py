<<<<<<< HEAD
'''
분류분석용 sklearn dataset 정리
'''
from sklearn import datasets # dataset 제공 library
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

######################################
# 분류분석에 적합한 데이터셋
######################################

# 1. wine 
'''
와인의 화학 조성을 사용하여 와인의 종류를 예측하기 위한 데이터

•타겟 변수 : y변수 
◦와인의 종류 : 0, 1, 2 세가지 값 

•특징 변수 : x변수  
◦알콜(Alcohol)
◦말산(Malic acid)
◦회분(Ash)
◦회분의 알칼리도(Alcalinity of ash) 
◦마그네슘(Magnesium)
◦총 폴리페놀(Total phenols)
◦플라보노이드 폴리페놀(Flavanoids)
◦비 플라보노이드 폴리페놀(Nonflavanoid phenols)
◦프로안토시아닌(Proanthocyanins)
◦색상의 강도(Color intensity)
◦색상(Hue)
◦희석 와인의 OD280/OD315 비율 (OD280/OD315 of diluted wines)
◦프롤린(Proline)
'''
from sklearn.datasets import load_wine
wine = load_wine()
print(wine.target_names) # ['class_0', 'class_1', 'class_2']
print(wine.feature_names)


X, y = load_wine(return_X_y=True)
print(np.shape(X)) # (178, 13) : matrix
print(np.shape(y)) # (178,) : vector

# numpy -> DataFrame 
wine_df = pd.DataFrame(X, columns=wine.feature_names)
wine_df['class'] = y

# class별 주요변수 간 산점도 
sn.pairplot(vars=["alcohol", "alcalinity_of_ash", "total_phenols", "flavanoids"], 
             hue="class", data=wine_df)
plt.show()


# 2. breast cancer 데이터셋
'''
유방암(breast cancer) 진단 데이터 

•타겟 변수 
 - 종양이 양성(benign)인지 악성(malignant)인지를 판별
•특징 변수(30개) 
 - 유방암 진단 사진으로부터 측정한 종양(tumar)의 특징값
'''
cancer = datasets.load_breast_cancer()
print(cancer)
print(cancer.DESCR)

cancer_x = cancer.data
cancer_y = cancer.target
print(np.shape(cancer_x)) # (569, 30) : matrix
print(np.shape(cancer_y)) # (569,) : vector

cencar_df = pd.DataFrame(cancer_x, columns=cancer.feature_names)
cencar_df['class'] = cancer.target
cencar_df.tail()

# 타겟 변수 기준 주요변수 간 산점도 
sn.pairplot(vars=["worst radius", "worst texture", "worst perimeter", "worst area"], 
             hue="class", data=cencar_df)
plt.show()



# 3. digits 데이터셋 - 숫자 예측(0~9)
'''
숫자 필기 이미지 데이터

•타겟 변수 
 - 0 ~ 9 : 10진수 정수 
•특징 변수(64픽셀) 
 -0부터 9까지의 숫자를 손으로 쓴 이미지 데이터
 -각 이미지는 0부터 15까지의 16개 명암을 가지는 8x8=64픽셀 해상도의 흑백 이미지
'''
digits = datasets.load_digits()
print(digits.DESCR)

print(digits.data.shape) # (1797, 64)
print(digits.target.shape) # (1797,)
print(digits) # 8x8 image of integer pixels in the range 0..16

# 첫번째 이미지 픽셀, 정답 
img2d = digits.data[0].reshape(8,8)
plt.imshow(img2d) # 0 확인 
digits.target[0] # 0 정답 

   

# 4. news group 
'''
- 20개의 뉴스 그룹 문서 데이터(문서 분류 모델 예문으로 사용)

•타겟 변수 
◦문서가 속한 뉴스 그룹 : 20개 

•특징 변수 
◦문서 텍스트 : 18,846
'''

from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all') # 'train', 'test'
# Downloading 20news dataset.

print(newsgroups.DESCR)
'''
**Data Set Characteristics:**

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========
'''

# data vs target
newsgroups.data # text
len(newsgroups.data) # 18846

newsgroups.target # array([10,  3, 17, ...,  3,  1,  7])
len(newsgroups.target) # 18846

# 뉴스 그룹 : 20개 이름 
newsgroups.target_names # ['alt.atheism', ... 'talk.religion.misc']


=======
'''
분류분석용 sklearn dataset 정리
'''
from sklearn import datasets # dataset 제공 library
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

######################################
# 분류분석에 적합한 데이터셋
######################################

# 1. wine 
'''
와인의 화학 조성을 사용하여 와인의 종류를 예측하기 위한 데이터

•타겟 변수 : y변수 
◦와인의 종류 : 0, 1, 2 세가지 값 

•특징 변수 : x변수  
◦알콜(Alcohol)
◦말산(Malic acid)
◦회분(Ash)
◦회분의 알칼리도(Alcalinity of ash) 
◦마그네슘(Magnesium)
◦총 폴리페놀(Total phenols)
◦플라보노이드 폴리페놀(Flavanoids)
◦비 플라보노이드 폴리페놀(Nonflavanoid phenols)
◦프로안토시아닌(Proanthocyanins)
◦색상의 강도(Color intensity)
◦색상(Hue)
◦희석 와인의 OD280/OD315 비율 (OD280/OD315 of diluted wines)
◦프롤린(Proline)
'''
from sklearn.datasets import load_wine
wine = load_wine()
print(wine.target_names) # ['class_0', 'class_1', 'class_2']
print(wine.feature_names)


X, y = load_wine(return_X_y=True)
print(np.shape(X)) # (178, 13) : matrix
print(np.shape(y)) # (178,) : vector

# numpy -> DataFrame 
wine_df = pd.DataFrame(X, columns=wine.feature_names)
wine_df['class'] = y

# class별 주요변수 간 산점도 
sn.pairplot(vars=["alcohol", "alcalinity_of_ash", "total_phenols", "flavanoids"], 
             hue="class", data=wine_df)
plt.show()


# 2. breast cancer 데이터셋
'''
유방암(breast cancer) 진단 데이터 

•타겟 변수 
 - 종양이 양성(benign)인지 악성(malignant)인지를 판별
•특징 변수(30개) 
 - 유방암 진단 사진으로부터 측정한 종양(tumar)의 특징값
'''
cancer = datasets.load_breast_cancer()
print(cancer)
print(cancer.DESCR)

cancer_x = cancer.data
cancer_y = cancer.target
print(np.shape(cancer_x)) # (569, 30) : matrix
print(np.shape(cancer_y)) # (569,) : vector

cencar_df = pd.DataFrame(cancer_x, columns=cancer.feature_names)
cencar_df['class'] = cancer.target
cencar_df.tail()

# 타겟 변수 기준 주요변수 간 산점도 
sn.pairplot(vars=["worst radius", "worst texture", "worst perimeter", "worst area"], 
             hue="class", data=cencar_df)
plt.show()



# 3. digits 데이터셋 - 숫자 예측(0~9)
'''
숫자 필기 이미지 데이터

•타겟 변수 
 - 0 ~ 9 : 10진수 정수 
•특징 변수(64픽셀) 
 -0부터 9까지의 숫자를 손으로 쓴 이미지 데이터
 -각 이미지는 0부터 15까지의 16개 명암을 가지는 8x8=64픽셀 해상도의 흑백 이미지
'''
digits = datasets.load_digits()
print(digits.DESCR)

print(digits.data.shape) # (1797, 64)
print(digits.target.shape) # (1797,)
print(digits) # 8x8 image of integer pixels in the range 0..16

# 첫번째 이미지 픽셀, 정답 
img2d = digits.data[0].reshape(8,8)
plt.imshow(img2d) # 0 확인 
digits.target[0] # 0 정답 

   

# 4. news group 
'''
- 20개의 뉴스 그룹 문서 데이터(문서 분류 모델 예문으로 사용)

•타겟 변수 
◦문서가 속한 뉴스 그룹 : 20개 

•특징 변수 
◦문서 텍스트 : 18,846
'''

from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all') # 'train', 'test'
# Downloading 20news dataset.

print(newsgroups.DESCR)
'''
**Data Set Characteristics:**

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========
'''

# data vs target
newsgroups.data # text
len(newsgroups.data) # 18846

newsgroups.target # array([10,  3, 17, ...,  3,  1,  7])
len(newsgroups.target) # 18846

# 뉴스 그룹 : 20개 이름 
newsgroups.target_names # ['alt.atheism', ... 'talk.religion.misc']


>>>>>>> 32f7d70641783dee9a7f41f16c0d9a0ed6467ceb
