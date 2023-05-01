#step03_continuous

# 연속형 변수 시각화 
# - 산점도, 산점도 행렬, boxplot 

import matplotlib.pyplot as plt
import seaborn as sn

# seaborn 한글과 음수부호, 스타일 지원 
sn.set(font="Malgun Gothic", 
            rc={"axes.unicode_minus":False}, style="darkgrid")

# dataset load 
iris = sn.load_dataset('iris')
tips = sn.load_dataset('tips')


x = iris.sepal_length

# 1-1. displot : 히스토그램
sn.displot(data=iris, x='sepal_length', kind='hist')  
plt.title('iris Sepal length hist') # 단위 : Count 
plt.show()


# 1-2. displot : 밀도분포곡선(hue=범주형변수) 
sn.displot(data=iris, x='sepal_length', kind="kde", hue='species') 
plt.title('iris Sepal length kde') # 단위 : Density
plt.show()


# 2. 산점도 행렬(scatter matrix)  
sn.pairplot(data=iris, hue='species') 
plt.show()


# 3. 산점도 : 연속형+연속형   
sn.scatterplot(x="sepal_length", y="petal_length", 
               hue='species', data=iris)
plt.title('산점도 행렬(scatter matrix)')
plt.show()


