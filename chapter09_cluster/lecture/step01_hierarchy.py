'''
계층적 군집분석(Hierarchical Clustering) 
 - 상향식(Bottom-up)으로 계층적 군집 형성 
 - 유클리드안 거리계산식 이용 
 - 숫자형 변수만 사용
 절차 : 입력자료 -> 거리계산 -> 덴드로그램 -> 군집수 -> 군집특징 
'''

from sklearn.datasets import load_iris # dataset
import pandas as pd # DataFrame
from scipy.cluster.hierarchy import linkage, dendrogram # 군집분석 tool
import matplotlib.pyplot as plt # 산점도 시각화 


# 1. dataset loading
iris = load_iris() # Load the data

X = iris.data # x변수 
y = iris.target # y변수(target) - 숫자형 : 거리계산 

# numpy -> DataFrame 
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = y # target 추가 
iris_df.shape # 150, 5)


### simple data : 3x3
import numpy as np

X = np.arange(1,10).reshape(3,3)
X
'''
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
'''
Z = linkage(X, method='complete')
Z.shape # (2, 4)
'''
array([[ 0.    ,  1.    ,  5.19615242,  2.    ], : 1 vs 2, 거리, 노드수 
       [ 2.    ,  3.    , 10.39230485,  3.   ]]) : 2 vs 3()
'''



# 2. 계층적 군집분석 : 거리계산 
clusters = linkage(iris_df, method='complete') # linkage로 거리계산 
'''
method = 'complete' : default - 완전연결 
method = 'single' : 단순연결
method = 'average' : 평균연결
'''

print(clusters)
clusters.shape # (149, 4)



# 3. 덴드로그램(dendrogram) 시각화 : 군집수 사용자가 결정 
plt.figure(figsize = (25, 10))
dendrogram(clusters)
plt.show()



# 4. 클러스터링 자르기
from scipy.cluster.hierarchy import fcluster # 클러스터 자르기 도구 
import numpy as np # 클러스터 빈도수 


# 클러스터 자르기
cut_cluster = fcluster(clusters, t=3, criterion='maxclust') 

help(fcluster)
'''
fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None)
 Z : ndarray
 t : scalar
 criterion : inconsistent / distance / maxcluster
'''

# 클러스터 빈도수 
unique, counts = np.unique(cut_cluster, return_counts=True)
print(unique, counts)
# [1 2 3] [50 34 66]



# 5. DF 칼럼 추가 
iris_df['cluster'] = cut_cluster


# 6. 계층적군집분석 시각화 
plt.scatter(iris_df['sepal length (cm)'], iris_df['petal length (cm)'],
            c=iris_df['cluster'])
plt.show()


iris_df.head()
iris_df.tail()


# 7. 군집별 특성 분석 
group = iris_df.groupby(by = 'cluster')
group.size()
'''
1    50
2    34
3    66
'''
group.mean() # 군집별 특성 보기 
'''
         sepal length (cm)  sepal width (cm)  ...  petal width (cm)   species
cluster                                       ...                            
1                 5.006000          3.428000  ...          0.246000  0.000000
2                 6.888235          3.100000  ...          2.123529  2.000000
3                 5.939394          2.754545  ...          1.445455  1.242424
'''

















