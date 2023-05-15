'''
계층적 군집분석(Hierarchical Clustering) 
 - 상향식(Bottom-up)으로 계층적 군집 형성 
 - 유클리드안 거리계산식 이용 
 - 숫자형 변수만 사용
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


# 2. 계층적 군집분석 
clusters = linkage(iris_df, method='complete')
'''
method = 'complete' : default - 완전연결 
method = 'simple' : 단순연결
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


# 클러스터 빈도수 
unique, counts = np.unique(cut_cluster, return_counts=True)
print(unique, counts)


# 5. DF 칼럼 추가 
iris_df['cluster'] = cut_cluster


# 6. 계층적군집분석 시각화 
plt.scatter(iris_df['sepal length (cm)'], iris_df['petal length (cm)'],
            c=iris_df['cluster'])
plt.show()


