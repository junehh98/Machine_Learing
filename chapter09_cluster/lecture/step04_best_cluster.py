'''
Best Cluster 찾는 방법 
    이상적인 클러스터 : 분리도 크게, 응집도 작게 
'''

from sklearn.cluster import KMeans # model 
from sklearn.datasets import load_iris # dataset 
import matplotlib.pyplot as plt # 시각화 

# 1. dataset load 
X, y = load_iris(return_X_y=True)
print(X.shape) # (150, 4)
print(X)


# 2. Best Cluster 
size = range(1, 11) # k값 범위
inertia = [] # interia : 중심점과 포인트 간 거래제곱합(응집도)

for k in size : 
    obj = KMeans(n_clusters = k) 
    model = obj.fit(X)
    inertia.append(model.inertia_) 

print(inertia)


# 3. best cluster 찾기 
plt.plot(size, inertia, '-o')
plt.xticks(size)
plt.show()





