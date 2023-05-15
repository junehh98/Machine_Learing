'''
Best Cluster 찾는 방법 
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
inertia = [] # 응집도 

for k in size : 
    obj = KMeans(n_clusters = k) 
    model = obj.fit(X)
    inertia.append(model.inertia_) 

print(inertia)


# 3. best cluster 찾기 
plt.plot(size, inertia, '-o')
plt.xticks(size)
plt.show()





