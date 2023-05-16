# -*- coding: utf-8 -*-
"""
문2) 아래와 같은 단계로 kMeans 알고리즘을 적용하여 확인적 군집분석을 수행하시오.

 <조건> 변수 설명 : tot_price : 총구매액, buy_count : 구매횟수, 
                   visit_count : 매장방문횟수, avg_price : 평균구매액

  단계1 : 3개 군집으로 군집화
 
  단계2: 원형데이터에 군집 예측치 추가
  
  단계3 : tot_price 변수와 가장 상관계수가 높은 변수로 산점도(색상 : 클러스터 결과)
  
  단계4 : 산점도에 군집의 중심점 시각화

   단계5 : 군집별 특성분석 : 우수고객 군집 식별
"""

import pandas as pd
from sklearn.cluster import KMeans # kMeans model
import matplotlib.pyplot as plt


sales = pd.read_csv("C:/ITWILL/5_Python_ML/data/product_sales.csv")
print(sales.info())
'''
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
tot_price      150 non-null float64 -> 총구매금액 
visit_count    150 non-null float64 -> 매장방문수 
buy_count      150 non-null float64 -> 구매횟수 
avg_price      150 non-null float64 -> 평균구매금액 
'''




# 단계1 : 3개 군집으로 군집화
obj = KMeans(n_clusters=3, max_iter=300, algorithm='auto')



# 단계2: 원형데이터에 군집 예측치 추가
model = obj.fit(sales)

pred = model.labels_
print(pred)


sales['predict'] = pred 


# 단계3 : tot_price 변수와 가장 상관계수가 높은 변수로 산점도(색상 : 클러스터 결과)
sales.corr(method='pearson')
'''
             tot_price  visit_count  buy_count  avg_price   predict
tot_price     1.000000     0.817954  -0.013051   0.871754  0.426995
visit_count   0.817954     1.000000  -0.230612   0.962757  0.634820
buy_count    -0.013051    -0.230612   1.000000  -0.278505 -0.503984
avg_price     0.871754     0.962757  -0.278505   1.000000  0.678979
predict       0.426995     0.634820  -0.503984   0.678979  1.000000
'''

plt.scatter(x=sales['tot_price'], y=sales['avg_price'], 
            c=sales['predict'])


# 단계4 : 산점도에 군집의 중심점 시각화
# 중심값
centers = model.cluster_centers_
print(centers)


plt.scatter(x=centers[:,0], y=centers[:,3], 
            c='r', marker='D')
plt.show()



# 단계5 : 군집별 특성분석 : 우수고객 군집 식별
group = sales.groupby('predict')
group.size()
'''
0    62
1    50
2    38
'''

group.mean()
'''
         tot_price  visit_count  buy_count  avg_price
predict                                              
0         5.901613     1.433871   2.754839   4.393548
1         5.006000     0.244000   3.284000   1.464000
2         6.850000     2.071053   3.071053   5.742105
'''








































