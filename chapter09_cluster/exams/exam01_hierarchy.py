'''
 문1) 신입사원 면접시험(interview.csv) 데이터 셋을 이용하여 다음과 같이 군집모델을 생성하시오.
 <조건1> 대상칼럼 : 가치관,전문지식,발표력,인성,창의력,자격증,종합점수 
 <조건2> 계층적 군집분석의 완전연결방식 적용 
 <조건3> 덴드로그램 시각화 : 군집 결과 확인   
'''

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster # 계층적 군집 model
import matplotlib.pyplot as plt

# data loading - 신입사원 면접시험 데이터 셋 
interview = pd.read_csv("C:/ITWILL/5_Python_ML/data/interview.csv", encoding='ms949')
print(interview.info())
'''
RangeIndex: 15 entries, 0 to 14
Data columns (total 9 columns):
'''

# <조건1> subset 생성 : no, 합격여부 칼럼을 제외한 나머지 칼럼 
subset = interview.drop(['no','합격여부'], axis=1)
subset.info()

subset.shape # (15, 7)



# <조건2> 계층적 군집 분석  완전연결방식 
clusters = linkage(subset, method='complete', metric='euclidean')

print(clusters)
clusters.shape # (14, 4)



# <조건3> 덴드로그램 시각화 : 군집 결과 확인
plt.figure(figsize = (10, 5))
dendrogram(clusters)
plt.show()



# <조건4> 군집 자르기 : 최대클러스터 개수 3개 지정  
cut_cluster = fcluster(clusters, t=3, criterion='maxclust') 

unique, counts = np.unique(cut_cluster, return_counts=True)
print(unique, counts) # [1 2 3] [5 5 5]



# <조건5> DF에 cluster 칼럼 추가 & 군집별 특성 분석(그룹 통계 이용)
subset['cluster'] = cut_cluster

group = subset.groupby(by = 'cluster')
group.size()
'''

1    5
2    5
3    5
'''
group.mean()
'''
          가치관  전문지식   발표력    인성   창의력  자격증  종합점수
cluster                                         
1        11.0       15.2    19.4     11.0    6.2     0.4     62.8
2        19.0       14.4    15.6     14.8    11.8    1.0     75.6 --> 가장 높음 
3        14.4       18.8    10.8     9.4     18.2    0.0     71.6
'''
plt.scatter(subset['가치관'],subset['전문지식'],
            c=subset['cluster'])
plt.show()














