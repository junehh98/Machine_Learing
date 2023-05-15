'''
 문1) 신입사원 면접시험(interview.csv) 데이터 셋을 이용하여 다음과 같이 군집모델을 생성하시오.
 <조건1> 대상칼럼 : 가치관,전문지식,발표력,인성,창의력,자격증,종합점수 
 <조건2> 계층적 군집분석의 완전연결방식 적용 
 <조건3> 덴드로그램 시각화 : 군집 결과 확인   
'''

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster # 계층적 군집 model
import matplotlib.pyplot as plt

# data loading - 신입사원 면접시험 데이터 셋 
interview = pd.read_csv("C:/ITWILL/5_Python-II/data/interview.csv", encoding='ms949')
print(interview.info())
'''
RangeIndex: 15 entries, 0 to 14
Data columns (total 9 columns):
'''

# <조건1> subset 생성 : no, 합격여부 칼럼을 제외한 나머지 칼럼 


# <조건2> 계층적 군집 분석  완전연결방식 


# <조건3> 덴드로그램 시각화 : 군집 결과 확인


# <조건4> 군집 자르기 : 최대클러스터 개수 3개 지정  


# <조건5> DF에 cluster 칼럼 추가 & 군집별 특성 분석(그룹 통계 이용)
