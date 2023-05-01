''' 
lecture01 > step02 관련문제

문2) score.csv 파일을 읽어와서 다음과 같이 처리하시오.
   조건1> tv 칼럼이 0인 관측치 2개 삭제 (조건식 이용)
   조건2> score, academy 칼럼만 추출하여 DataFrame 생성
   조건3> score, academy 칼럼의 평균 계산 
   - <<출력 결과 >> 참고    
   
<<출력 결과 >>
   score  academy
1     75        1
2     77        1
3     83        2
4     65        0
5     80        3
6     83        3
7     70        1
9     79        2
score      76.500
academy     1.625   
'''

import pandas as pd

path = r"c:/ITWILL/5_Python_ML/data" # file 경로 변경 

score = pd.read_csv(path + '/score.csv')
print(score.info())
print(score)

# 조건1
score1 = score[score.tv > 0]

# 조건2
score1 = score[['score', 'academy']]
score1

# 조건3
score1.mean(axis=0)
'''
score      78.9
academy     1.9
'''






















