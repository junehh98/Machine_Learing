'''
lecture01 > step02 관련문제

문3) wdbc_data.csv 파일을 읽어와서 단계별로 x, y 변수를 생성하시오.
     <단계1> : 파일 가져오기, 정보 확인 
     <단계2> : y변수 : diagnosis 
              x변수 : id 칼럼 제외  나머지 30개 칼럼
'''
import pandas as pd

path = r"c:/ITWILL/5_Python_ML/data" # file 경로 변경 

# <단계1> : 파일 가져오기, 정보 확인 
wdbc = pd.read_csv(path + '/wdbc_data.csv')
wdbc.head()

# <단계2> : y변수, x변수 선택
names = list(wdbc.columns)
names

wdbc_x = wdbc[names[2:]]
wdbc_x.columns

wdbc_y = wdbc.diagnosis
wdbc_y

