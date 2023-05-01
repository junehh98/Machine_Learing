'''
문4) ~/chap03_Numpy/data 폴더에 포함된 전체 텍스트 파일(*.txt) 전체를 읽어서 리스트에 저장하시오. 
'''

import pandas as pd
from glob import glob # 파일 검색 패턴 사용

# text file 경로 
path = r"C:\ITWILL\5_Python_ML\workspace\chapter03_Numpy" # 파일 기본 경로 



full_text = [] # 텍스트 저장 list 

for file in glob(path + '/data/*.txt') :
    print(file)
    file = open(file = file, mode='r', encoding = 'utf-8')
    text_data = file.read()
    file.close()
    full_text.append(text_data)
    
# 텍스트 확인
print(full_text)
len(full_text)
