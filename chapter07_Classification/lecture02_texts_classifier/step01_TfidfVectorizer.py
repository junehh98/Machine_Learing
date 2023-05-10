# -*- coding: utf-8 -*-
"""
TFiDF 단어 생성기 : TfidfVectorizer  
  1. 단어 생성기(word tokenizer) & 단어 사전(word dictionary) 
  2. 희소행렬(sparse matrix) : 단어 출현 비율에 의해서 가중치 적용 행렬 
    1) TF 가중치 : 단어출현빈도수  
    2) TFiDF 가중치 : 단어출현빈도수(TF) x 문서출현빈도수의 역수(iDF) 
"""

from sklearn.feature_extraction.text import TfidfVectorizer # 단어 생성기

# 테스트 문장 
sentences = [
    "Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow.",
    "Professor Plum has a green plant in his study.",
    "Miss Scarlett watered Professor Plum's green plant while he was away from his office last week."
]

print(sentences)


# 1. 단어 생성기
obj = TfidfVectorizer() # 생성자 


# 2. 단어 사전 
fit = obj.fit(sentences) # 문장 적용 
voca = fit.vocabulary_ 


# 3. 희소행렬(sparse matrix)
sparse_mat = obj.fit_transform(sentences)
print(sparse_mat)

















