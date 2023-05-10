# -*- coding: utf-8 -*-
"""
스팸 메시지 -> 희소행렬 만들기 

1. csv file 가져오기 
2. texts, target 전처리 
3. max features
4. 희소행렬(TFiDF 가중치)
"""

import pandas as pd # csv file 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import LabelEncoder 

# 1. csv file 가져오기  
path = r"C:\ITWILL\5_Python_ML\data"
spam_data = pd.read_csv(path + '/spam_data.csv', header=None, encoding='utf-8')
print(spam_data.head())
'''
      0                                                  1
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...
'''

# 2. texts, target 전처리

# 1) target 전처리 : 레이블 인코딩 
target = spam_data[0]

y = LabelEncoder().fit_transform(target) # 0 or 1 

# 2) texts 전처리 : 공백, 특수문자, 숫자  
texts = spam_data[1]

print('전처리 전')
print(texts)

# << texts 전처리 함수 >> 
import string # texts 전처리
def text_prepro(texts): # 문단(sentences)
    # Lower case : 문단 -> 문장 -> 영문소문자 변경  
    texts = [x.lower() for x in texts]
    # Remove punctuation : 문단 -> 문장 -> 음절 -> 필터링 -> 문장  
    texts = [''.join(ch for ch in st if ch not in string.punctuation) for st in texts]
    # Remove numbers : 문단 -> 문장 -> 음절 -> 필터링 -> 문장 
    texts = [''.join(ch for ch in st if ch not in string.digits) for st in texts]
    # Trim extra whitespace : 문단 -> 문장 -> 공백 제거 
    texts = [' '.join(x.split()) for x in texts]
    return texts

texts = text_prepro(texts)
print('전처리 후 ')
print(texts)

# 3. max features : 희소행렬의 차원 결정 
fit = TfidfVectorizer().fit(texts) # 단어 생성기 
voca = fit.vocabulary_
print(voca)

max_features = 5000 # 전체 단어수 

# 4. sparse matrix : max features 지정 
tfidf = TfidfVectorizer(max_features = max_features, stop_words='english')

# 희소행렬 
sparse_mat = tfidf.fit_transform(texts)
print(sparse_mat)


