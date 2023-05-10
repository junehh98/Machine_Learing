# -*- coding: utf-8 -*-
"""
스팸 메시지 -> 희소행렬(가중치 : TFiDF) 만들기 

1. csv file 가져오기 
2. texts(X), target(y) 전처리 
3. max features
4. 희소행렬(TFiDF 가중치)
5. train/test split[추가]
6. file save[추가]
"""

import pandas as pd # csv file 
from sklearn.feature_extraction.text import TfidfVectorizer # X변수 
from sklearn.preprocessing import LabelEncoder # y변수 인코딩 

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
y # [0, 0, 1, ..., 0, 0, 0]

# 2) texts 전처리 : 공백, 특수문자, 숫자  
texts = spam_data[1]
len(texts) # 5574

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
    #texts = [''.join(ch for ch in st if ch not in string.digits) for st in texts]
    # Trim extra whitespace : 문단 -> 문장 -> 공백 제거 
    texts = [' '.join(x.split()) for x in texts]
    return texts

# 함수 호출 
texts = text_prepro(texts)
print('전처리 후 ')
print(texts)


# 3. max features : 희소행렬의 차원 결정 
fit = TfidfVectorizer().fit(texts) # 단어 생성기 
voca = fit.vocabulary_
print(voca)
len(voca) # 9541

max_features = 5000 # 전체 단어수 

# 4. sparse matrix : max features 지정 
tfidf = TfidfVectorizer(max_features = max_features, 
                        stop_words='english')

# 희소행렬 
sparse_mat = tfidf.fit_transform(texts)
print(sparse_mat)

# numpy 변환 
sparse_mat_arr = sparse_mat.toarray()
sparse_mat_arr.shape # (5574, 5000)

# 5. train/test split[추가]
from sklearn.model_selection import train_test_split  

x_train, x_test, y_train, y_test = train_test_split(
    sparse_mat_arr, y, test_size=0.3)


# 6. file save[추가]  
new_spam_data = (x_train, x_test, y_train, y_test)

import numpy as np

# *.npy 확장자 자동 부여  
np.save(path + "/new_spam_data", new_spam_data) 










