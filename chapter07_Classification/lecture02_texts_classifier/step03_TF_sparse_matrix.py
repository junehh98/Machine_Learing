# -*- coding: utf-8 -*-
"""
스팸 메시지 -> 희소행렬(가중치 : TF) 만들기 

1. csv file 가져오기 
2. texts, target 전처리 
3. max features
4. 희소행렬(TF 가중치)
"""

import string # texts 전처리 
import pandas as pd # csv file 
from sklearn.feature_extraction.text import CountVectorizer # X변수 : 희소행렬 

# 1. csv file 가져오기 
path = r"C/Users/junehh98/Desktop/itwill/5_Python_ML/data"
spam_data = pd.read_csv(path + '/spam_data.csv', header=None, encoding='utf-8')
print(spam_data.info())
print(spam_data.head())


# 2. texts 전처리

texts = spam_data[1]

print('전처리 전')
print(texts)

# texts 전처리 함수 
def text_prepro(texts):
    # Lower case 
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers 
    texts = [''.join(c for c in x if c not in string.digits) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return texts


texts = text_prepro(texts)
print('전처리 후 ')
print(texts)


# 3. max features : 희소행렬의 열 수(word size)
fit = CountVectorizer().fit(texts) # 단어 생성기 
voca = fit.vocabulary_ # 단어 사전 
print(voca)

max_features = len(voca) # 전체 단어수 


# 4. sparse matrix : max features 지정 
tf = CountVectorizer(max_features = max_features)

# 희소행렬 
sparse_mat = tf.fit_transform(texts)
print(sparse_mat)

# numpy array 변환 
sparse_mat_arr = sparse_mat.toarray()
print(sparse_mat_arr)





