# -*- coding: utf-8 -*-
"""
한글문서를 대상으로 한 문서분류기 생성  
"""
from konlpy.tag import Okt # 한글 형태소분석기 
okt = Okt() # 객체 생성  

import pandas as pd # csv file 
from sklearn.feature_extraction.text import TfidfVectorizer # X변수 : 희소행렬
from sklearn.preprocessing import LabelEncoder # y변수 : 레이블 인코딩 
from sklearn.model_selection import train_test_split # datase split 

from sklearn.naive_bayes import MultinomialNB # nb model
from sklearn.metrics import accuracy_score, confusion_matrix # 평가 


# 1. csv file 가져오기 : cafe에서 file 다운로드(daum_movie_reviews.csv) 
path = r"C:\itwill\5_Python_ml\data"
movie_review = pd.read_csv(path + '/daum_movie_reviews.csv', encoding='utf-8')

print(movie_review.info())
'''
RangeIndex: 14725 entries, 0 to 14724
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   review  14725 non-null  object  : 영화 후기 
 1   rating  14725 non-null  int64   : 점수(1~10)
 2   date    14725 non-null  object  : 작성 날짜
 3   title   14725 non-null  object  : 영화 제목 
'''


# 2. 변수 선택 
new_df = movie_review[['review', 'title']]  

# 영화제목 유일값 확인 
new_df.title.unique() 

# X변수 선택 
review = new_df.review # 영화후기(한글자연어) 

# y변수 인코딩   
y = LabelEncoder().fit_transform(y = new_df.title)



# 3. train/test split
x_train, x_test, y_train, y_test = train_test_split(
    review, y, test_size=0.25, random_state=0)



# 4. 희소행렬(sparse matrix)   
tfidf = TfidfVectorizer(tokenizer = okt.nouns, max_features = 2000,
                        analyzer='word', min_df=5, max_df=0.5)
'''
tokenizer = okt.nouns : 한글토큰생성기
max_features = 2000 : 최대 사용할 단어개수 
analyzer='word' : 특징(feature)이 단어 또는 문자인지 지정(default='word')
min_df=5 : 너무 희소하게 나오는 단어는 제거(5개 이하로 나온 단어는 무시) 
max_df=0.5 : 너무 자주 나타나는 단어는 제거("문서에서 10% 이상 나온 단어는 무시") 
max_df=25 : "25번 이상 나온 단어는 무시"
'''


# x_train 희소행렬 
x_train_tfidf = tfidf.fit_transform(x_train) # 객체에 적용 & 희소행렬 변환 
x_train_tfidf.shape 

# x_test 희소행렬 
x_test_tfidf = tfidf.transform(x_test) # transform() 함수 
x_test_tfidf.shape 


# 5. NB model & 평가 
model = MultinomialNB().fit(X = x_train_tfidf, y = y_train)

y_pred = model.predict(X = x_test_tfidf) # 예측치 
y_true = y_test # 관측치(정답)

acc = accuracy_score(y_true, y_pred)
print('분류정확도 =', acc)

con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)



