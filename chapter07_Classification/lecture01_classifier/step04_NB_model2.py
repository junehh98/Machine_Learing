'''
3. MultinomialNB : x변수가 단어 빈도수(텍스트 데이터)를 분류할 때 적합
'''

###############################
### news groups 분류 
###############################

from sklearn.naive_bayes import MultinomialNB # tfidf 문서분류 
from sklearn.datasets import fetch_20newsgroups # news 데이터셋 
from sklearn.feature_extraction.text import TfidfVectorizer# 희소행렬 
from sklearn.metrics import accuracy_score, confusion_matrix # model 평가 


# 1. dataset 가져오기 
newsgroups = fetch_20newsgroups(subset='all') # train/test load 
# subset = 'all'는 데이터세트에서 사용 가능한 모든 뉴스그룹 게시물을 가져오도록 지정


print(newsgroups.DESCR)
'''
**Data Set Characteristics:**

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========
'''
print(newsgroups.target_names) # 20개 뉴스 그룹(주제)
print(len(newsgroups.target_names)) # 20개 뉴스 그룹 



# 2. train set 선택 : 4개 뉴스 그룹 -> x는 text데이터   
#cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
cats = newsgroups.target_names[:4] # 0 1 2 3 


news_train = fetch_20newsgroups(subset='train',categories=cats)
news_data = news_train.data # X :texts
news_target = news_train.target # y : 0 ~ 3



# 3. sparse matrix - 희소행렬(document term matrix)
obj = TfidfVectorizer() # 문서(texts) 토큰(단어)
# 단어의 빈도수 캡처

sparse_train = obj.fit_transform(news_data) # 문서 -> DTM생성 
sparse_train.shape # (2245, 62227) -> (문서개수, 단어개수)
print(sparse_train)
'''
(문서위치, 단어위치)           가중치
(0, 61522)	         0.07559311503474199
(0, 26590)	         0.07559311503474199
(0, 29842)	         0.0652429131425446
'''



# 4. NB 모델 생성 
nb = MultinomialNB() # alpha=.01 (default=1.0)
model = nb.fit(sparse_train, news_target) # 훈련셋 적용 



# 5. test dataset 4개 뉴스그룹 대상 : 희소행렬
news_test = fetch_20newsgroups(subset='test', categories=cats)
news_test_data = news_test.data # X(texts) -> 자연어
y_true = news_test.target # y(0~3)


sparse_test = obj.transform(news_test_data) # 함수명 주의  
sparse_test.shape 
print(sparse_test)


# 6. model 평가 
y_pred = model.predict(sparse_test) # 예측치 


# 1) accuracy
acc = accuracy_score(y_true, y_pred)
print('accuracy =', acc) 


# 2) confusion matrix
con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)
















