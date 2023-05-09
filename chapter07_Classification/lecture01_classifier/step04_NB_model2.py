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
print(newsgroups.target_names) # 20개 뉴스 그룹 
print(len(newsgroups.target_names)) # 20개 뉴스 그룹 


# 2. train set 선택 : 4개 뉴스 그룹  
#cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
cats = newsgroups.target_names[:4]

news_train = fetch_20newsgroups(subset='train',categories=cats)
news_data = news_train.data # texts
news_target = news_train.target # 0 ~ 3


# 3. sparse matrix
obj = TfidfVectorizer()
sparse_train = obj.fit_transform(news_data)
sparse_train.shape # (2245, 62227)


# 4. NB 모델 생성 
nb = MultinomialNB() # alpha=.01 (default=1.0)
model = nb.fit(sparse_train, news_target) # 훈련셋 적용 


# 5. test dataset 4개 뉴스그룹 대상 : 희소행렬
news_test = fetch_20newsgroups(subset='test', categories=cats)
news_test_data = news_test.data
y_true = news_test.target

sparse_test = obj.transform(news_test_data) # 함수명 주의  
sparse_test.shape 


# 6. model 평가 
y_pred = model.predict(sparse_test) # 예측치 

acc = accuracy_score(y_true, y_pred)
print('accuracy =', acc) 

# 2) confusion matrix
con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)

