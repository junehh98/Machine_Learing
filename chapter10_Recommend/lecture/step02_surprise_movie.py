# -*- coding: utf-8 -*-
"""
- 특이값 분해(SVD) 알고리즘 이용 추천 시스템
- scikit-surprise 패키지 설치 
  conda install -c conda-forge scikit-surprise
"""

import pandas as pd # csv file 
from surprise import SVD # SVD model 
from surprise import Reader, Dataset # SVD dataset 


# 1. dataset loading 
ratings = pd.read_csv('C:/ITWILL/5_Python_ML/data/movie_rating.csv')
print(ratings) #  평가자[critic]   영화[title]  평점[rating]


# 2. pivot table 작성 : row(영화제목), column(평가자), cell(평점)
print('movie_ratings')
movie_ratings = pd.pivot_table(ratings,
               index = 'title',
               columns = 'critic',
               values = 'rating').reset_index()


# 3. SVD dataset 
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings, reader)


# 4. train/test set 생성 
trainset = data.build_full_trainset() # 훈련셋 
testset = trainset.build_testset() # 검정셋 


# 5. SVD model 생성 
model = SVD(random_state=123).fit(trainset) # seed값 적용 


# 6. 전체 사용자 평점 예측치 
all_pred = model.test(testset)
print(all_pred)


# 7. Toby 사용자 미관람 영화 추천 예측 
user_id  = 'Toby' # 추천 대상자 
items = ['Just My','Lady','The Night']   
actual_rating = 0

for item_id in items :
    svd_pred = model.predict(user_id, item_id, actual_rating)
    print(svd_pred)









