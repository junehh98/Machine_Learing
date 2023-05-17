# -*- coding: utf-8 -*-
"""
문) food 데이터셋을 대상으로 작성된 피벗테이블(pivot table)을 보고 'g' 사용자가 아직
    섭취하지 않은 음식을 대상으로 추천하는 모델을 생성하고, 추천 결과를 확인하시오. 
"""

import pandas as pd
from surprise import SVD, accuracy # SVD model 생성 
from surprise import Reader, Dataset # SVD data set 생성 


# 1. 데이터 가져오기 
food = pd.read_csv('C:/ITWILL/5_Python_ML/data/food.csv')
print(food.info()) #    uid(user)  menu(item) count


# 2. 피벗테이블 작성 
ptable = pd.pivot_table(food, 
                        values='count',
                        index='uid',
                        columns='menu', 
                        aggfunc= 'mean')

ptable.shape # (7, 5)
ptable.values
ptable.info()


# 3. rating 데이터셋 생성    
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(food, reader)


# 4. train/test set 생성 
from surprise.model_selection import train_test_split

# Dataset 자료이용
trainset, testset = train_test_split(data, random_state=100)


# 5. model 생성 : train set 이용 
svd_model= SVD(random_state=100).fit(trainset)

preds = svd_model.test(testset)


accuracy.mse(preds) 
accuracy.rmse(preds)



# 6. 'g' 사용자 대상 음식 추천 예측 
user_id  = 'g' 
items = ['식빵','우유','치킨']   
actual_rating = 0

for item_id in items :
    svd_pred = svd_model.predict(user_id, item_id, actual_rating)
    print(svd_pred)
'''
user: g          item: 식빵         r_ui = 0.00   est = 3.21   {'was_impossible': False}
user: g          item: 우유         r_ui = 0.00   est = 2.98   {'was_impossible': False}
user: g          item: 치킨         r_ui = 0.00   est = 3.14   {'was_impossible': False}
'''



