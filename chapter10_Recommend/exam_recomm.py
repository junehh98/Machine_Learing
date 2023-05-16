# -*- coding: utf-8 -*-
"""
문) food 데이터셋을 대상으로 작성된 피벗테이블(pivot table)을 보고 'g' 사용자가 아직
    섭취하지 않은 음식을 대상으로 추천하는 모델을 생성하고, 추천 결과를 확인하시오. 
"""

import pandas as pd
from surprise import SVD # SVD model 생성 
from surprise import Reader, Dataset # SVD data set 생성 


# 1. 데이터 가져오기 
food = pd.read_csv('C:/ITWILL/5_Python_ML/data/food.csv')
print(food.info()) #    uid(user)  menu(item) count


# 2. 피벗테이블 작성 
ptable = pd.pivot_table(food, 
                        values='count',
                        index='uid',
                        columns='menu', 
                        aggfunc= 'sum')
ptable

# 3. rating 데이터셋 생성    


# 4. train/test set 생성 


# 5. model 생성 : train set 이용 


# 6. 'g' 사용자 대상 음식 추천 예측 





