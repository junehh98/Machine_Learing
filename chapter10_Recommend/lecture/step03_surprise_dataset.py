'''
surprise Dataset 이용 
'''
import pandas as pd # DataFrame 생성 
from surprise import SVD, accuracy # SVD model 생성, 평가  
from surprise import Reader, Dataset # SVD dataset 생성  

############################
## suprise Dataset
############################

# 1. dataset loading 
ratings = pd.read_csv('C:/ITWILL/5_Python_ML/data/u.data', sep='\t', header=None)
print(ratings) 


# 칼럼명 수정 (유저ID, 영화ID, 평점, timestamp )
ratings.columns = ['userId','movieId','rating','timestamp']
ratings.info() 
'''
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 4 columns):
 #   Column     Non-Null Count   Dtype
---  ------     --------------   -----
 0   userId     100000 non-null  int64
 1   movieId    100000 non-null  int64
 2   rating     100000 non-null  int64
 3   timestamp  100000 non-null  int64
'''

ratings = ratings.drop('timestamp', axis = 1)

ratings.rating.value_counts()
'''
4    34174
3    27145
5    21201
2    11370
1     6110  --> 1 ~ 5까지의 범위 
'''



# 2. pivot table 작성 : row(영화제목), column(평가자), cell(평점)
movie_ratings = pd.pivot_table(ratings,
               index = 'userId',
               columns = 'movieId',
               values = 'rating')#.reset_index()

movie_ratings.shape # (943, 1683) --> 943명의 user, 1683개의 영화
                    # (943, 1682) -> reset.index 삭제 (user, item)

movie_ratings.isnull().sum()



# 3. SVD dataset 
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings, reader)



# 4. train/test split
from surprise.model_selection import train_test_split

# Dataset 자료이용
trainset, testset = train_test_split(data, random_state=0)



# 5. svd model
svd_model= SVD(random_state=123).fit(trainset)
dir(svd_model)
'''
 test() : 전체 test 사용자 평점예측(관람영화)
 predict() : 특정한 사용자에 평점 예측 
'''



# 5. 전체 testset 평점 예측
preds = svd_model.test(testset) # 실제평점과 예측된 평점
print(len(preds)) # 20,000  --> 80,000개는 train으로 사용 

# 예측결과 출력 
print('user\tmovie\trating\test_rating')
for p in preds[:5] : 
    print(p.uid, p.iid, p.r_ui, p.est, sep='\t\t')
'''
user	movie	rating	est_rating
269		17		2.0		2.697369252580903
704		382		4.0		3.425149329906973
829		475		4.0		3.8548670627807327
747		274		4.0		3.9399633165611663
767		98		5.0		4.8264822102570335
'''
    

    
# 6. model 평가 
accuracy.mse(preds)  # 0.8976351072297628 
accuracy.rmse(preds) # 0.947436070260027



# 7.추천대상자 평점 예측 
type(movie_ratings) # pandas.core.frame.DataFrame

movie_ratings.iloc[:5, :10] # [행(user), 열(item)]
'''
movieId   1    2    3    4    5    6    7    8    9    10
userId                                                   
1        5.0  3.0  4.0  3.0  3.0  5.0  4.0  1.0  5.0  3.0
2        4.0  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  2.0
3        NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
4        NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
5        4.0  3.0  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
'''

# 1) 추천 대상자 선정  userID = 5

# 2) 미관람 아이템 선정 movieID = 1~10 
uid = '5'
items = movie_ratings.columns[:10] # 추천 item
actual_rating = movie_ratings.iloc[4,:10].fillna(0) # NaN값은 0으로 초기화


for iid, ar in zip(items, actual_rating) :
    svd_pred = svd_model.predict(uid, iid, ar)
    print(svd_pred)
'''
user: 5          item: 1          r_ui = 4.00   est = 3.96   {'was_impossible': False}
user: 5          item: 2          r_ui = 3.00   est = 3.37   {'was_impossible': False}
user: 5          item: 3          r_ui = 0.00   est = 3.23   {'was_impossible': False}
user: 5          item: 4          r_ui = 0.00   est = 3.63   {'was_impossible': False}
user: 5          item: 5          r_ui = 0.00   est = 3.38   {'was_impossible': False}
user: 5          item: 6          r_ui = 0.00   est = 3.83   {'was_impossible': False}
user: 5          item: 7          r_ui = 0.00   est = 3.83   {'was_impossible': False}
user: 5(추천)    item: 8          r_ui = 0.00   est = 4.04   {'was_impossible': False}
user: 5          item: 9          r_ui = 0.00   est = 4.02   {'was_impossible': False}
user: 5          item: 10         r_ui = 0.00   est = 3.93   {'was_impossible': False}
'''





















