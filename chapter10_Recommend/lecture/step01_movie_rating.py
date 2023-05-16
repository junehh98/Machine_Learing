'''
영화 추천 시스템 알고리즘
 - 추천 대상자 : Toby   
 - 유사도 평점 = 미관람 영화평점 * Toby와의 유사도
 - 추천 영화 예측 = 유사도 평점 / Toby와의 유사도
'''

import pandas as pd

# 데이터 가져오기 
ratings = pd.read_csv(r'C:\ITWILL\5_Python_ML\data\movie_rating.csv')
print(ratings) 





### 1. pivot table 작성 : row(영화제목), column(평가자), cell(평점)
movie_ratings = pd.pivot_table(ratings,
               index = 'title', # 행
               columns = 'critic', # 열 
               values = 'rating').reset_index()

print(movie_ratings)  
'''
critic      title  Claudia  Gene  Jack  Lisa  Mick  Toby
0         Just My      3.0   1.5   NaN   3.0   2.0   NaN
1            Lady      NaN   3.0   3.0   2.5   3.0   NaN
2          Snakes      3.5   3.5   4.0   3.5   4.0   4.5
3        Superman      4.0   5.0   5.0   3.5   3.0   4.0
4       The Night      4.5   3.0   3.0   3.0   3.0   NaN
5          You Me      2.5   3.5   3.5   2.5   2.0   1.0
'''




### 2. 사용자 유사도 계산(상관계수 R)  
sim_users = movie_ratings.corr().reset_index() # corr(method='pearson')
print(sim_users) 
'''
critic   critic   Claudia      Gene      Jack      Lisa      Mick      Toby
0       Claudia  1.000000  0.314970  0.028571  0.566947  0.566947  0.893405
1          Gene  0.314970  1.000000  0.963796  0.396059  0.411765  0.381246
2          Jack  0.028571  0.963796  1.000000  0.747018  0.211289  0.662849
3          Lisa  0.566947  0.396059  0.747018  1.000000  0.594089  0.991241
4          Mick  0.566947  0.411765  0.211289  0.594089  1.000000  0.924473
5          Toby  0.893405  0.381246  0.662849  0.991241  0.924473  1.000000
'''



### 3. Toby 미관람 영화 추출  
# 1) movie_ratings table에서 title, Toby 칼럼으로 subset 작성 
toby_rating = movie_ratings[['title', 'Toby']]  
print(toby_rating)
'''
critic title     Toby
0    Just My     NaN
1       Lady     NaN
2     Snakes     4.5
3   Superman     4.0
4  The Night     NaN
5     You Me     1.0
'''



# 2) Toby 미관람 영화제목 추출 
# 형식) DF.칼럼[DF.칼럼.isnull()]
toby_not_see = toby_rating.title[toby_rating.Toby.isnull()] 
print(toby_not_see) # rating null 조건으로 title 추출 
'''
0      Just My
1         Lady
4    The Night
'''
type(toby_not_see)




# 3) raw data에서 Toby 미관람 영화만 subset 생성 
rating_t = ratings[ratings.title.isin(toby_not_see)] # 3편 영화제목 
print(rating_t)
'''
     critic      title  rating
0      Jack       Lady     3.0
4      Jack  The Night     3.0
5      Mick       Lady     3.0
:
30     Gene  The Night     3.0
'''



# 4. Toby 미관람 영화 + Toby 유사도 join
# 1) Toby 유사도 추출 
toby_sim = sim_users[['critic','Toby']] # critic vs Toby 유사도 
toby_sim
'''
critic   critic      Toby
0       Claudia  0.893405
1          Gene  0.381246
2          Jack  0.662849
3          Lisa  0.991241
4          Mick  0.924473
5          Toby  1.000000
'''

# 2) 평가자 기준 병합  
rating_t = pd.merge(rating_t, toby_sim, on='critic')
print(rating_t)
'''
     critic      title  rating      Toby
0      Jack       Lady     3.0  0.662849
1      Jack  The Night     3.0  0.662849
2      Mick       Lady     3.0  0.924473
'''



### 5. 유사도 평점 계산 = Toby미관람 영화 평점 * Tody 유사도 
rating_t['sim_rating'] = rating_t.rating * rating_t.Toby
print(rating_t)
'''
     critic      title  rating      Toby  sim_rating
0      Jack       Lady     3.0    0.662849    1.988547
1      Jack  The Night     3.0    0.662849    1.988547
2      Mick       Lady     3.0    0.924473    2.773420
[해설] Toby 미관람 영화평점과 Tody유사도가 클 수록 유사도 평점은 커진다.
'''



### 6. 영화제목별 rating, Toby유사도, 유사도평점의 합계
group_sum = rating_t.groupby(['title']).sum() # 영화 제목별 합계
'''
           rating      Toby  sim_rating
title                                  
Just My       9.5  3.190366    8.074754
Lady         11.5  2.959810    8.383808
The Night    16.5  3.853215   12.899752
'''
 


### 7. Toby 영화추천 예측 = 유사도평점합계 / Tody유사도합계
print('\n*** 영화 추천 결과 ***')
group_sum['predict'] = group_sum.sim_rating / group_sum.Toby
print(group_sum)
'''
           rating      Toby  sim_rating   predict
title                                            
Just My       9.5  3.190366    8.074754  2.530981
Lady         11.5  2.959810    8.383808  2.832550
The Night    16.5  3.853215   12.899752  3.347790 -> 추천영화 
'''
