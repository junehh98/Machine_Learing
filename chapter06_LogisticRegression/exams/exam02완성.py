'''
문2) wine 데이터셋을 이용하여 조건에 맞게 단계별로 로지스틱회귀모델(다항분류)을 생성하시오. 
  조건1> train/test - 70:30비율
  조건2> y 변수 : wine.target 
  조건3> x 변수 : wine.data
  조건4> 모델 평가 : confusion_matrix, 분류정확도[accuracy]
'''

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 단계1. wine 데이터셋 로드 
wine = load_wine()
dir(wine)
'''
['DESCR', : dataset 설명문(wine.DESCR) 
 'data', : 독립변수 로드(wine.data)
 'feature_names', : 독립변수 이름(wine.feature_names)
 'frame', 
 'target', : 종속변수 로드(wine.target)
 'target_names'] : 종속변수 범주(wine.target_names)
'''

# 단계2. x, y변수 선택 
wine_x = wine.data # x변수 
wine_y = wine.target # y변수
wine_x.shape # (178, 13)
print(wine_x)

wine_y.shape # (178,)
wine_y # 0 ~ 2 : label encoding 


# 단계3. train/test split(70:30)
wine_x_train, wine_x_test, wine_y_train, wine_y_test = train_test_split(
    wine_x, wine_y, test_size=0.3, random_state=123)


# 단계4. model 생성  : solver='lbfgs', multi_class='multinomial'
model = LogisticRegression(solver='lbfgs', 
        multi_class='multinomial').fit(X=wine_x_train, y=wine_y_train)

# 단계5. 모델 평가 : accuracy, confusion matrix
y_pred = model.predict(X = wine_x_test) # class 예측 

con_mat = metrics.confusion_matrix(y_true=wine_y_test, 
                         y_pred=y_pred)
con_mat 
'''
array([[13,  1,  0],
       [ 0, 18,  0],
       [ 0,  1, 21]],
'''

acc = metrics.accuracy_score(y_true=wine_y_test, 
                         y_pred=y_pred)
print(acc) # 0.962962962

      
# 단계6. test셋으로 확률 예측하여 class가 2인 관측치만 출력(예시 참고) 
'''
    y정답     y예측치
0     2  0.938881
2     2  0.959511
5     2  0.999500
7     2  0.843381
8     2  0.999861
10    2  0.996617
11    2  0.800701
12    2  0.999426
15    2  0.922597
20    2  0.553276
21    2  0.993744
22    2  0.999964
23    2  0.945606
25    2  0.999076
26    2  0.990768
32    2  0.768173
34    2  0.999338
36    2  0.242703
40    2  0.996522
41    2  0.950416
51    2  0.605581
52    2  0.993588
'''

# 1) y 확률 예측 
y_pred_prob = model.predict_proba(X = wine_x_test)

y_pred_prob.shape # (54, 3) : 3개 class 확률 예측 

y_pred_prob_class2 = y_pred_prob[:, 2] # class 2 확률 예측 추출 

# 2) y 관측치와 y 확률 예측 이용 DF 생성
import pandas as pd  # DataFrame 생성 

result = pd.DataFrame({'y정답': wine_y_test, 
                       'y예측치':y_pred_prob_class2})

result.head()
'''
   y정답      y예측치
0    2  0.938881
1    1  0.084039
2    2  0.959511
3    1  0.020436
4    1  0.000348
'''

# 3) class 2 만 subset 생성 
new_result = result[result['y정답'] == 2] 
print(new_result.head())
'''
   y정답      y예측치
0    2  0.938881
2    2  0.959511
5    2  0.999500
7    2  0.843381
8    2  0.999861
'''







