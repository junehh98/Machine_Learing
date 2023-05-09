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
from sklearn.metrics import confusion_matrix, accuracy_score 



# 단계1. wine 데이터셋 로드 
wine = load_wine()

# 단계2. x, y변수 선택 
X = wine.data # x변수 
y = wine.target # y변수


# 단계3. train/test split(70:30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=100)


# 단계4. model 생성  : solver='lbfgs', multi_class='multinomial'
lr = LogisticRegression(random_state=123,
                   solver='lbfgs',
                   max_iter=100, 
                   multi_class='multinomial')

model = lr.fit(X=X_train, y=y_train) # 모델 학습 

# 단계5. 모델 평가 : accuracy, confusion matrix
y_pred = model.predict(X_test)


con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)
'''
[[14  0  0]
 [ 1 18  0]
 [ 1  0 20]]
'''


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy =', accuracy) # Accuracy = 0.9629629629629629



# 단계6. test셋으로 확률 예측하여 class가 2인 관측치만 출력(예시 참고) 


'''
    y정답      y예측치
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



# new_result = result[result['y정답']==2]

























