"""
특징변수 데이터변환(features scaling) : 이물질 제거 

 1. 특징변수(x변수) : 값의 크기(scale)에 따라 model 영향을 미치는 경우
      ex) 범죄율(-0.01~0.99), 주택가격(99~999)
   1) 표준화 : X변수를 대상으로 정규분포가 될 수 있도록 평균=0, 표준편차=1로 통일 시킴 
      -> 회귀모델, SVM 계열은 X변수가 정규분포라고 가정하에 학습이 진행되므로 표준화를 적용 
      
   2) 최소-최대 정규화 : 서로 다른 척도(값의 범위)를 갖는 X변수를 대상으로 최솟값=0, 최댓값=1로 통일 시킴 
      -> 트리모델 계열(회귀모델 계열이 아닌 경우)에서 서로 다른 척도를 갖는 경우 적용 


 2. 타깃변수(y변수) : 로그변환(log1p() 함수 이용 ) 
"""



from sklearn.datasets import load_boston # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.linear_model import LinearRegression # model 
from sklearn.metrics import mean_squared_error, r2_score # 평가 
from sklearn.datasets import fetch_openml


from sklearn.preprocessing import minmax_scale # 정규화 : 트리계열
from scipy.stats import zscore # 표준화 : 회귀계열
import numpy as np # 로그변환 : y변수


boston = fetch_openml(name='boston', version=1)

# 1. dataset load
X = boston.data
y = boston.target 

X.shape # (506, 13)
y.shape # (506,)


# x,y변수 스케일링 안됨 
X.mean() # 70.07396704469443
X.max() # 711.0
X.min() # 0.0

y.mean() # 22.532806324110677
y.max() # 50.0
y.min() # 5.0



# 2. 피처스케일링(features scaling)  
def scaling(X, y, kind='none') : 
    # 1. x변수 스케일링 선택   
    if kind == 'minmax_scale' :  
        X_trans = minmax_scale(X) 
    elif kind == 'zscore' : 
        X_trans = zscore(X)   
    else :
        X_trans = X 
    
    # 2. y변수 : 로그변환 
    if kind != 'none' :
        y = np.log1p(np.abs(y))   
    
    # 3. train/test split 
    X_train,X_test,y_train,y_test = train_test_split(
        X_trans, y, test_size = 30, random_state=1)   
    
    print(f"scaling 방법 : {kind}, X 평균 = {X_trans.mean()}")
    return X_train, X_test, y_train, y_test


# 함수 호출 : 스케일링 방법 선택 
X_train, X_test, y_train, y_test = scaling(X, y, 'none')



# 3. model 생성하기
model = LinearRegression().fit(X=X_train, y=y_train) # 지도학습 


# 4. model 평가하기
model_train_score = model.score(X_train, y_train) 
model_test_score = model.score(X_test, y_test) 
print('model train score =', model_train_score)
print('model test score =', model_test_score)


y_pred = model.predict(X_test)
y_true = y_test
print('R2 score =',r2_score(y_true, y_pred))  
mse = mean_squared_error(y_true, y_pred)
print('MSE =', mse)




