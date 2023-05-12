'''
- XGBoost 앙상블 모델 테스트
- Anaconda Prompt에서 패키지 설치 
  pip install xgboost
'''

from xgboost import XGBClassifier # model
from xgboost import plot_importance # 중요변수(x) 시각화  
from sklearn.datasets import make_blobs # 클러스터 생성 dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# 1. 데이터셋 로드 : blobs
X, y = make_blobs(n_samples=2000, n_features=4, centers=3, # 이항분류시 centers 2
                  cluster_std=2.5, random_state=123)

'''
n_samples : 표본개수
n_features : x변수 개수
centers : y class 개수
cluster_std : 데이터 분류의 복잡도 
'''

X.shape # (2000, 4)
y.shape # (2000,)
y # array([1, 1, 0, ..., 0, 0, 2])


# blobs 데이터 분포 시각화 
plt.title("three cluster dataset")
plt.scatter(X[:, 0], X[:, 1], s=100, c=y,  marker='o') # color = y범주
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


# 2. 훈련/검정 데이터셋 생성
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)


# 3. XGBOOST model 
xgb = XGBClassifier(objective='multi:softprob') # 이항분류시 바꿀부분
'''
y class = 2 : objective='binary:logistic' : sigmoid 활성함수
y class = 3 : objective='multi:softprob' : softmax 활성함수 
'''

'''
objective={'binary:logistic', 'multi:softprob'}
'''


# train data 이용 model 생성 
model = xgb.fit(X_train, y_train, eval_metric='merror') 
'''
eval_metric={'error', 'merror'}
이항분류 : eval_metric={'error'} - y가 2개
다항분류 : eval_metric={'merror'} - y가 3개이상 
'''
print(model) # 사용한 parameter 정보 확인


# 4. model 평가 
y_pred = model.predict(X_test) 
acc = accuracy_score(y_test, y_pred)
print('분류정확도 =', acc)
# 분류정확도 = 0.9066666666666666


report = classification_report(y_test, y_pred)
print(report)


# 5. fscore 중요변수 시각화  
fscore = model.get_booster().get_score(importance_type='gain') # get_fscore()

print("fscore:",fscore) 
# fscore: {'f0': 790.0, 'f1': 707.0, 'f2': 976.0, 'f3': 777.0} # 높을수록 중요한
# fscore: {'f0': 0.0011542499996721745, 'f1': 0.013701604679226875, 'f2': 65.74845123291016, 'f3': 0.056733760982751846}

# 중요변수 시각화
plot_importance(model) 
plt.show()



