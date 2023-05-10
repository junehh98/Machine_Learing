# -*- coding: utf-8 -*-
"""
문서 분류기 성능평가
 - NB vs SVM 
"""

import numpy as np 
from sklearn.naive_bayes import MultinomialNB # nb model
from sklearn.svm import SVC  # svm model 
from sklearn.metrics import accuracy_score, confusion_matrix # 평가 
import time 


# file load 
path = r"/Users/junehh98/Desktop/itwill/5_Python_ML/data"
x_train, x_test, y_train, y_test = np.load(path + "/new_spam_data.npy", 
                                           allow_pickle=True)
x_train.shape 
x_test.shape 

#######################
### NB model
#######################
nb = MultinomialNB()

chktime = time.time()
model = nb.fit(X = x_train, y = y_train)
chktime = time.time() - chktime
print('실행 시간 : ', chktime)

y_pred = model.predict(X = x_test) # 예측치 
y_true = y_test # 관측치

acc = accuracy_score(y_true, y_pred)
print('NB 분류정확도 =', acc)

con_mat = confusion_matrix(y_true, y_pred)
print(con_mat)


#######################
### SVM model
#######################
svm = SVC(kernel = 'linear')

chktime = time.time()
model2 = svm.fit(X = x_train, y = y_train)
chktime = time.time() - chktime
print('실행 시간 : ', chktime)

y_pred2 = model2.predict(X = x_test)
acc = accuracy_score(y_true, y_pred2)
print('svm 분류정확도 =', acc)

con_mat = confusion_matrix(y_true, y_pred2)
print(con_mat)
