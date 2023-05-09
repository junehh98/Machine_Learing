# -*- coding: utf-8 -*-
"""
step02_logisticRegression_multiClass.py

 - 다항분류기(multi class classifier)  
"""
from sklearn.datasets import load_digits # dataset
from sklearn.linear_model import LogisticRegression # model 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import confusion_matrix, accuracy_score # model 평가 


# 1. dataset loading 
digits = load_digits()

image = digits.data # x변수 
label = digits.target # y변수 

image.shape # (1797, 64) : (size, 픽셀)
image.max() # 16.0
image.min() # 0.0

label.shape # (1797,) : 정답(0~9)
label # [0, 1, 2, ..., 8, 9, 8]

# 2. train_test_split
img_train, img_test, lab_train, lab_test = train_test_split(
                 image, label, 
                 test_size=0.3, 
                 random_state=123)


# 3. model 생성 
lr = LogisticRegression(random_state=123,
                   solver='lbfgs',
                   max_iter=100, 
                   multi_class='auto')
'''
multi_class : {'auto', 'ovr', 'multinomial'}

multi_class='auto' : 다항분류(multinomial)
multi_class='ovr' : 이항분류기(sigmoid function) 
multi_class='multinomial' : 다항분류기(softmax function) 
'''

model = lr.fit(X=img_train, y=lab_train) # multi_class='multinomial'


# 4. model 평가 
y_pred = model.predict(img_test) # class 예측 

# 1) 혼동행렬(confusion matrix)
con_mat = confusion_matrix(lab_test, y_pred)
con_mat
'''
array([[59,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0, 55,  0,  0,  1,  0,  0,  0,  0,  0],
       [ 0,  0, 53,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 45,  0,  0,  0,  0,  0,  1],
       [ 0,  0,  0,  0, 60,  0,  0,  1,  0,  0],
       [ 0,  0,  0,  0,  0, 52,  0,  2,  0,  3],
       [ 0,  1,  0,  0,  0,  1, 55,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0,  0, 50,  0,  0],
       [ 0,  4,  0,  0,  1,  0,  0,  0, 43,  0],
       [ 0,  1,  0,  0,  0,  0,  0,  0,  2, 50]],
'''
      
# 2) 분류정확도(Accuracy)
accuracy = accuracy_score(lab_test, y_pred)
print('Accuracy =', accuracy) 
# Accuracy = 0.9666666666666667


# 3) heatmap 시각화 
import matplotlib.pyplot as plt
import seaborn as sn
  
# confusion matrix heatmap 
plt.figure(figsize=(6,6)) # size
sn.heatmap(con_mat, annot=True, fmt=".3f",
           linewidths=.5, square = True) 
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: ', format(accuracy,'.6f')
plt.title(all_sample_title, size = 18)
plt.show()

