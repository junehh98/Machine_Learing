# -*- coding: utf-8 -*-
"""
step05_image_reshape.py

1. image shape & reshape 
2. image file read & show
"""

import matplotlib.pyplot as plt  
from sklearn.datasets import load_digits 


# 1. image shape & reshape

# 1) dataset load 
digits = load_digits() # 머신러닝 모델에서 사용되는 데이터셋 
'''
입력변수(X) : 숫자(0~9) 필기체의 흑백 이미지 
출력변수(y) : 10진 정수
'''
dir(digits)

X = digits.data # 입력변수(X) 추출 
y = digits.target # 출력변수(y) 추출 
X.shape # (1797, 64) : (size, fixel)
y.shape # (1797,)
type(X) # numpy.ndarray

X[0].shape # (64,) : 1d
X[0].max() # 0~15


# 2) image reshape 
first_img = X[3].reshape(8,8) # 모양변경 : 2d 
first_img.shape # (8, 8) # 2차원으로 변경해야 시각화가 가능함 


# 3) image show 
plt.imshow(X=first_img, cmap='gray')
plt.show()

y[0] # 0
y[3] # 3


# 전체 이미지 모양 변경(1d -> 2d)
X_2d = X.reshape(-1, 8, 8) # -1 : 전체 이미지 
X_2d.shape # (1797, 8, 8)

plt.imshow(X=X_2d[-1], cmap='gray')
plt.show()

y[-1] # 8



# 2. image file read & show
import matplotlib.image as img # 이미지 읽기 

# image file path 
path = r"C:/ITWILL/5_Python_ML/workspace/chapter03_Numpy/data" # 이미지 경로 

# 1) image 읽기 
img_arr = img.imread(path + "/test1.jpg")
img_arr
type(img_arr) # numpy.ndarray
img_arr.shape # (360, 540, 3) : (h, w, c)


# 2) image 출력 
plt.imshow(img_arr)
plt.show()






















