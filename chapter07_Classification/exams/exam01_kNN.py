'''
문1) 다음과 같은 알려진 집단(2개의 집단을 갖는 4개의 데이터 셋)을 대상으로 
     kNN 알고리즘을 적용하여 분류대상이 어떤 집단에 분류되는지를 확인하시오.
     (단 최근접이웃 k=3 적용) - [ppt 5 페이지 참고] 
     
     <출력 결과>
     분류 결과 :  B
'''

# 1. 데이터셋 구성 

# 1) 알려진 집단 
p1 = [1.2, 1.1] # A집단 
p2 = [1.0, 1.0] # A집단 
p3 = [1.8, 0.8] # B집단
p4 = [2, 0.9]   # B집단


# 2) 알려지지 않은 집단 : 분류대상
p5 = [1.6, 0.85] 


# 3) y변수 : 알려진 집단의 클래스 
y_class = [0, 0, 1, 1] 


# 4) 알려진 집단의 클래스 이름(class name) 
class_label = ['A','B'] 


# 2. 분류기 생성 
from sklearn.neighbors import KNeighborsClassifier # class


know = [p1,p2,p3,p4] # 알려진 집단 
unKnow = [p5]# 알려지지 않은 집단 

knn = KNeighborsClassifier(n_neighbors = 3)  
model = knn.fit(X = know, y = y_class) 


# 3. 분류기 평가
y_pred = model.predict(X = unKnow) # 분류대상 
print(y_pred) # [1]

idx = y_pred[0]

print('분류결과:', class_label[idx])





