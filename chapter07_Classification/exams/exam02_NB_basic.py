'''
 문2) 다음과 같은 날씨와 비 관계를 나타낸 테이블을 참고하여 Naive Bayes 확률이론을 
      적용한 주어진 확률을 구하시오. 


  <날씨와 비 관계 테이블> ppt.14 참고 
--------------------------
         yes   no   행합  
--------------------------         
맑은날    2     8    10
흐린날    6     4    10
--------------------------
  열합    8     12   20
--------------------------  
''' 

### [문제1] 맑은날 비가 올 확률 구하기 
'''
 조건부확률 표현 : P(yes|맑은날) = P(맑은날|yes) * P(yes) / P(맑은날)
'''


# 단계1. 사전확률  
'''
P(맑은날) = 10/20 : 사건 A, 0.2
P(yes) = 8/20  : 사건 B, 0.8
'''

# 단계2. 결합확률 
'''
P(맑은날|yes) = 0.25 * 0.4 /0.5
'''

# 단계3. 베이즈 확률 이론(조건부확률) : P(B|A) = P(A|B).P(B) / P(A) 
'''
 P(yes|맑은날) = P(맑은날|yes) * P(yes) / P(맑은날)
'''

P =  0.25 * 0.4 /0.5 # 확률 계산 
print(f'P(yes|맑은날) = {P*100}%')  # 결과 출력 





### [문제2] 흐린날 비가 안올 확률 구하기 
'''
조건부확률 표현 : P(no|흐린날) = P(흐린날|no) * P(no) / P(흐린날)
'''

# 단계1. 사전확률  
'''
P(흐린날) = 10/20, 0.5
P(no) = 12/20, 0.6
'''

# 단계2. 결합확률 
'''
P(흐린날|no) = 4/12 = 0.33
'''


# 단계3. 베이즈 확률 이론(조건부확률)
'''
 P(no|흐린날) = P(흐린날|no) * P(no) / P(흐린날)
'''
 
P =  0.33 * 0.6 /0.5 # 확률 계산 
print(f'P(no|흐린날) = {P*100}%') # 결과 출력 




