'''
Naive Bayes 이론에 근거한 통계적 분류기
 1. GaussianNB  : x변수가 연속형이고, 정규분포인 경우 
 2. BernoulliNB  : x변수가 이진(binary) 데이터인 경우(x변수가 0과 1인 더미변수)
 3. MultinomialNB : x변수가 단어 빈도수(텍스트 데이터)를 분류할 때 적합

관련 문서 : 
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html 
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
'''


'''
사전확률 : P(A), P(B) - 사건이 발생하기 전에 사전에 알고 있는 확률 
사후확률(결합확률) : P(A|B) - 두 사건 A와 B가 결합하여 일어날 확률
베이즈 확률 이론 : 사전확률과 사후확률을 이용한 조건부 확률  
  P(B|A) = P(A|B).P(B) / P(A) : 확률의 곱셈 법칙 
'''


  
'''
 <SPAM/HAM메시지 vs VIAGRA단어 관계>   (ppt.13 참고)
-----------------------------------
         YES       NO      행합계   
-----------------------------------         
SPAM      4/20   16/20    20/100
HAM       1/80   79/80    80/100
-----------------------------------
 열합계   5/100  95/100    100
----------------------------------- 
''' 

##################################################################
### Example1) 비아그라 단어(A)가 포함된 메시지가 스팸(B)일 확률은?
##################################################################

'''
 조건부확률 표현 : P(SPAM|VIAGRA_YES) = P(VIAGRA_YES|SPAM) * P(SPAM) / P(VIAGRA_YES)
'''



# 단계1. 사전확률 : 사전에 알고 있는 확률 


# P(SPAM) 확률 
Pspam = 20/100  #0.2
# P(HAM) 확률 
Pham = 80/100 # 0.8


# P(VIAGRA YES) 확률 
Pviagra_YES = 5/100 # 0.05 
# P(VIAGRA NO) 확률 
Pviagra_NO = 95/100 # 0.95



# 단계2. 배반사건 : 전체 합 = 1 
P_tot1 = Pspam + Pham  
P_tot2 = Pviagra_YES + Pviagra_NO 


 
# 단계3. 사후확률(결합확률) : P(VIAGRA_YES|SPAM)
Pviagra_YES_spam = 4/20



# 단계4. 베이즈 확률 이론(조건부확률)
# P(SPAM|VIAGRA_YES) = P(VIAGRA_YES|SPAM) * P(SPAM) / P(VIAGRA_YES)

P = Pviagra_YES_spam * Pspam / Pviagra_YES
print('P(SPAM|VIAGRA_YES) = {0:.2f}%'.format(P*100)) # P(SPAM|VIAGRA_YES) = 80.00%



#####################################################################
### Example2) 비아그라 단어(A)가 포함되지 않은 메시지가 햄(B)일 확률은?
#####################################################################

'''
 조건부확률 표현 : P(HAM|VIAGRA_NO) = P(VIAGRA_NO|HAM) * P(HAM) / P(VIAGRA_NO)
'''



# 단계1. 사전확률 : 사전에 알고 있는 확률 

# P(SPAM) 확률 
Pspam = 20/100 # 0.2
# P(HAM) 확률 
Pham = 80/100 # 0.8

# P(VIAGRA YES) 확률 
Pviagra_YES = 5/100 # 0.05
# P(VIAGRA NO) 확률 
Pviagra_NO = 95/100 # 0.95




# 단계2. 배반사건(둘 중 하나의 상태만 가능한 경우) 전체 합 = 1 
P_tot1 = Pspam + Pham # 1.0 
P_tot2 = Pviagra_YES + Pviagra_NO # 1.0


 
# 단계3. 사후확률(결합확률) : P(VIAGRA_NO|HAM)
Pviagra_NO_ham = None


# 단계4. 베이즈 확률 이론(조건부확률)
# P(HAM|VIAGRA_NO) = P(VIAGRA_NO|HAM) * P(HAM) / P(VIAGRA_NO)

P = None
print('P(HAM|VIAGRA_NO) = {0:.2f}%'.format(P*100)) 
















