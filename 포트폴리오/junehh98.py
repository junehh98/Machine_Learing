import FinanceDataReader as fdr
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from googletrans import Translator
from afinn import Afinn
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from nltk.tokenize import word_tokenize


samsung = fdr.DataReader('005930', start = '2021-01-01', end = '2021-12-31')
samsung.isnull().value_counts()
samsung = samsung.drop(['Open','High','Low'], axis = 1)
print(samsung)
type(samsung)



date_samsung = samsung.index
close_samsung = samsung.Close
change_samsung = samsung.Change

samsung_df = pd.DataFrame({'date':date_samsung, 'close' : close_samsung, 
                         'change' : change_samsung*100},
             columns=['date', 'close', 'change',])

samsung_df
len(samsung_df)

samsung_df = samsung_df.reset_index(drop=True)
samsung_df.to_csv('samsung_df', index=False)

print(samsung_df)



############# ##########################크롤링 #######################

stock_code = "005930"
start_date = "20210101"
end_date = "20211231"
titles = []
dates = []
previews = []

current_date = datetime.datetime.strptime(start_date, "%Y%m%d")
end_date = datetime.datetime.strptime(end_date, "%Y%m%d")

while current_date <= end_date:
    if current_date.weekday() >= 5 or current_date.strftime("%Y-%m-%d") in holidays:
        current_date += datetime.timedelta(days=1)
        continue

    for page in tqdm(range(1, 4)):
        url = f"https://finance.naver.com/news/news_search.nhn?rcdate=&q={stock_code}&sm=all.basic&pd=4&stDate={current_date.strftime('%Y%m%d')}&enDate={current_date.strftime('%Y%m%d')}&page={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("dd", class_="articleSubject")
        
        if len(articles) == 0:
            break

        for article in articles:
            title = article.a.text.strip()
            preview_tag = article.find_next_sibling("dt")
            preview = preview_tag.text.strip() if preview_tag else ""
            titles.append(title)
            dates.append(current_date.strftime("%Y-%m-%d"))
            previews.append(preview)

        time.sleep(0.45)

    current_date += datetime.timedelta(days=1)





print(news.isnull().values.any()) #null값이 있는지 체크
news_data = news.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(news_data.isnull().values.any()) # Null 값이 존재하는지 확인

news_data
#############################################################################



# 저장, 불러오기 
news_data.to_csv('news_csv_2021', index=False)
news_df = pd.read_csv('news_csv_2021', encoding = 'utf-8')
news_df



# 분석하기
news_df
samsung_df

len(samsung_df)
news_df["date"].nunique()

samsung_df["date"].nunique()


news_df['date'] = pd.to_datetime(news_df['date'])



# 겹치는 요일만 합치기 
merged_df = pd.merge(samsung_df, news_df, on='date')
merged_df



merged_df = merged_df.groupby(['date', 'change'])['Title'].apply(' '.join).reset_index()
merged_df



# KOR -> ENG 번역, 감성점수 구하기 
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    words = word_tokenize(text)
    stop_words = set(['i', 'the', 'a', 'is', 'that'])
    words = [w for w in words if (w not in stop_words) and (len(w) > 2)]
    text = ' '.join(words)
    return text



def translate_and_sentiment(df): # 번역하고 감성점수 구하기 
    translator = Translator()
    afinn = Afinn(language='en')

    eng_news = []
    sentiments = []
    for title in tqdm(df['Title']):
        try:
            eng_title = translator.translate(title, dest='en').text
            eng_title = preprocess_text(eng_title) # 전처리 함수 적용 
            eng_news.append(eng_title)
            sentiment = afinn.score(eng_title)
            sentiments.append(sentiment)
        except Exception as e:
            print(e)
            eng_news.append('')
            sentiments.append(0)

    df['Translated Title'] = eng_news
    df['Sentiment'] = sentiments

    return df

merged_df = translate_and_sentiment(merged_df) # 함수적용 


'''
AFINN은  -5에서 +5 사이의 숫자로 점수가 매겨진 단어 목록  0 ~ 100사이의 결과가 나옴 
한계점 : 긍정, 부정 기준을 잡기가 애매함
아직 활성화된 한국어 감성사전이 없어서 번역해서 감성분석을 해야하는게 부정확함
같은 단어가 긍정일수도 부정일수도 있음 
주식 시장의 단어사전이 아닌 일반 단어사전을 사용해서 정확도가 낮음 
'''

mean_sentiment = merged_df['Sentiment'].mean()
median_sentiment = merged_df['Sentiment'].median()
print(mean_sentiment) # 43.891129032258064
print(median_sentiment) # 45


print(merged_df[merged_df['Sentiment'] >= 40])
print(merged_df[merged_df['Sentiment'] < 40])


merged_df['score'] = np.where((merged_df['Sentiment'] >= 40) & (merged_df['change'] >= 0), 'TRUE', 'FALSE')
merged_df

merged_df.info()
merged_df
merged_df = merged_df.drop(['Title','Translated Title'], axis = 1)


merged_df.to_csv("merged_df", index = False)
merged_df = pd.read_csv('merged_df', encoding = 'utf-8')




len(merged_df[merged_df['score'] == True]) # True 개수
'''
   실제로 변동율이 올랐을때 감성점수가 긍정이거나
   변동율이 떨어졌을때 감성점수가 부정인 행 -> 95개 
'''


len(merged_df[merged_df['score'] == False]) # False 개수  -> 153개

# 정확도 구하기 
acc = len(merged_df[merged_df['score'] == True])/(len(merged_df[merged_df['score'] == True])+len(merged_df[merged_df['score'] == False]))
print(round(acc, 2))#  약 38%


corr = pearsonr(merged_df['change'],merged_df['Sentiment'])
corr
# PearsonRResult(statistic=-0.05333039172616527, pvalue=0.40304231375608907)

sns.regplot(x='change', y='Sentiment', data=merged_df)
plt.title(f'Pearson Correlation: {corr:.2f}')
plt.show()