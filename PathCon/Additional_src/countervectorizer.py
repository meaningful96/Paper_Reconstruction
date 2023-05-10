"""
Created on meaningful96

DL Project
"""

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

x_data = np.array([
  'me free lottery',
  'free get free you',
  'you free scholarship',
  'free to contact me',
  'you won award',
  'you ticket lottery',
  'you ticket lottery',
  'you ticket lottery',
  'you ticket lottery',
  'you ticket lottery'
])
y_data = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

nltk.download('punkt')
nltk.download('stopwords')
#print(stopwords.fileids()) #['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'kazakh', 'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish'] #불용어를 제공하는 국가의 언어
#print(stopwords.words('english')[:10]) #['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"
for i, document in enumerate(x_data):
    document = BeautifulSoup(document, "html.parser").text #HTML 태그 제거
    words = word_tokenize(document)
    #print(words) #
    clean_words = []
    for word in words:
        word = word.lower()
        if word not in stopwords.words('english'): #불용어 제거
            stemmer = SnowballStemmer('english')
            word = stemmer.stem(word) #어간 추출
            clean_words.append(word)
    #print(clean_words) #['봄', '신제품', '소식']
    document = ' '.join(clean_words)
    x_data[i] = document
    
vectorizer = CountVectorizer()
vectorizer.fit(x_data)

x_data = vectorizer.transform(x_data)
print(x_data.shape) #(10, 7) 

print(vectorizer)
print(vectorizer.vocabulary_)

print(x_data)