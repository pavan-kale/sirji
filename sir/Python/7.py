import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
text="To understand the need for creating a class let’s consider an example, let’s say you wanted to track the number of dogs that may have different attributes like breed, and age. If a list is used, the first element could be the dog’s breed while the second element could represent its age. Let’s suppose there are 100 different dogs, then how would you know which element is supposed to be which? What if you wanted to add other properties to these dogs? This lacks organization and it’s the exact need for classes."
tokens=word_tokenize(text)
lemmatizer=WordNetLemmatizer()
lemmatized_tokens=[lemmatizer.lemmatize(token) for token in tokens]
pos_tags=pos_tag(tokens)
stop_words=set(stopwords.words('english'))
filtered_tokens=[token for token in lemmatized_tokens if token.lower()not in stop_words]
print("===============Original Text================= ")
print(text)
print("================Tokenized Text===============")
print(tokens)
print("================Lemmatized Token================== ")
print(lemmatized_tokens)
print("===============Pos Tagging======================")
print(pos_tags)
print("================Stop words removal===============")
print(filtered_tokens)
print("================Stemming===============")
e_words=["wait","waits","waiting","waited"]
ps=PorterStemmer()
for w in e_words:
	rootWord=ps.stem(w)
print(rootWord)	
print("================TF and IDF===============")
d0 = 'Geeks for geeks'
d1 = 'Geeks'
d2 = 'r2j'
string = [d0, d1, d2]
tfidf = TfidfVectorizer()
result = tfidf.fit_transform(string)
print(result)