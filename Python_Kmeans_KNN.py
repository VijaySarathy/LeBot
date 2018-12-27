#Import Libraries
#%config IPCompleter.greedy=True

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
from collections import Counter

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Cleaning the text sentences so that punctuation marks, stop words & digits are removed

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y


#K-Neighbor and K-Means is performed on 3 different types of data

path = "/Users/Vijay/Great Lakes/Learnings/Text Analytics/Sentence.txt"

train_clean_sentence = []
fp = open(path,'r')
for line in fp:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentence.append(cleaned)
    
vectorizer = TfidfVectorizer(stop_words = 'english')
X=vectorizer.fit_transform(train_clean_sentence)


# Creating true labels for 30 training sentences
y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2

#Classification using K-nn
modelknn = KNeighborsClassifier(n_neighbors = 5)
modelknn.fit(X,y_train)

#clustering using k-Means
modelkmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter=200, n_init=100)
modelkmeans.fit(X)

#Predit the Test Data
test_sentence = ["Chemical compounds are used for preparing bombs based on some reactions",\
"Cricket is a boring game where the batsman only enjoys the game and bowlers suffer",\
"Machine learning is a area of Artificial intelligence"]

test_clean_sentence = []
for test in test_sentence:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+","",cleaned)
    test_clean_sentence.append(cleaned)
    
    Test = vectorizer.transform(test_clean_sentence)
    
    true_test_labels = ['Cricket', 'AI','Chemistry']
    Perdict_label_Knn = modelknn.predict(Test)
    Predict_label_Kmeans = modelkmeans.predict(Test)
    

Predict_label_Kmeans
Predict_label_Kmeans

