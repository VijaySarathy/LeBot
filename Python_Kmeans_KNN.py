{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Text Classification and Clustering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import Libraries\n",
    "\n",
    "%config IPCompleter.greedy=True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.cluster import KMeans\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem.wordnet import WordNetLemmatizer\n",
    "import string\n",
    "import re\n",
    "from collections import Counter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "stop = set(stopwords.words('english'))\n",
    "exclude = set(string.punctuation)\n",
    "lemma = WordNetLemmatizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cleaning the text sentences so that punctuation marks, stop words & digits are removed\n",
    "\n",
    "def clean(doc):\n",
    "    stop_free = \" \".join([i for i in doc.lower().split() if i not in stop])\n",
    "    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)\n",
    "    normalized = \" \".join(lemma.lemmatize(word) for word in punc_free.split())\n",
    "    processed = re.sub(r\"\\d+\",\"\",normalized)\n",
    "    y = processed.split()\n",
    "    return y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "#K-Neighbor and K-Means is performed on 3 different types of data\n",
    "\n",
    "path = \"/Users/Vijay/Great Lakes/Learnings/Text Analytics/Sentence.txt\"\n",
    "\n",
    "train_clean_sentence = []\n",
    "fp = open(path,'r')\n",
    "for line in fp:\n",
    "    line = line.strip()\n",
    "    cleaned = clean(line)\n",
    "    cleaned = ' '.join(cleaned)\n",
    "    train_clean_sentence.append(cleaned)\n",
    "    \n",
    "vectorizer = TfidfVectorizer(stop_words = 'english')\n",
    "X=vectorizer.fit_transform(train_clean_sentence)\n",
    "\n",
    "\n",
    "# Creating true labels for 30 training sentences\n",
    "y_train = np.zeros(30)\n",
    "y_train[10:20] = 1\n",
    "y_train[20:30] = 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "           metric_params=None, n_jobs=None, n_neighbors=5, p=2,\n",
       "           weights='uniform')"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Classification using K-nn\n",
    "modelknn = KNeighborsClassifier(n_neighbors = 5)\n",
    "modelknn.fit(X,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=200,\n",
       "    n_clusters=3, n_init=100, n_jobs=None, precompute_distances='auto',\n",
       "    random_state=None, tol=0.0001, verbose=0)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#clustering using k-Means\n",
    "modelkmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter=200, n_init=100)\n",
    "modelkmeans.fit(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Predit the Test Data\n",
    "test_sentence = [\"Chemical compounds are used for preparing bombs based on some reactions\",\\\n",
    "\"Cricket is a boring game where the batsman only enjoys the game and bowlers suffer\",\\\n",
    "\"Machine learning is a area of Artificial intelligence\"]\n",
    "\n",
    "test_clean_sentence = []\n",
    "for test in test_sentence:\n",
    "    cleaned_test = clean(test)\n",
    "    cleaned = ' '.join(cleaned_test)\n",
    "    cleaned = re.sub(r\"\\d+\",\"\",cleaned)\n",
    "    test_clean_sentence.append(cleaned)\n",
    "    \n",
    "    Test = vectorizer.transform(test_clean_sentence)\n",
    "    \n",
    "    true_test_labels = ['Cricket', 'AI','Chemistry']\n",
    "    Perdict_label_Knn = modelknn.predict(Test)\n",
    "    Predict_label_Kmeans = modelkmeans.predict(Test)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 1, 0], dtype=int32)"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Predict_label_Kmeans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2., 0., 1.])"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Perdict_label_Knn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
