#GEID 1011139542
#Kancharapu Anil Kumar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import os
from spacy.lang.en import English
import spacy
import en_core_web_sm
import pandas as pd
from pandas import DataFrame
from spacy import displacy
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt
from spacy.tokens import Span
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler

parser = English()
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y
#text=open( "textfiles\apple.txt",encoding="utf-8").read()
def clustering(text):
    train_clean_sentences = []

    te= nltk.tokenize.sent_tokenize(text)
    for sentence in te:

        sentence = sentence.strip()
        cleaned = clean(sentence)
        cleaned = ' '.join(cleaned)
        train_clean_sentences.append(cleaned)
    #print(train_clean_sentences)
    wcss=[]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(train_clean_sentences)

    modelkmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)
    modelkmeans.fit(X)
    clf = KMeans(n_clusters = 3,init = 'k-means++')
    labels = clf.fit_predict(X)

    cluster0=[]
    cluster1=[]
    cluster2=[]
    text0=''
    text1=''
    text3=''
    for i,label in enumerate(labels):
        if label==0:
            cluster0.append(train_clean_sentences[i])
            text0=text0+ train_clean_sentences[i]
        elif label==1:
            cluster1.append(train_clean_sentences[i])
            text1=text1+ train_clean_sentences[i]
        else:
            cluster2.append(train_clean_sentences)
            text3=text3+ train_clean_sentences[i]

    print(text0 +"\n")
    print(text1+"\n")
    print(text3)
