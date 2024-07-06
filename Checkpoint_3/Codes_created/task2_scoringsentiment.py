#K. Anil Kumar
#GEID :1011139542

#Calculating Sentiment with the trained model
import pandas as pd
import numpy as np
import re
import csv
import nltk
from nltk.corpus import stopwords

from numpy import asarray
from numpy import zeros

#Keras will function in python version<=3.6
from numpy import array
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

model=load_model('models//trained_models//sentiment.h5')
#print(data['Summary'][0])
#a=data.shape
#rows=a[0]
maxlen = 100
# print(rows)
# print(data.head())

def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]

#Calculating Sentiment with the trained model

def scoring_sentiments(summary):
    text = summary
    for i in range(1, len(text), 1):
        if (text[i - 1] == "." and (text[i] >= 'A' and text[i] <= 'Z')):
            text = replacer(text, ". ", i - 1)
    # print(text)
    instance = text
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(instance)
    instance = tokenizer.texts_to_sequences(instance)
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    Sentiment = model.predict(instance)
    return Sentiment[0][0]


