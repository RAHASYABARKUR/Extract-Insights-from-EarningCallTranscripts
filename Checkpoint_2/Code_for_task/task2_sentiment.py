#K. Anil Kumar
#GEID No 1011139542

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
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

#Dataset Link : : https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

movie_reviews = pd.read_csv("IMDB Dataset.csv")

movie_reviews.isnull().values.any()

movie_reviews.shape

#print(movie_reviews.head())

'''
Graph to check the number of positive and negative sentiment in the review
import seaborn as sns

sns.countplot(x='sentiment', data=movie_reviews)'''

def preprocess_text(sen):

    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))

print(X[3])

y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
#print(vocab_size)

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#GloVe file is used for word to vector
#Glove file link: https://www.kaggle.com/terenceliu4444/data

embeddings_dictionary = dict()
glove_file = open(r"C:\Users\anilkumar\PycharmProjects\example\glove.txt", encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
from keras.layers.recurrent import LSTM

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#print(model.summary())

history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

# print("Test Score:", score[0])
# print("Test Accuracy:", score[1])

#Plotting the model accuracy and model loss
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


data=pd.read_csv(r"C:\Users\anilkumar\deutschebank.csv", sep=',', encoding='latin-1')

#print(data['Summary'][0])
a=data.shape
rows=a[0]
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
Sentiment = []
for i in range(0, rows):
    text = data['Summary'][i]
    for i in range(1, len(text), 1):
        if (text[i - 1] == "." and (text[i] >= 'A' and text[i] <= 'Z')):
            text = replacer(text, ". ", i - 1)
    # print(text)
    instance = [text]
    instance = tokenizer.texts_to_sequences(instance)
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    a = model.predict(instance)
    Sentiment.append(a[0][0])
print(Sentiment)


list1= data['ï»¿Company']
list2= data['Quadrant']
list3= data['Year']
list4= data['Summary']
list5= data['Linguistic Complexity(Flesch Scores)']
list6= Sentiment

rows = zip(list1,list2,list3,list4,list5,list6)

#Writing all columns into csv file

with open('deutschebank.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(['Company', 'Quandrant', 'Year', 'Summary', 'Linguistic Complexity(Flesch Scores)', 'Sentiment'])
    for row in rows:
        writer.writerow(row)
