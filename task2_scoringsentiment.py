#GEID 1011139585 & 1011139542
#Annlin Chako & Kancharapu Anil Kumar

import string
import nltk

import matplotlib.pyplot as plt
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def sentiment(value):

     if value>0:
        emotion.append("positive")  # positive
     elif value==0:
         emotion.append("neutral")  # neutral
     else:
        emotion.append("negative")  #negative


def sentiment_analyze(text_sentiment):
    score=SentimentIntensityAnalyzer().polarity_scores(text_sentiment)
    sentiment(score['compound'])



emotion=[]
def sentiment_analysis(text):
    te= nltk.tokenize.sent_tokenize(text)
    #text_tokenize = word_tokenize(text_plain, "english")
    

    for sen in te:
     text_lower = sen.lower()
    # Text without special characters.
     text_plain = text_lower.translate(str.maketrans('', '', string.punctuation))
     sentiment_analyze(text_plain)

    y = Counter(emotion)
    fig, ax1 = plt.subplots()
    ax1.bar(y.keys(), y.values())
    fig.autofmt_xdate()
    plt.savefig("analysis.png")
    plt.show()
#te=open("textfiles/apple//apple-inc-aapl-q1-2020-earnings-call-transcript.txt",encoding="utf-8").read()

