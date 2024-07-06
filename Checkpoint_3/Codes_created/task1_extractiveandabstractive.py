#GEID 1011139547
#Rahasya Barkur

from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import pdb
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import gensim
import os
import nltk
import string
from gensim.models import Word2Vec
from gensim.summarization.summarizer import summarize

max_len=50


pretrained_model = 'models//trained_models//newmodel.h5'
def summarizing(f_name):
    text=open(f_name,encoding="ISO_8859-1").read()
    text.encode('utf-8').strip()
    Sentences = sent_tokenize(text)
    words=[]
    for i in range(0,len(Sentences),1):
        #print(i,len(Sentences[i].split()))
        words.append(len(Sentences[i].split()))
    #print(words)
    refined_sentences=[]
    refined_text=""
    for i in range(0,len(Sentences),1):
        if words[i]<=50 and words[i]>=10:
            refined_sentences.append(Sentences[i])
            refined_text=refined_text+Sentences[i]+" "
            
    print("No.of sentences in the transcrpt you've chosen is : ",len(refined_sentences))
    r=float(input("Enter the ratio of the number of sentences you desire in your summary, desirable range is 0.4 to 0.6: "))
    summary=summarize(refined_text,ratio = r)
    refined_sentences=sent_tokenize(summary)
    print(len(refined_sentences),"sentences will be present in your summary")
    print("Compressing every sentence")
    embedding_size = 300
    print("This might take a while")
    model = Word2Vec(refined_sentences, min_count=1,size=300)
    model.intersect_word2vec_format(r'models\trained_models\GoogleNews-vectors-negative300.bin.gz',lockf=1.0,binary=True)
    modell=model.train(refined_sentences,total_examples=len(refined_sentences),epochs=20)
    model = gensim.models.KeyedVectors.load_word2vec_format(r'models\trained_models\GoogleNews-vectors-negative300.bin.gz', binary=True)



    decoder_input = np.zeros((len(refined_sentences),max_len,embedding_size))

    print("Loading Word2Vec done ...")
    for i in range(len(refined_sentences)):
        sentence_em = []
        words_in_sentences = refined_sentences[i].split()
        words_in_sentences = words_in_sentences[::-1]
        for j in range(max_len):
            word_em = np.zeros((1,embedding_size))
            if j < len(words_in_sentences):
                        try:
                            word_em = model[words_in_sentences[j].translate(str.maketrans('','',string.punctuation))]
                            word_em = word_em/(np.sqrt(np.linalg.norm(word_em)))
                            
                        except:
                            try:
                              yo=1
                              word_em = model1[words_in_sentences[j].translate(str.maketrans('','',string.punctuation))]
                              word_em = word_em/(np.sqrt(np.linalg.norm(word_em)))
                            except:
                              word_em=0
            decoder_input[i,j,:] = word_em
    #del modell
    #del model
    from keras.models import load_model
    model=load_model(pretrained_model)
    print("Predicting...")
    predictions = model.predict(decoder_input,batch_size=1,verbose=0)
    threshold = 0.5
    predictions[predictions>threshold] = 1
    predictions[predictions<=threshold] = 0
    texts=[]
    text=""
    prediction_sentence=[]
    for i in range(0,predictions.shape[0],1):    
        sentence = refined_sentences[i].split()
        pred_sen = []
        for j in range( 0,len(sentence),1):
            if predictions[i,j] == [1]:
                pred_sen.append(sentence[j])
                text=text+" " + sentence[j]
        text.capitalize()
        last_char = text[-1]
        if(last_char != "."):
            text=text + "."
        prediction_sentence.append(pred_sen)
    comment_words = '' 
    stopwords = list(STOPWORDS)
    freq=['quarter','year','will','you','us','thanks','years','going','know','product','services','time','new','march','billion','stores','weã','ve','think',
          'continue','end','point','re','seen','well','including','see','provide','quarterly'] 
    stopwords.extend(freq)  

    tokens = text.split() 
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
      tokens[i] = tokens[i].lower() 
    comment_words += " ".join(tokens)+" "
      
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0)
    plt.savefig("results//summaryresults//wordcloud.png")
    plt.show()
    
    return text

