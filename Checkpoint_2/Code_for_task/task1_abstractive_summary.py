#GEID 1011139547
#Rahasya Barkur

from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import pdb
from nltk.tokenize import sent_tokenize

import gensim
import os
import nltk
import string
from gensim.models import Word2Vec
nltk.download('punkt')

max_len=40


pretrained_model = 'trained_models//try60.h5'
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
    for i in range(0,len(Sentences),1):
        if words[i]<=40 and words[i]>=15:
            refined_sentences.append(Sentences[i])
    print(len(refined_sentences))
    embedding_size = 300
    model = Word2Vec(refined_sentences, min_count=1,size=300)
    model.intersect_word2vec_format(r'trained_models\GoogleNews-vectors-negative300.bin.gz',lockf=1.0,binary=True)
    modell=model.train(refined_sentences,total_examples=len(refined_sentences),epochs=20)
    model = gensim.models.KeyedVectors.load_word2vec_format(r'trained_models\GoogleNews-vectors-negative300.bin.gz', binary=True)



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
        last_char = text[-1]
        if(last_char != "."):
            text=text + "."
        prediction_sentence.append(pred_sen)
    return text


