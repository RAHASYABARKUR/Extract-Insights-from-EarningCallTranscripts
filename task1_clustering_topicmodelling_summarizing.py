#GEID 1011139547
#Rahasya Barkur

from gensim.models import Word2Vec
import nltk
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import gensim
spacy.load('en')
from gensim import corpora
import pickle
from spacy.lang.en import English
parser = English()
import matplotlib.pyplot as plt
import pytextrank
from gensim.summarization.summarizer import summarize
import os
import warnings
warnings.filterwarnings("ignore")
en_stop = set(nltk.corpus.stopwords.words('english'))


def get_lemma(word):
    return WordNetLemmatizer().lemmatize(word)
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    
    return tokens
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
def vectorise(sent,m):
    vec=[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                 vec=m[w]
            else:
                vec=np.add(vec,m[w])
            numw+=1
        except:
            pass
    return np.asarray(vec)/numw
                
def find_topics(text,Sentences):
    lda_tokens = tokenize(text)
    en_stop = set(nltk.corpus.stopwords.words('english'))
    text_data = []
    x=[]
    i=0
    for sentence in Sentences:
        tokens = prepare_text_for_lda(sentence)
        i=i+len(tokens)
        text_data.append(tokens)
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    NUM_TOPICS = 1
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    topics = ldamodel.print_topics(num_words=1)
    return topics

def summarizing(f_name):
    text=open(f_name,encoding="utf-8").read()
    Sentences = sent_tokenize(text)
    x=[]
    for sentence in Sentences:
        x.append(tokenize(sentence))
        m=Word2Vec(x,min_count = 1)
    l=[]
    print("Vectorising.......")
    for i in x:
        l.append(vectorise(i,m))
    
    X=np.array(l)
    print("Clustering.......")
    clf = KMeans(n_clusters  = 2,init = 'k-means++')
    labels = clf.fit_predict(X)
    cluster0=[]
    cluster1=[]
    text0=''
    text1=''
    
    for i,label in enumerate(labels):
        if label==0:
            cluster0.append(Sentences[i])
            text0=text0+" " +Sentences[i]
        else:
            cluster1.append(Sentences[i])
            text1=text1+" " +Sentences[i]
    print("Finding Topics related to the clusters........")
    t0=find_topics(text0,cluster0)
    t1=find_topics(text1,cluster1)
    Topic0 = (t0[0][1].split('*'))[1]
    Topic1 = (t1[0][1].split('*'))[1]
    print("Summarizing ")
    
    s0=summarize(text0,ratio=0.1)+"\n"
    
    
    s1=summarize(text1,ratio=0.1)+"\n"
    return(Topic0,s0,Topic1,s1)
#data_folder = "C:\\Users\\rahas\\OneDrive\\Documents\\Citi\\final\\textfiles"

#f_name = input("Enter the name of the text file you want to find summary of")
#f_name="apple.txt"

#file=os.path.join(data_folder,f_name)


#summarizing(file)    
   
    

