#GEID 1011139585
#Annlin Chako
# Below is the code to pre- process the data required for training ner model.
# Around 40 transcripts for each company has been used to train the model.
# All the transcripts have been extracted from seekingalpha site.
#https://seekingalpha.com/article/33539-apple-f2q07-qtr-end-3-31-07-earnings-call-transcript
# the above url is used to extract 2007 Q-2 transcript.
#data file created for each company which includes all processed training data for specifc company
#has been included in the submission.

import  spacy
import en_core_web_sm
import pandas as pd
from pandas import DataFrame
from spacy import displacy
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt
from spacy.tokens import Span
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler
import nltk
import pickle
from csv import writer
 
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
def add(colname,ctgname,path,rulername,a,s,nlp):
  s[a]=EntityRuler(nlp, overwrite_ents=True)
  col_list = [colname]
  ner= pd.read_csv(path, usecols=col_list)
  ner=ner.dropna()
  Prod_list= ner[colname].tolist()
  for f in Prod_list:
     s[a].add_patterns([{"label": ctgname, "pattern": f}])
  s[a].name = rulername
  nlp.add_pipe(s[a])


def prept_data(path,finance,begin):
  nlp = spacy.load('en')
  #read from text file
  text=open(path,encoding="ISO-8859-1").read()
  s=['a','b','c','d','e','f']
  #convert to lower
  text=text.lower()
   
  #adding list
  path="list for ner - Sheet1.csv"
  if(finance==0):
    add("ORG","ORG",path,'rulerorg',0,s,nlp)
    add("APPLE","PRODUCT",path,"rulerprod",1,s,nlp)
  else:
    add("FINPRODUCT","PRODUCT",path,"rulerprod",0,s,nlp)
    add("ORG","ORG",path,'rulerorg',0,s,nlp)
  add("DISEASE","DISEASE",path,"rulerdis",2,s,nlp)
  add("QUARTER","QUARTER",path,"rulerquat",3,s,nlp)
  add("PERSON","PERSON",path,"rulerpers",4,s,nlp)
  add("COUNTRY","COUNTRY",path,"rulercountry",5,s,nlp)
  doc = nlp(text)
  updates_ents =[(x.start_char,x.end_char,x.label_) for x in doc.ents]
  return(text,updates_ents)
finance=0
begin=0
to_train=[]
#path=[] include the list of path to all transcripts required for training
for i in range(len(path)):
  text,update_ents=prept_data(path[i],int(finance),int(i))
  to_train.append((text,{'entities':(update_ents)}))
  print(len(to_train))
with open('/content/applfile.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(to_train[:], filehandle)









    
    pickle.dump(to_train[:], filehandle)
