#GEID 1011139585
#Annlin Chako
import spacy
import en_core_web_sm
import pandas as pd
from pandas import DataFrame
from wordcloud import WordCloud,ImageColorGenerator
from spacy import displacy
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt
from spacy.tokens import Span
from spacy.tokens import Doc
from spacy.pipeline import EntityRuler
nlp = spacy.load('en_core_web_sm')



def add(colname,ctgname,path,rulername,a,s):
  s[a]=EntityRuler(nlp, overwrite_ents=True)
  col_list = [colname]
  ner= pd.read_csv(path, usecols=col_list)
  
  ner=ner.dropna()
  Prod_list= ner[colname].tolist()
  for f in Prod_list:
     s[a].add_patterns([{"label": ctgname, "pattern": f}])
  s[a].name = rulername
  nlp.add_pipe(s[a])
  
#adding list
def ner(text,path,finance):
  text=text.lower()
  s=['a','b','c']
  #print(finance)
  
  if(finance==0):
    add("PRODUCT","PRODUCT",path,"rulerprod",0,s)
  else:
    add("FINPRODUCT","PRODUCT",path,"rulerprod",0,s)
  add("DISEASE","DISEASE",path,"rulerdis",1,s)
  add("ORG","ORG",path,'rulerorg',2,s)

  #NER of DOC
  doc = nlp(text)
  updates_ents =[(x.text,x.label_) for x in doc.ents]
  df=pd.DataFrame(updates_ents,columns=['word','label']) 
  df.to_csv(r'export_df.csv', index = False, header=True)
  list_of_sents=[nlp(sent.text) for sent in doc.sents]
  list_of_ner=[doc for doc in list_of_sents if doc.ents]

  #bar plot
  select=df.loc[df['label'] == 'PRODUCT']
  dff=pd.DataFrame(select['word'].value_counts())
  dff.plot(kind='bar')
  plt.show()
  text=select['word'].value_counts()

  #gpe
  select=df.loc[df['label'] == 'GPE']
  dff=pd.DataFrame(select['word'].value_counts())
  dff.drop(dff[dff['word'] < 2].index, inplace = True) 
  dff.plot(kind='bar')
  #plt.savefig("gpe.png")
  plt.show()


