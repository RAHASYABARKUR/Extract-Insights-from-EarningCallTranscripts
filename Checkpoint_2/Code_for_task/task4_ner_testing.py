#GEID 1011139585
#Annlin Chako
# install barchart race
#pip install bar_chart_race

# install files

import spacy
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import bar_chart_race as bcr
import random
import matplotlib.cm as cm
import matplotlib as mpl

# load model
def load_model(path,path_text,name,col_list,i,quarter,prodfile,countryfile):
  text=open(path_text,encoding="ISO-8859-1").read()
  nlp_model=spacy.load(path)
  doc=nlp_model(text)
  updates_ents =[(x.text,x.label_) for x in doc.ents]
  df=pd.DataFrame(updates_ents,columns=['word','label']) 
  df.to_csv(name, index = False, header=True)
  rep= pd.read_csv('replace - Sheet1.csv', usecols=col_list)
#product list
  df_prod=df.loc[df['label'] == 'PRODUCT']
  prod_list= rep[col_list[2]].tolist()
  prod_replist=rep[col_list[3]].tolist()
  for j in range(len(prod_list)):
    df_prod=df_prod.replace(prod_list[j],prod_replist[j])
  dff_prod=pd.DataFrame(df_prod['word'].value_counts())
  new_prod=dff_prod.transpose()
  if(i==0):
    new_prod.insert(0,'quarter',quarter)
    new_prod.to_csv(prodfile, index =False, header=True)
  else:
    new_prod.insert(0,'quarter',quarter)
    new_prod.to_csv(r'prod2.csv', index =False, header=True)
    data = pd.read_csv(prodfile) 
    data2 = pd.read_csv("prod2.csv")    
    for col in data2.columns: 
      if col not in data.columns:
        data[col] = np.nan
      data.at[20,col] = data2[col].values[0]
    data=data.replace(np.nan,0)
    data.to_csv(prodfile, index =False, header=True)

#country list
  df_con=df.loc[df['label'] == 'COUNTRY']
  country_list= rep['Country'].tolist()
  country_replist=rep['Country replace'].tolist()
  for j in range(len(country_list)):
     df_con=df_con.replace(country_list[j],country_replist[j])
  dff_con=pd.DataFrame(df_con['word'].value_counts())
  new_con=dff_con.transpose()
  if(i==0):
    new_con.insert(0,'quarter',quarter)
    new_con.to_csv(countryfile, index =False, header=True)
  else:
    new_con.insert(0,'quarter',quarter)
    new_con.to_csv(r'country2.csv', index =False, header=True)
    data = pd.read_csv(countryfile) 
    data2 = pd.read_csv("country2.csv")    
    for col in data2.columns: 
      if col not in data.columns:
        data[col] = np.nan
      data.at[20,col] = data2[col].values[0]
    data=data.replace(np.nan,0)
    data.to_csv(countryfile, index =False, header=True)


def nercall(model_save_name,col_list,col_rep_list,prodfile,countryfile,pngname,mp4name,pc):
  path=F"/models/{model_save_name}" 
  filelist= pd.read_csv('path.csv', usecols=col_list)
  path_text= filelist[col_list[0]].tolist()
  name= filelist[col_list[1]].tolist()
  quarter= filelist[col_list[2]].tolist()
  for i in range(len(path_text)):
    load_model(path,path_text[i],name[i],col_rep_list,i,quarter[i],prodfile,countryfile) 

# Scatter plot for country 
  df = pd.read_csv(countryfile)
  df.loc['Column_Total']= df.sum(numeric_only=True, axis=0)
  df= df.sort_values(by ='Column_Total', axis=1,ascending=False)
  cols = list(df.columns)
  cols = [cols[-1]] + cols[:-1]
  df = df[cols]
  df.drop(df.iloc[:,9:], inplace=True, axis=1)
  df=df[:-1]
  country_list=list(df)
  del(country_list[0])
  colors = iter(cm.rainbow(np.linspace(0, 1, len(country_list))))
  fig, ax = plt.subplots(figsize=(20, 10))
  for i in range(len(country_list)):
    country=country_list[i]
    df.plot(kind='scatter',x='quarter',y=country,color=next(colors),ax=ax,label=country,s=250)
  plt.xlabel('Quarter')
  plt.ylabel('Count')
  plt.title('Scatter plot indicatiing the frequency of country during call')
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=.1)
  plt.savefig(pngname)

#bar_chart_race on products
  df_prod = pd.read_csv(prodfile)
  df_prod.loc['Column_Total']= df_prod.sum(numeric_only=True, axis=0)
  df_prod= df_prod.sort_values(by ='Column_Total', axis=1,ascending=False)
  cols = list(df_prod.columns)
  cols = [cols[-1]] + cols[:-1]
  df_prod = df_prod[cols]
  df_prod.drop(df_prod.iloc[:,pc:], inplace=True, axis=1)
  df_prod=df_prod[:-1]
  df_prod.to_csv(r'prod_plot.csv', index =False, header=True)
  df = pd.read_csv('prod_plot.csv', index_col='quarter', 
                  parse_dates=['quarter'])
  bcr.bar_chart_race(df=df, steps_per_period=30, fixed_order=True, period_length=8000,orientation='v',filename=mp4name)
