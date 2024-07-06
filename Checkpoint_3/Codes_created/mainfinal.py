
import os
import pandas as pd
from task1_extractiveandabstractive import summarizing

from task3_scoringcomplexity import scores
from task2_scoringsentiment import scoring_sentiments
from task4_NER import nercall



data_folder = os.getcwd()
print("The current working directory is ",os.getcwd())
f_name = input("Enter the path to the text/transcript file which has to be summarized from the current directory: ")

print("Task 1: Summarization of the calls")

file=os.path.join(data_folder,f_name)

summary=summarizing(f_name)
#print(summary)
with open('results//summaryresults//summary_of'+ f_name.split('/')[2],'w',encoding="ISO_8859-1") as a_writer:
    a_writer.write(summary)

print("Task 2: Sentimental Analysis")
sentiments=scoring_sentiments(summary)
print("The sentimental score is:",sentiments)
print("Task 3: Extracting / scoring linguistic complexity of the calls ")
scores(summary)
print("Task 4: NER")

name=f_name.split('/')[1]
mainlist= pd.read_csv('csvfiles//main - Sheet1.csv')
comp= mainlist[name].tolist()
model_save_name = comp[0]
col_list=[comp[1],comp[2],comp[3]]
col_rep_list=[comp[4],comp[5],comp[6],comp[7]]
prodfile= comp[8]
countryfile=comp[9]
pngname=comp[10]
mp4name=comp[11]
pc=comp[12]
quarter=(f_name.split('/')[2]).split('q')[1]
quarter=quarter[0]
year=(f_name.split('/')[2]).split('20')[1]

if (year[0:2]):
    year="20"+year[0:2]
else:
    year = "2020"
quarter_call='q'+quarter+'_'+year
nercall(model_save_name,col_list,col_rep_list,prodfile,countryfile,pngname,mp4name,int(pc),quarter_call)

