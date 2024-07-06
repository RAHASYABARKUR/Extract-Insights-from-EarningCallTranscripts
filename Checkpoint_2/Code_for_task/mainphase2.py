
import os
import pandas as pd
from task1_abstractive_summary import summarizing

from task3_scoringcomplexity import scores
from task4_ner_testing import nercall




data_folder = os.getcwd()
print("The current working directory is ",os.getcwd())

print("Task 1: Summarization of the calls")
f_name = input("Enter the path to the text/transcript file which has to be summarized from the current directory: ")
file=os.path.join(data_folder,f_name)
summary=summarizing(f_name)
print(summary)
print("Task 3: Extracting / scoring linguistic complexity of the calls ")
scores(summary)

collist=['apple','amazon','boa']
print("Task 4: Entity extraction")
mainlist= pd.read_csv('main - Sheet1.csv', usecols=collist)
comp= mainlist['apple'].tolist()
#filename= comp[0]
model_save_name = comp[0]
col_list=[comp[1],comp[2],comp[3]]
col_rep_list=[comp[4],comp[5],comp[6],comp[7]]
prodfile=comp[8]
countryfile=comp[9]
pngname=comp[10]
mp4name=comp[11]
pc=comp[12]

nercall(model_save_name,col_list,col_rep_list,prodfile,countryfile,pngname,mp4name,int(pc))


