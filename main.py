#GEID 1011139547
#Rahasya Barkur
import os

from task1_clustering_topicmodelling_summarizing import summarizing
from task2_scoringsentiment import sentiment_analysis
from task3_scoringcomplexity import scores
from task4_entityextraction import ner
from task6_clusteringtradeideas import clustering



data_folder = os.getcwd()
print("The current working directory is ",os.getcwd())
print("Task 1: Summarization of the calls")
f_name = input("Enter the path to the text file which has to be summarized from the current directory: ")
file=os.path.join(data_folder,f_name)
text=open(f_name,encoding="utf-8").read()

t0,s0,t1,s1=summarizing(file)
print("The summary is generated on the topics ",t0,"and",t1)
summary = s0 + s1
print("Task 2: Extracting / scoring sentiment of the calls ")
sentiment_analysis(summary)

print("Task 3: Extracting / scoring linguistic complexity of the calls ")
scores(text)
print("Task 4: Entity extraction")
path=input("Enter the path to the csv file which contains the list of ner from the current directory: ")

finance=int(input("Enter '0' if its a non - finance company or else enter '1': "))
ner(text,path,finance)

print("Task 6: Clustering/nearest neighbours of trade ideas/companies with similarities in a quarter")
clustering(text)
