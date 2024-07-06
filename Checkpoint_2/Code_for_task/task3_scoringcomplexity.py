#GEID 1011139547
#rahasya barkur
import textstat

def scores(text):
    dale_chall=textstat.dale_chall_readability_score(text)
    print("Dale-Chall Score: ",dale_chall)
    flesch=textstat.flesch_reading_ease(text)
    print("Flesch Score: ",flesch)
