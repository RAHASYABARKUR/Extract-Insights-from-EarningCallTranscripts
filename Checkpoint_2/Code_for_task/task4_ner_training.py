#GEID 1011139585
#Annlin Chako
# Below code is used to train the ner model, training needs to be done for each company
# as the product data set vary from company to company

#install files
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
import random

#load traindata file
model_name='applfile.data'   # for a specific company, use respective datafile from 'data file' folder.
with open(model_name, 'rb') as filehandle:
    # read the data as binary data stream
    to_train= pickle.load(filehandle)


#train data
    
def train_spacy(data,iterations):
    TRAIN_DATA = data
    print(TRAIN_DATA)
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations['entities']:
          ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp
prdnlp = train_spacy(to_train, 20)

#save the model to disk
model_save_name = 'appl.pt'  #input model name, given is of apple.
prdnlp.to_disk(model_save_name)
